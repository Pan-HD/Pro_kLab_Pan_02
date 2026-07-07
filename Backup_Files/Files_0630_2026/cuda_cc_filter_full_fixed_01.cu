#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "../../CommonComponents/SegmentationConfig.h"

struct CCRegionStat
{
    int area;
    int xmin;
    int xmax;
    int ymin;
    int ymax;
};

__global__ void makeNppForegroundMaskKernel(
    const unsigned char* src,
    size_t srcStep,
    int width,
    int height,
    unsigned char* dst,
    size_t dstStep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned char v = src[y * srcStep + x];

#if FG_PIXEL == 255
    // NPP labels non-zero pixels. In white-foreground mode, keep all non-zero
    // pixels as foreground, matching the old behavior but forcing a binary mask.
    dst[y * dstStep + x] = (v != 0) ? 255 : 0;
#else
    // NPP labels non-zero pixels. In black-foreground mode, invert the mask so
    // the logical foreground becomes non-zero before NPP connected components.
    dst[y * dstStep + x] = (v == 0) ? 255 : 0;
#endif
}

__global__ void initCCStatsKernel(CCRegionStat* stats, int labelLimit)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > labelLimit) return;

    stats[id].area = 0;
    stats[id].xmin = INT_MAX;
    stats[id].xmax = -1;
    stats[id].ymin = INT_MAX;
    stats[id].ymax = -1;
}

__global__ void computeCCStatsKernel(
    const Npp32u* label,
    int width,
    int height,
    int labelLimit,
    CCRegionStat* stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Npp32u id = label[idx];

    // Guard against any uncompressed or invalid label value. This should not
    // happen after nppiCompressMarkerLabelsUF, but without the guard a single
    // stray label can write out of bounds and make fitness non-deterministic.
    if (id == 0 || id > (Npp32u)labelLimit) return;

    atomicAdd(&stats[id].area, 1);
    atomicMin(&stats[id].xmin, x);
    atomicMax(&stats[id].xmax, x);
    atomicMin(&stats[id].ymin, y);
    atomicMax(&stats[id].ymax, y);
}

__global__ void computeCCRemoveMaskKernel(
    const CCRegionStat* stats,
    int labelLimit,
    int minArea,
    float aspectRange,
    unsigned char* removeMask)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > labelLimit) return;

    if (id == 0)
    {
        removeMask[id] = 1;
        return;
    }

    CCRegionStat s = stats[id];

    int w = s.xmax - s.xmin + 1;
    int h = s.ymax - s.ymin + 1;

    unsigned char remove = 0;

    if (s.area <= 0 || w <= 0 || h <= 0)
    {
        remove = 1;
    }
    else
    {
        float aspect = (float)w / (float)h;

        if (s.area < minArea)
            remove = 1;

        if (aspectRange > 0.0f)
        {
            float minAspect = 1.0f / aspectRange;
            float maxAspect = aspectRange;

            if (aspect < minAspect || aspect > maxAspect)
                remove = 1;
        }
    }

    removeMask[id] = remove;
}

__global__ void generateCCFilteredBinaryKernel(
    const Npp32u* label,
    const unsigned char* removeMask,
    int width,
    int height,
    int labelLimit,
    unsigned char* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Npp32u id = label[idx];

    if (id == 0 || id > (Npp32u)labelLimit || removeMask[id])
        output[idx] = BG_PIXEL;
    else
        output[idx] = FG_PIXEL;
}

#define CUDA_CHECK_LOCAL(call)                                           \
do {                                                                     \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        throw std::runtime_error(std::string("CUDA Error: ") +            \
            cudaGetErrorString(err));                                     \
    }                                                                    \
} while (0)

#define NPP_CHECK_LOCAL(call)                                             \
do {                                                                     \
    NppStatus st = (call);                                                \
    if (st != NPP_SUCCESS) {                                              \
        throw std::runtime_error("NPP Error, status = " +                 \
            std::to_string((int)st));                                     \
    }                                                                    \
} while (0)

cv::cuda::GpuMat executeCCFilterCUDA(
    const cv::cuda::GpuMat& src,
    const std::vector<double>& params)
{
    CV_Assert(!src.empty());
    CV_Assert(src.type() == CV_8UC1);

    int width = src.cols;
    int height = src.rows;
    int pixelCount = width * height;

    int minArea =
        params.size() > 0
        ? std::max<int>(1, static_cast<int>(params[0]))
        : 1;
    float aspectRange =
        params.size() > 1
        ? std::max<float>(0.0f, static_cast<float>(params[1]))
        : 0.0f;

    NppiSize roi;
    roi.width = width;
    roi.height = height;

    int srcStep = width * sizeof(Npp8u);
    int labelStep = width * sizeof(Npp32u);

    Npp8u* dSrcLinear = nullptr;
    Npp32u* dLabel = nullptr;
    Npp8u* dOutput = nullptr;

    Npp8u* dLabelBuffer = nullptr;
    Npp8u* dCompressBuffer = nullptr;
    CCRegionStat* dStats = nullptr;
    unsigned char* dRemoveMask = nullptr;

    try
    {
        CUDA_CHECK_LOCAL(cudaMalloc(&dSrcLinear, pixelCount * sizeof(Npp8u)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dLabel, pixelCount * sizeof(Npp32u)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dOutput, pixelCount * sizeof(Npp8u)));

        dim3 block2D(16, 16);
        dim3 grid2D(
            (width + block2D.x - 1) / block2D.x,
            (height + block2D.y - 1) / block2D.y);

        makeNppForegroundMaskKernel<<<grid2D, block2D>>>(
            src.ptr<Npp8u>(),
            src.step,
            width,
            height,
            dSrcLinear,
            srcStep);
        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        CUDA_CHECK_LOCAL(cudaMemset(dLabel, 0, pixelCount * sizeof(Npp32u)));

        int labelBufferSize = 0;
        NPP_CHECK_LOCAL(nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &labelBufferSize));

        CUDA_CHECK_LOCAL(cudaMalloc(&dLabelBuffer, labelBufferSize));

        NPP_CHECK_LOCAL(nppiLabelMarkersUF_8u32u_C1R(
            dSrcLinear,
            srcStep,
            dLabel,
            labelStep,
            roi,
            nppiNormInf,
            dLabelBuffer));

        CUDA_CHECK_LOCAL(cudaFree(dLabelBuffer));
        dLabelBuffer = nullptr;

        int nStartingNumber = pixelCount;
        int compressBufferSize = 0;

        NPP_CHECK_LOCAL(nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
            nStartingNumber,
            &compressBufferSize));

        CUDA_CHECK_LOCAL(cudaMalloc(&dCompressBuffer, compressBufferSize));

        int maxLabel = 0;

        NPP_CHECK_LOCAL(nppiCompressMarkerLabelsUF_32u_C1IR(
            dLabel,
            labelStep,
            roi,
            nStartingNumber,
            &maxLabel,
            dCompressBuffer));

        CUDA_CHECK_LOCAL(cudaFree(dCompressBuffer));
        dCompressBuffer = nullptr;

        if (maxLabel <= 0)
        {
            cv::cuda::GpuMat out(height, width, CV_8UC1);
            out.setTo(cv::Scalar(BG_PIXEL));
            CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

            cudaFree(dSrcLinear);
            cudaFree(dLabel);
            cudaFree(dOutput);

            return out;
        }

        // Allocate by the theoretical label limit instead of maxLabel. This is a
        // defensive fix: if an NPP/compression edge case leaves a label larger
        // than maxLabel, the old code would access dStats/removeMask out of bounds.
        int labelLimit = nStartingNumber;

        CUDA_CHECK_LOCAL(cudaMalloc(&dStats, (labelLimit + 1) * sizeof(CCRegionStat)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dRemoveMask, (labelLimit + 1) * sizeof(unsigned char)));

        int block1D = 256;
        int grid1D = (labelLimit + 1 + block1D - 1) / block1D;

        initCCStatsKernel<<<grid1D, block1D>>>(dStats, labelLimit);
        CUDA_CHECK_LOCAL(cudaGetLastError());

        computeCCStatsKernel<<<grid2D, block2D>>>(
            dLabel,
            width,
            height,
            labelLimit,
            dStats);
        CUDA_CHECK_LOCAL(cudaGetLastError());

        computeCCRemoveMaskKernel<<<grid1D, block1D>>>(
            dStats,
            labelLimit,
            minArea,
            aspectRange,
            dRemoveMask);
        CUDA_CHECK_LOCAL(cudaGetLastError());

        generateCCFilteredBinaryKernel<<<grid2D, block2D>>>(
            dLabel,
            dRemoveMask,
            width,
            height,
            labelLimit,
            dOutput);
        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        cv::cuda::GpuMat out(height, width, CV_8UC1);

        CUDA_CHECK_LOCAL(cudaMemcpy2D(
            out.ptr<Npp8u>(),
            out.step,
            dOutput,
            width * sizeof(Npp8u),
            width * sizeof(Npp8u),
            height,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        cudaFree(dSrcLinear);
        cudaFree(dLabel);
        cudaFree(dOutput);
        cudaFree(dStats);
        cudaFree(dRemoveMask);

        return out;
    }
    catch (...)
    {
        if (dSrcLinear) cudaFree(dSrcLinear);
        if (dLabel) cudaFree(dLabel);
        if (dOutput) cudaFree(dOutput);
        if (dLabelBuffer) cudaFree(dLabelBuffer);
        if (dCompressBuffer) cudaFree(dCompressBuffer);
        if (dStats) cudaFree(dStats);
        if (dRemoveMask) cudaFree(dRemoveMask);
        throw;
    }
}
