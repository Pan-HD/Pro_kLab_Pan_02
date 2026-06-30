#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "../CommonComponents/SegmentationConfig.h"

struct CCRegionStat
{
    int area;
    int xmin;
    int xmax;
    int ymin;
    int ymax;
};

__global__ void initCCStatsKernel(CCRegionStat* stats, int maxLabel)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > maxLabel) return;

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
    CCRegionStat* stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Npp32u id = label[idx];

    if (id == 0) return;

    atomicAdd(&stats[id].area, 1);
    atomicMin(&stats[id].xmin, x);
    atomicMax(&stats[id].xmax, x);
    atomicMin(&stats[id].ymin, y);
    atomicMax(&stats[id].ymax, y);
}

__global__ void computeCCRemoveMaskKernel(
    const CCRegionStat* stats,
    int maxLabel,
    int minArea,
    float aspectRange,
    unsigned char* removeMask)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > maxLabel) return;

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
    unsigned char* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Npp32u id = label[idx];

    if (id == 0 || removeMask[id])
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
        CUDA_CHECK_LOCAL(cudaMalloc(&dSrcLinear, width * height * sizeof(Npp8u)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dLabel, width * height * sizeof(Npp32u)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dOutput, width * height * sizeof(Npp8u)));

        CUDA_CHECK_LOCAL(cudaMemcpy2D(
            dSrcLinear,
            srcStep,
            src.ptr<Npp8u>(),
            src.step,
            srcStep,
            height,
            cudaMemcpyDeviceToDevice));

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

        int nStartingNumber = width * height;
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

            cudaFree(dSrcLinear);
            cudaFree(dLabel);
            cudaFree(dOutput);

            return out;
        }

        CUDA_CHECK_LOCAL(cudaMalloc(&dStats, (maxLabel + 1) * sizeof(CCRegionStat)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dRemoveMask, (maxLabel + 1) * sizeof(unsigned char)));

        int block1D = 256;
        int grid1D = (maxLabel + 1 + block1D - 1) / block1D;

        initCCStatsKernel << <grid1D, block1D >> > (dStats, maxLabel);
        CUDA_CHECK_LOCAL(cudaGetLastError());

        dim3 block2D(16, 16);
        dim3 grid2D(
            (width + block2D.x - 1) / block2D.x,
            (height + block2D.y - 1) / block2D.y);

        computeCCStatsKernel << <grid2D, block2D >> > (
            dLabel,
            width,
            height,
            dStats);

        CUDA_CHECK_LOCAL(cudaGetLastError());

        computeCCRemoveMaskKernel << <grid1D, block1D >> > (
            dStats,
            maxLabel,
            minArea,
            aspectRange,
            dRemoveMask);

        CUDA_CHECK_LOCAL(cudaGetLastError());

        generateCCFilteredBinaryKernel << <grid2D, block2D >> > (
            dLabel,
            dRemoveMask,
            width,
            height,
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