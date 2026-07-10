#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstdio>
#include "../../CommonComponents/SegmentationConfig.h"

// Adding part-03
// Direction A: use ordinary NPP LabelMarkers instead of LabelMarkersUF.

// NOTE:
// This Direction-A file uses legacy ordinary NPP LabelMarkers APIs:
//   nppiLabelMarkersGetBufferSize_8u32u_C1R
//   nppiLabelMarkers_8u32u_C1R
//   nppiCompressMarkerLabels_32u_C1IR
// These APIs are not available in some newer CUDA/NPP SDKs. If your NPP
// headers do not declare them, this file cannot be compiled with that SDK;
// use an older CUDA/NPP toolkit or move to deterministic custom GPU CCL.
// Keep stage hash / support mismatch debug for determinism testing.
#define CC_FILTER_STAGE_DEBUG 1

#if CC_FILTER_STAGE_DEBUG
template <typename T>
static unsigned long long fnv1aHashDeviceBuffer(
    const T* dptr,
    size_t count)
{
    std::vector<T> h(count);
    cudaError_t err = cudaMemcpy(
        h.data(),
        dptr,
        count * sizeof(T),
        cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") +
            cudaGetErrorString(err));
    }

    const unsigned char* p =
        reinterpret_cast<const unsigned char*>(h.data());

    size_t bytes = count * sizeof(T);

    unsigned long long hash = 1469598103934665603ull;

    for (size_t i = 0; i < bytes; ++i) {
        hash ^= (unsigned long long)p[i];
        hash *= 1099511628211ull;
    }

    return hash;
}

static unsigned long long gCCDebugCallId = 0;

#define CCDBG_HASH_U8(name, ptr, count)                                      \
    do {                                                                     \
        unsigned long long h = fnv1aHashDeviceBuffer<Npp8u>(ptr, count);      \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, name, h);               \
    } while (0)

#define CCDBG_HASH_U32(name, ptr, count)                                     \
    do {                                                                     \
        unsigned long long h = fnv1aHashDeviceBuffer<Npp32u>(ptr, count);     \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, name, h);               \
    } while (0)

#define CCDBG_HASH_STATS(name, ptr, count)                                   \
    do {                                                                     \
        unsigned long long h = fnv1aHashDeviceBuffer<CCRegionStat>(ptr, count); \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, name, h);               \
    } while (0)

#define CCDBG_HASH_UCHAR(name, ptr, count)                                   \
    do {                                                                     \
        unsigned long long h = fnv1aHashDeviceBuffer<unsigned char>(ptr, count); \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, name, h);               \
    } while (0)
#endif

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
    // Ordinary nppiLabelMarkers_8u32u_C1R will use nMinVal=0 below,
    // so non-zero pixels are foreground and zero pixels are background.
    dst[y * dstStep + x] = (v != 0) ? 255 : 0;
#else
    // Ordinary nppiLabelMarkers_8u32u_C1R will use nMinVal=0 below,
    // so invert black foreground into non-zero foreground before NPP CCL.
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
    // happen after nppiCompressMarkerLabels, but without the guard a single
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

__global__ void countLabelSupportMismatchKernel(
    const Npp8u* fg,
    const Npp32u* label,
    int n,
    unsigned int* fgZero,
    unsigned int* bgNonzero)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bool isFg = (fg[i] != 0);
    bool hasLabel = (label[i] != 0);

    if (isFg && !hasLabel) {
        atomicAdd(fgZero, 1);
    }

    if (!isFg && hasLabel) {
        atomicAdd(bgNonzero, 1);
    }
}

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

#if CC_FILTER_STAGE_DEBUG
    unsigned long long callId = ++gCCDebugCallId;
    printf("[CCDBG] call=%llu width=%d height=%d minArea=%d aspectRange=%.6f\n",
        callId, width, height, minArea, aspectRange);
#endif

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
    unsigned int* dFgZero = nullptr;
    unsigned int* dBgNonzero = nullptr;
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

        makeNppForegroundMaskKernel << <grid2D, block2D >> > (
            src.ptr<Npp8u>(),
            src.step,
            width,
            height,
            dSrcLinear,
            srcStep);
        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());
        // Adding part-05
#if CC_FILTER_STAGE_DEBUG
        CCDBG_HASH_U8("dSrcLinear", dSrcLinear, pixelCount);
#endif

        CUDA_CHECK_LOCAL(cudaMemset(dLabel, 0, pixelCount * sizeof(Npp32u)));

        int labelBufferSize = 0;
        NPP_CHECK_LOCAL(nppiLabelMarkersGetBufferSize_8u32u_C1R(
            roi,
            &labelBufferSize));

        CUDA_CHECK_LOCAL(cudaMalloc(&dLabelBuffer, labelBufferSize));
        // Explicitly clear NPP scratch buffer for determinism testing.
        CUDA_CHECK_LOCAL(cudaMemset(dLabelBuffer, 0, labelBufferSize));
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        int maxLabelRaw = 0;

        // Direction A: ordinary LabelMarkers, not LabelMarkersUF.
        // nMinVal=0 means pixels <= 0 are background and non-zero pixels are
        // connected-component foreground. nppiNormInf means 8-connectivity.
        NPP_CHECK_LOCAL(nppiLabelMarkers_8u32u_C1R(
            dSrcLinear,
            srcStep,
            dLabel,
            labelStep,
            roi,
            (Npp8u)0,
            nppiNormInf,
            &maxLabelRaw,
            dLabelBuffer));

        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

#if CC_FILTER_STAGE_DEBUG
        printf("[CCDBG] call=%llu maxLabelRaw=%d\n", callId, maxLabelRaw);
        CCDBG_HASH_U32("dLabel_raw", dLabel, pixelCount);
#endif

        // Adding part-12
#if CC_FILTER_STAGE_DEBUG
        CUDA_CHECK_LOCAL(cudaMalloc(&dFgZero, sizeof(unsigned int)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dBgNonzero, sizeof(unsigned int)));

        CUDA_CHECK_LOCAL(cudaMemset(dFgZero, 0, sizeof(unsigned int)));
        CUDA_CHECK_LOCAL(cudaMemset(dBgNonzero, 0, sizeof(unsigned int)));

        int block = 256;
        int grid = (pixelCount + block - 1) / block;

        countLabelSupportMismatchKernel << <grid, block >> > (
            dSrcLinear,
            dLabel,
            pixelCount,
            dFgZero,
            dBgNonzero);

        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        unsigned int hFgZero = 0;
        unsigned int hBgNonzero = 0;

        CUDA_CHECK_LOCAL(cudaMemcpy(
            &hFgZero,
            dFgZero,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK_LOCAL(cudaMemcpy(
            &hBgNonzero,
            dBgNonzero,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost));

        printf(
            "[CCDBG] call=%llu rawSupportMismatch fgZero=%u bgNonzero=%u\n",
            callId,
            hFgZero,
            hBgNonzero);

        CUDA_CHECK_LOCAL(cudaFree(dFgZero));
        dFgZero = nullptr;
        CUDA_CHECK_LOCAL(cudaFree(dBgNonzero));
        dBgNonzero = nullptr;
#endif

        CUDA_CHECK_LOCAL(cudaFree(dLabelBuffer));
        dLabelBuffer = nullptr;

        if (maxLabelRaw <= 0)
        {
            cv::cuda::GpuMat out(height, width, CV_8UC1);
            out.setTo(cv::Scalar(BG_PIXEL));
            CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

            CUDA_CHECK_LOCAL(cudaFree(dSrcLinear));
            dSrcLinear = nullptr;
            CUDA_CHECK_LOCAL(cudaFree(dLabel));
            dLabel = nullptr;
            CUDA_CHECK_LOCAL(cudaFree(dOutput));
            dOutput = nullptr;

            return out;
        }

        // For ordinary LabelMarkers, nStartingNumber must be the value returned
        // by nppiLabelMarkers_8u32u_C1R, not ROI width * ROI height.
        int nStartingNumber = maxLabelRaw;
        int compressBufferSize = 0;

        NPP_CHECK_LOCAL(nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
            nStartingNumber,
            &compressBufferSize));

        CUDA_CHECK_LOCAL(cudaMalloc(&dCompressBuffer, compressBufferSize));
        // Explicitly clear NPP scratch buffer for determinism testing.
        CUDA_CHECK_LOCAL(cudaMemset(dCompressBuffer, 0, compressBufferSize));
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        int maxLabel = 0;

        // Direction A: ordinary CompressMarkerLabels for ordinary LabelMarkers
        // output. Do not use nppiCompressMarkerLabelsUF here.
        NPP_CHECK_LOCAL(nppiCompressMarkerLabels_32u_C1IR(
            dLabel,
            labelStep,
            roi,
            nStartingNumber,
            &maxLabel,
            dCompressBuffer));

        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        // Adding part-07
#if CC_FILTER_STAGE_DEBUG
        printf("[CCDBG] call=%llu maxLabel=%d\n", callId, maxLabel);
        CCDBG_HASH_U32("dLabel_compressed", dLabel, pixelCount);
#endif

        CUDA_CHECK_LOCAL(cudaFree(dCompressBuffer));
        dCompressBuffer = nullptr;

        if (maxLabel <= 0)
        {
            cv::cuda::GpuMat out(height, width, CV_8UC1);
            out.setTo(cv::Scalar(BG_PIXEL));
            CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

            CUDA_CHECK_LOCAL(cudaFree(dSrcLinear));
            dSrcLinear = nullptr;
            CUDA_CHECK_LOCAL(cudaFree(dLabel));
            dLabel = nullptr;
            CUDA_CHECK_LOCAL(cudaFree(dOutput));
            dOutput = nullptr;

            return out;
        }

        // Allocate up to the original maximum raw label ID. After ordinary
        // compression, valid labels should be in 1..maxLabel, but using
        // nStartingNumber is a defensive upper bound against any sparse label
        // edge case while still being much smaller than pixelCount.
        int labelLimit = nStartingNumber;

        CUDA_CHECK_LOCAL(cudaMalloc(&dStats, (labelLimit + 1) * sizeof(CCRegionStat)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dRemoveMask, (labelLimit + 1) * sizeof(unsigned char)));

        int block1D = 256;
        int grid1D = (labelLimit + 1 + block1D - 1) / block1D;

        initCCStatsKernel << <grid1D, block1D >> > (dStats, labelLimit);
        CUDA_CHECK_LOCAL(cudaGetLastError());
#if CC_FILTER_STAGE_DEBUG
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());
#endif

        computeCCStatsKernel << <grid2D, block2D >> > (
            dLabel,
            width,
            height,
            labelLimit,
            dStats);
        CUDA_CHECK_LOCAL(cudaGetLastError());

        // Adding part-08
#if CC_FILTER_STAGE_DEBUG
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());
        CCDBG_HASH_STATS("dStats", dStats, labelLimit + 1);
#endif

        computeCCRemoveMaskKernel << <grid1D, block1D >> > (
            dStats,
            labelLimit,
            minArea,
            aspectRange,
            dRemoveMask);
        CUDA_CHECK_LOCAL(cudaGetLastError());

#if CC_FILTER_STAGE_DEBUG
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());
        CCDBG_HASH_UCHAR("dRemoveMask", dRemoveMask, labelLimit + 1);
#endif

        generateCCFilteredBinaryKernel << <grid2D, block2D >> > (
            dLabel,
            dRemoveMask,
            width,
            height,
            labelLimit,
            dOutput);
        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        // Adding part-09
#if CC_FILTER_STAGE_DEBUG
        CCDBG_HASH_U8("dOutput", dOutput, pixelCount);
#endif

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
        if (dFgZero) cudaFree(dFgZero);
        if (dBgNonzero) cudaFree(dBgNonzero);
        if (dStats) cudaFree(dStats);
        if (dRemoveMask) cudaFree(dRemoveMask);
        throw;
    }
}
