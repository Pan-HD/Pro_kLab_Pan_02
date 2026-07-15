#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstdio>

#include "../../../CommonComponents/SegmentationConfig.h"

// =============================================================
// Stable deterministic CUDA connected-component filter
//
// This version intentionally removes NPP LabelMarkersUF from the
// CC_FILTER implementation.  It uses a deterministic GPU union-by-min
// connected-component labeling fallback:
//   - foreground pixel initial label = pixelIndex + 1
//   - background label = 0
//   - repeatedly merge 8-neighbour components by atomicMin to the
//     smaller root label
//   - path-compress labels until convergence
//   - compute area/bbox stats and apply the existing CC filtering rule
//
// The final component label of each connected component is the minimum
// pixel index + 1 in that component, so identical input masks should
// produce identical labels and identical filtered output.
// =============================================================
// Stable build: debug hash/copy/printf path is disabled for training.

#define CC_FILTER_STAGE_DEBUG 0
#define DET_CCL_PRINT_ITERATIONS 0

#define CUDA_CHECK_LOCAL(call)                                             \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA Error: ") +         \
                cudaGetErrorString(err__));                                \
        }                                                                  \
    } while (0)

struct CCRegionStat
{
    int area;
    int xmin;
    int xmax;
    int ymin;
    int ymax;
};

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
        hash ^= static_cast<unsigned long long>(p[i]);
        hash *= 1099511628211ull;
    }

    return hash;
}

static unsigned long long gCCDebugCallId = 0;

#define CCDBG_HASH_U8(name, ptr, count)                                    \
    do {                                                                   \
        unsigned long long h__ =                                           \
            fnv1aHashDeviceBuffer<unsigned char>((ptr), (count));           \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, (name), h__);        \
    } while (0)

#define CCDBG_HASH_U32(name, ptr, count)                                   \
    do {                                                                   \
        unsigned long long h__ =                                           \
            fnv1aHashDeviceBuffer<unsigned int>((ptr), (count));            \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, (name), h__);        \
    } while (0)

#define CCDBG_HASH_STATS(name, ptr, count)                                 \
    do {                                                                   \
        unsigned long long h__ =                                           \
            fnv1aHashDeviceBuffer<CCRegionStat>((ptr), (count));            \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, (name), h__);        \
    } while (0)

#define CCDBG_HASH_UCHAR(name, ptr, count)                                 \
    do {                                                                   \
        unsigned long long h__ =                                           \
            fnv1aHashDeviceBuffer<unsigned char>((ptr), (count));           \
        printf("[CCDBG] call=%llu %s=%llu\n", callId, (name), h__);        \
    } while (0)
#endif

// Convert the GP child output into a binary foreground mask used by CCL.
// dFg is stored linearly with step = width.
__global__ void makeForegroundMaskLinearKernel(
    const unsigned char* src,
    size_t srcStep,
    int width,
    int height,
    unsigned char* dFg)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned char v = src[y * srcStep + x];

#if FG_PIXEL == 255
    // White foreground: any non-zero child output is treated as foreground.
    dFg[idx] = (v != 0) ? 255 : 0;
#else
    // Black foreground: zero child output is the logical foreground.
    dFg[idx] = (v == 0) ? 255 : 0;
#endif
}

__global__ void initDeterministicLabelsKernel(
    const unsigned char* dFg,
    unsigned int* dLabel,
    int pixelCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelCount) return;

    dLabel[idx] = (dFg[idx] != 0) ? static_cast<unsigned int>(idx + 1) : 0u;
}

__device__ __forceinline__ unsigned int findRootDeterministic(
    const unsigned int* labels,
    unsigned int label)
{
    if (label == 0u) return 0u;

    // Labels are 1-based pixel indices. Parent pointers monotonically
    // decrease because unions use atomicMin, so cycles should not occur
    // for valid labels. This loop intentionally has no small fixed guard:
    // a long 1-pixel-wide component can temporarily form a long parent
    // chain before compression.
    unsigned int cur = label;

    while (true) {
        unsigned int parent = labels[cur - 1u];

        if (parent == 0u || parent == cur) {
            return cur;
        }

        cur = parent;
    }
}

__global__ void mergeLabelsByMinKernel(
    const unsigned char* dFg,
    unsigned int* labels,
    int width,
    int height,
    unsigned int* changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (dFg[idx] == 0) return;

    unsigned int root = findRootDeterministic(labels, labels[idx]);
    if (root == 0u) return;

    // 8-neighbour connectivity, equivalent to nppiNormInf used previously.
    for (int dy = -1; dy <= 1; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) continue;

        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            if (dx == 0 && dy == 0) continue;
            if (nx < 0 || nx >= width) continue;

            int nidx = ny * width + nx;
            if (dFg[nidx] == 0) continue;

            unsigned int nroot = findRootDeterministic(labels, labels[nidx]);
            if (nroot == 0u || nroot == root) continue;

            unsigned int lo = root < nroot ? root : nroot;
            unsigned int hi = root < nroot ? nroot : root;

            unsigned int old = atomicMin(&labels[hi - 1u], lo);
            if (old > lo) {
                atomicExch(changed, 1u);
            }

            root = lo;
        }
    }
}

__global__ void compressLabelsKernel(
    const unsigned char* dFg,
    unsigned int* labels,
    int pixelCount,
    unsigned int* changed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelCount) return;

    if (dFg[idx] == 0) {
        if (labels[idx] != 0u) {
            labels[idx] = 0u;
            atomicExch(changed, 1u);
        }
        return;
    }

    unsigned int oldLabel = labels[idx];
    unsigned int root = findRootDeterministic(labels, oldLabel);

    if (root != oldLabel) {
        labels[idx] = root;
        atomicExch(changed, 1u);
    }
}

__global__ void countLabelSupportMismatchKernel(
    const unsigned char* fg,
    const unsigned int* label,
    int n,
    unsigned int* fgZero,
    unsigned int* bgNonzero)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bool isFg = (fg[i] != 0);
    bool hasLabel = (label[i] != 0);

    if (isFg && !hasLabel) {
        atomicAdd(fgZero, 1u);
    }

    if (!isFg && hasLabel) {
        atomicAdd(bgNonzero, 1u);
    }
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
    const unsigned int* label,
    int width,
    int height,
    int labelLimit,
    CCRegionStat* stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned int id = label[idx];

    if (id == 0u || id > static_cast<unsigned int>(labelLimit)) return;

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

    if (id == 0) {
        removeMask[id] = 1;
        return;
    }

    CCRegionStat s = stats[id];

    int w = s.xmax - s.xmin + 1;
    int h = s.ymax - s.ymin + 1;

    unsigned char remove = 0;

    if (s.area <= 0 || w <= 0 || h <= 0) {
        remove = 1;
    }
    else {
        float aspect = static_cast<float>(w) / static_cast<float>(h);

        if (s.area < minArea) {
            remove = 1;
        }

        if (aspectRange > 0.0f) {
            float minAspect = 1.0f / aspectRange;
            float maxAspect = aspectRange;

            if (aspect < minAspect || aspect > maxAspect) {
                remove = 1;
            }
        }
    }

    removeMask[id] = remove;
}

__global__ void generateCCFilteredBinaryKernel(
    const unsigned int* label,
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
    unsigned int id = label[idx];

    if (id == 0u || id > static_cast<unsigned int>(labelLimit) || removeMask[id]) {
        output[idx] = BG_PIXEL;
    }
    else {
        output[idx] = FG_PIXEL;
    }
}

#if CC_FILTER_STAGE_DEBUG
static void debugPrintSupportMismatch(
    unsigned long long callId,
    const char* name,
    const unsigned char* dFg,
    const unsigned int* dLabel,
    int pixelCount)
{
    unsigned int* dFgZero = nullptr;
    unsigned int* dBgNonzero = nullptr;

    CUDA_CHECK_LOCAL(cudaMalloc(&dFgZero, sizeof(unsigned int)));
    CUDA_CHECK_LOCAL(cudaMalloc(&dBgNonzero, sizeof(unsigned int)));

    CUDA_CHECK_LOCAL(cudaMemset(dFgZero, 0, sizeof(unsigned int)));
    CUDA_CHECK_LOCAL(cudaMemset(dBgNonzero, 0, sizeof(unsigned int)));

    int block = 256;
    int grid = (pixelCount + block - 1) / block;

    countLabelSupportMismatchKernel << <grid, block >> > (
        dFg,
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
        "[CCDBG] call=%llu %s fgZero=%u bgNonzero=%u\n",
        callId,
        name,
        hFgZero,
        hBgNonzero);

    CUDA_CHECK_LOCAL(cudaFree(dFgZero));
    CUDA_CHECK_LOCAL(cudaFree(dBgNonzero));
}
#endif

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
    printf(
        "[CCDBG] call=%llu width=%d height=%d minArea=%d aspectRange=%.6f\n",
        callId,
        width,
        height,
        minArea,
        aspectRange);
#endif

    unsigned char* dFgLinear = nullptr;
    unsigned int* dLabel = nullptr;
    unsigned char* dOutput = nullptr;
    unsigned int* dChanged = nullptr;
    CCRegionStat* dStats = nullptr;
    unsigned char* dRemoveMask = nullptr;

    try
    {
        CUDA_CHECK_LOCAL(cudaMalloc(&dFgLinear, pixelCount * sizeof(unsigned char)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dLabel, pixelCount * sizeof(unsigned int)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dOutput, pixelCount * sizeof(unsigned char)));
        CUDA_CHECK_LOCAL(cudaMalloc(&dChanged, sizeof(unsigned int)));

        dim3 block2D(16, 16);
        dim3 grid2D(
            (width + block2D.x - 1) / block2D.x,
            (height + block2D.y - 1) / block2D.y);

        const int block1D = 256;
        const int gridPixels = (pixelCount + block1D - 1) / block1D;

        makeForegroundMaskLinearKernel << <grid2D, block2D >> > (
            src.ptr<unsigned char>(),
            src.step,
            width,
            height,
            dFgLinear);

        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

#if CC_FILTER_STAGE_DEBUG
        CCDBG_HASH_U8("dSrcLinear", dFgLinear, pixelCount);
#endif

        initDeterministicLabelsKernel << <gridPixels, block1D >> > (
            dFgLinear,
            dLabel,
            pixelCount);

        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

#if CC_FILTER_STAGE_DEBUG
        CCDBG_HASH_U32("dLabel_init", dLabel, pixelCount);
        debugPrintSupportMismatch(
            callId,
            "labelInitSupportMismatch",
            dFgLinear,
            dLabel,
            pixelCount);
#endif

        int iter = 0;
        const int maxIter = std::max<int>(1024, width + height);

        for (; iter < maxIter; ++iter) {
            CUDA_CHECK_LOCAL(cudaMemset(dChanged, 0, sizeof(unsigned int)));

            mergeLabelsByMinKernel << <grid2D, block2D >> > (
                dFgLinear,
                dLabel,
                width,
                height,
                dChanged);

            CUDA_CHECK_LOCAL(cudaGetLastError());
            CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

            compressLabelsKernel << <gridPixels, block1D >> > (
                dFgLinear,
                dLabel,
                pixelCount,
                dChanged);

            CUDA_CHECK_LOCAL(cudaGetLastError());
            CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

            unsigned int hChanged = 0;
            CUDA_CHECK_LOCAL(cudaMemcpy(
                &hChanged,
                dChanged,
                sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

#if CC_FILTER_STAGE_DEBUG && DET_CCL_PRINT_ITERATIONS
            printf(
                "[CCDBG] call=%llu iter=%d changed=%u\n",
                callId,
                iter,
                hChanged);
#endif

            if (hChanged == 0u) {
                break;
            }
        }

        if (iter >= maxIter) {
            throw std::runtime_error(
                "Deterministic CCL did not converge before maxIter");
        }

#if CC_FILTER_STAGE_DEBUG
        printf(
            "[CCDBG] call=%llu detCCL_iters=%d\n",
            callId,
            iter + 1);
        CCDBG_HASH_U32("dLabel_final", dLabel, pixelCount);
        debugPrintSupportMismatch(
            callId,
            "labelFinalSupportMismatch",
            dFgLinear,
            dLabel,
            pixelCount);
#endif

        // Labels are sparse and bounded by pixelCount, because component roots
        // are deterministic 1-based pixel indices.  Allocate stats for the full
        // label range rather than compressing labels through a non-deterministic
        // library call.
        int labelLimit = pixelCount;

        CUDA_CHECK_LOCAL(cudaMalloc(
            &dStats,
            (labelLimit + 1) * sizeof(CCRegionStat)));

        CUDA_CHECK_LOCAL(cudaMalloc(
            &dRemoveMask,
            (labelLimit + 1) * sizeof(unsigned char)));

        int gridLabels = (labelLimit + 1 + block1D - 1) / block1D;

        initCCStatsKernel << <gridLabels, block1D >> > (dStats, labelLimit);
        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        computeCCStatsKernel << <grid2D, block2D >> > (
            dLabel,
            width,
            height,
            labelLimit,
            dStats);

        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

#if CC_FILTER_STAGE_DEBUG
        CCDBG_HASH_STATS("dStats", dStats, labelLimit + 1);
#endif

        computeCCRemoveMaskKernel << <gridLabels, block1D >> > (
            dStats,
            labelLimit,
            minArea,
            aspectRange,
            dRemoveMask);

        CUDA_CHECK_LOCAL(cudaGetLastError());
        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

#if CC_FILTER_STAGE_DEBUG
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

#if CC_FILTER_STAGE_DEBUG
        CCDBG_HASH_U8("dOutput", dOutput, pixelCount);
#endif

        cv::cuda::GpuMat out(height, width, CV_8UC1);

        CUDA_CHECK_LOCAL(cudaMemcpy2D(
            out.ptr<unsigned char>(),
            out.step,
            dOutput,
            width * sizeof(unsigned char),
            width * sizeof(unsigned char),
            height,
            cudaMemcpyDeviceToDevice));

        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        cudaFree(dFgLinear);
        cudaFree(dLabel);
        cudaFree(dOutput);
        cudaFree(dChanged);
        cudaFree(dStats);
        cudaFree(dRemoveMask);

        return out;
    }
    catch (...)
    {
        if (dFgLinear) cudaFree(dFgLinear);
        if (dLabel) cudaFree(dLabel);
        if (dOutput) cudaFree(dOutput);
        if (dChanged) cudaFree(dChanged);
        if (dStats) cudaFree(dStats);
        if (dRemoveMask) cudaFree(dRemoveMask);
        throw;
    }
}
