#include "cuda_cc_filter.h"

#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(                                        \
            std::string("CUDA Error: ") + cudaGetErrorString(err));      \
    }                                                                   \
} while (0)

__global__ void computeRemoveMaskKernel(
    const RegionStat* stats,
    int maxLabel,
    int minArea,
    int maxArea,
    float minAspect,
    float maxAspect,
    unsigned char* removeMask
)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id > maxLabel)
        return;

    if (id == 0)
    {
        removeMask[id] = 1;
        return;
    }

    RegionStat s = stats[id];

    int width = s.xmax - s.xmin + 1;
    int height = s.ymax - s.ymin + 1;

    unsigned char remove = 0;

    if (s.area <= 0 || width <= 0 || height <= 0)
    {
        remove = 1;
    }
    else
    {
        float aspect = static_cast<float>(width) / static_cast<float>(height);

        if (minArea >= 0 && s.area < minArea)
            remove = 1;

        if (maxArea >= 0 && s.area > maxArea)
            remove = 1;

        if (minAspect > 0.0f && aspect < minAspect)
            remove = 1;

        if (maxAspect > 0.0f && aspect > maxAspect)
            remove = 1;
    }

    removeMask[id] = remove;
}

void computeRemoveMaskCUDA(
    const RegionStat* dStats,
    int maxLabel,
    int minArea,
    int maxArea,
    float minAspect,
    float maxAspect,
    unsigned char** dRemoveMaskOut
)
{
    if (!dStats)
        throw std::runtime_error("computeRemoveMaskCUDA: dStats is null.");

    if (maxLabel <= 0)
        throw std::runtime_error("computeRemoveMaskCUDA: maxLabel must be > 0.");

    unsigned char* dRemoveMask = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dRemoveMask,
        (maxLabel + 1) * sizeof(unsigned char)
    ));

    int block = 256;
    int grid = (maxLabel + 1 + block - 1) / block;

    computeRemoveMaskKernel << <grid, block >> > (
        dStats,
        maxLabel,
        minArea,
        maxArea,
        minAspect,
        maxAspect,
        dRemoveMask
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    *dRemoveMaskOut = dRemoveMask;
}

std::vector<unsigned char> downloadRemoveMaskCUDA(
    unsigned char* dRemoveMask,
    int maxLabel
)
{
    if (!dRemoveMask)
        throw std::runtime_error("downloadRemoveMaskCUDA: dRemoveMask is null.");

    std::vector<unsigned char> hMask(maxLabel + 1);

    CUDA_CHECK(cudaMemcpy(
        hMask.data(),
        dRemoveMask,
        (maxLabel + 1) * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));

    return hMask;
}

void freeRemoveMaskCUDA(unsigned char* dRemoveMask)
{
    if (dRemoveMask)
        cudaFree(dRemoveMask);
}