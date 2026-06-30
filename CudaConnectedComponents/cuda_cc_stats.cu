#include "cuda_cc_stats.h"

#include <stdexcept>
#include <string>
#include <climits>

#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(                                        \
            std::string("CUDA Error: ") + cudaGetErrorString(err));      \
    }                                                                   \
} while (0)

__global__ void initRegionStatsKernel(
    RegionStat* stats,
    int maxLabel
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > maxLabel)
        return;

    stats[idx].area = 0;
    stats[idx].xmin = INT_MAX;
    stats[idx].xmax = -1;
    stats[idx].ymin = INT_MAX;
    stats[idx].ymax = -1;
}

__global__ void computeRegionStatsKernel(
    const unsigned int* label,
    int width,
    int height,
    RegionStat* stats
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    unsigned int id = label[y * width + x];

    if (id == 0)
        return;

    atomicAdd(&stats[id].area, 1);

    atomicMin(&stats[id].xmin, x);
    atomicMax(&stats[id].xmax, x);

    atomicMin(&stats[id].ymin, y);
    atomicMax(&stats[id].ymax, y);
}

void computeRegionStatsCUDA(
    const unsigned int* dLabel,
    int width,
    int height,
    int maxLabel,
    RegionStat** dStatsOut
)
{
    if (dLabel == nullptr)
        throw std::runtime_error("computeRegionStatsCUDA: dLabel is null.");

    if (width <= 0 || height <= 0)
        throw std::runtime_error("computeRegionStatsCUDA: invalid image size.");

    if (maxLabel <= 0)
        throw std::runtime_error("computeRegionStatsCUDA: maxLabel must be > 0.");

    RegionStat* dStats = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dStats,
        (maxLabel + 1) * sizeof(RegionStat)
    ));

    int block1D = 256;
    int grid1D = (maxLabel + 1 + block1D - 1) / block1D;

    initRegionStatsKernel << <grid1D, block1D >> > (
        dStats,
        maxLabel
        );

    CUDA_CHECK(cudaGetLastError());

    dim3 block2D(16, 16);
    dim3 grid2D(
        (width + block2D.x - 1) / block2D.x,
        (height + block2D.y - 1) / block2D.y
    );

    computeRegionStatsKernel << <grid2D, block2D >> > (
        dLabel,
        width,
        height,
        dStats
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    *dStatsOut = dStats;
}

std::vector<RegionStat> downloadRegionStatsCUDA(
    RegionStat* dStats,
    int maxLabel
)
{
    if (dStats == nullptr)
        throw std::runtime_error("downloadRegionStatsCUDA: dStats is null.");

    std::vector<RegionStat> hStats(maxLabel + 1);

    CUDA_CHECK(cudaMemcpy(
        hStats.data(),
        dStats,
        (maxLabel + 1) * sizeof(RegionStat),
        cudaMemcpyDeviceToHost
    ));

    return hStats;
}

void freeRegionStatsCUDA(RegionStat* dStats)
{
    if (dStats)
        cudaFree(dStats);
}