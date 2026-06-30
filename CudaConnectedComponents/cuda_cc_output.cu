#include "cuda_cc_output.h"

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(                                        \
            std::string("CUDA Error: ") + cudaGetErrorString(err));      \
    }                                                                   \
} while (0)

__global__ void generateFilteredBinaryKernel(
    const unsigned int* label,
    const unsigned char* removeMask,
    int width,
    int height,
    unsigned char* output
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    unsigned int id = label[idx];

    if (id == 0)
    {
        output[idx] = 0;
        return;
    }

    output[idx] = removeMask[id] ? 0 : 255;
}

void generateFilteredBinaryCUDA(
    const unsigned int* dLabel,
    const unsigned char* dRemoveMask,
    int width,
    int height,
    unsigned char** dOutputOut
)
{
    if (!dLabel)
        throw std::runtime_error("generateFilteredBinaryCUDA: dLabel is null.");

    if (!dRemoveMask)
        throw std::runtime_error("generateFilteredBinaryCUDA: dRemoveMask is null.");

    if (width <= 0 || height <= 0)
        throw std::runtime_error("generateFilteredBinaryCUDA: invalid image size.");

    unsigned char* dOutput = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dOutput,
        width * height * sizeof(unsigned char)
    ));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    generateFilteredBinaryKernel << <grid, block >> > (
        dLabel,
        dRemoveMask,
        width,
        height,
        dOutput
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    *dOutputOut = dOutput;
}

void freeFilteredBinaryCUDA(unsigned char* dOutput)
{
    if (dOutput)
        cudaFree(dOutput);
}