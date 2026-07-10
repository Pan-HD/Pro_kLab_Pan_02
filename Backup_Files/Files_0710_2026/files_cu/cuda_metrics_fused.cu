#include "cuda_metrics_fused.cuh"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cstdio>

namespace
{

    __global__ void fusedMetricsKernel(
        const unsigned char* pred,
        size_t predStep,
        const unsigned char* gt,
        size_t gtStep,
        int rows,
        int cols,
        int fgPixel,
        unsigned long long* counters)
    {
        __shared__ unsigned int s_tp;
        __shared__ unsigned int s_fp;
        __shared__ unsigned int s_fn;
        __shared__ unsigned int s_invalid;

        if (threadIdx.x == 0) {
            s_tp = 0;
            s_fp = 0;
            s_fn = 0;
            s_invalid = 0;
        }

        __syncthreads();

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = rows * cols;

        if (idx < total) {
            const int y = idx / cols;
            const int x = idx % cols;

            const unsigned char p =
                pred[y * predStep + x];

            const unsigned char g =
                gt[y * gtStep + x];

            const unsigned char FG =
                static_cast<unsigned char>(fgPixel);

            const unsigned char BG =
                (FG == 255) ? static_cast<unsigned char>(0)
                : static_cast<unsigned char>(255);

            const bool predFG = (p == FG);
            const bool predBG = (p == BG);

            const bool gtFG = (g == FG);
            const bool gtBG = (g == BG);

            // Important:
            // Match old CUDA scoring semantics exactly.
            //
            // Old path:
            //   predFG = pred == FG_PIXEL
            //   predBG = pred == BG_PIXEL
            //   gtFG   = gt   == FG_PIXEL
            //   gtBG   = gt   == BG_PIXEL
            //
            //   TP = predFG & gtFG
            //   FP = predFG & gtBG
            //   FN = predBG & gtFG
            //
            // Old path checks only whether prediction is binary.
            // It does not penalize non-binary GT pixels.
            const bool predInvalid = !(predFG || predBG);

            if (predInvalid) {
                atomicAdd(&s_invalid, 1u);
            }
            else {
                if (predFG && gtFG) {
                    atomicAdd(&s_tp, 1u);
                }
                else if (predFG && gtBG) {
                    atomicAdd(&s_fp, 1u);
                }
                else if (predBG && gtFG) {
                    atomicAdd(&s_fn, 1u);
                }
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(&counters[0], static_cast<unsigned long long>(s_tp));
            atomicAdd(&counters[1], static_cast<unsigned long long>(s_fp));
            atomicAdd(&counters[2], static_cast<unsigned long long>(s_fn));
            atomicAdd(&counters[3], static_cast<unsigned long long>(s_invalid));
        }
    }

    inline void checkCudaError(cudaError_t err, const char* msg)
    {
        if (err != cudaSuccess) {
            std::fprintf(
                stderr,
                "[cuda_metrics_fused] %s failed: %s\n",
                msg,
                cudaGetErrorString(err));
        }
    }

} // namespace

MetricsGPUFused calcMetricsOneGPUFused(
    const cv::cuda::GpuMat& pred,
    const cv::cuda::GpuMat& gt,
    int fgPixel,
    cv::cuda::Stream& stream)
{
    MetricsGPUFused out;

    if (pred.empty() || gt.empty()) {
        out.invalid = 1;
        return out;
    }

    if (pred.rows != gt.rows || pred.cols != gt.cols) {
        out.invalid = 1;
        return out;
    }

    if (pred.type() != CV_8UC1 || gt.type() != CV_8UC1) {
        out.invalid = 1;
        return out;
    }

    cudaStream_t cudaStream =
        cv::cuda::StreamAccessor::getStream(stream);

    unsigned long long* dCounters = nullptr;

    cudaError_t err =
        cudaMalloc(
            &dCounters,
            sizeof(unsigned long long) * 4);

    if (err != cudaSuccess) {
        checkCudaError(err, "cudaMalloc(dCounters)");
        out.invalid = 1;
        return out;
    }

    err =
        cudaMemsetAsync(
            dCounters,
            0,
            sizeof(unsigned long long) * 4,
            cudaStream);

    if (err != cudaSuccess) {
        checkCudaError(err, "cudaMemsetAsync(dCounters)");
        cudaFree(dCounters);
        out.invalid = 1;
        return out;
    }

    const int total =
        pred.rows * pred.cols;

    const int blockSize = 256;
    const int gridSize =
        (total + blockSize - 1) / blockSize;

    fusedMetricsKernel << <gridSize, blockSize, 0, cudaStream >> > (
        pred.ptr<unsigned char>(),
        pred.step,
        gt.ptr<unsigned char>(),
        gt.step,
        pred.rows,
        pred.cols,
        fgPixel,
        dCounters);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        checkCudaError(err, "fusedMetricsKernel launch");
        cudaFree(dCounters);
        out.invalid = 1;
        return out;
    }

    unsigned long long hCounters[4] = { 0, 0, 0, 0 };

    err =
        cudaMemcpyAsync(
            hCounters,
            dCounters,
            sizeof(unsigned long long) * 4,
            cudaMemcpyDeviceToHost,
            cudaStream);

    if (err != cudaSuccess) {
        checkCudaError(err, "cudaMemcpyAsync(hCounters)");
        cudaFree(dCounters);
        out.invalid = 1;
        return out;
    }

    err =
        cudaStreamSynchronize(cudaStream);

    if (err != cudaSuccess) {
        checkCudaError(err, "cudaStreamSynchronize");
        cudaFree(dCounters);
        out.invalid = 1;
        return out;
    }

    cudaFree(dCounters);

    out.tp =
        static_cast<long long>(hCounters[0]);

    out.fp =
        static_cast<long long>(hCounters[1]);

    out.fn =
        static_cast<long long>(hCounters[2]);

    out.invalid =
        static_cast<long long>(hCounters[3]);

    return out;
}