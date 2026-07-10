#pragma once
#include <opencv2/core/cuda.hpp>

struct MetricsGPUFused
{
    long long tp = 0;
    long long fp = 0;
    long long fn = 0;
    long long invalid = 0;
};

MetricsGPUFused calcMetricsOneGPUFused(
    const cv::cuda::GpuMat& pred,
    const cv::cuda::GpuMat& gt,
    int fgPixel,
    cv::cuda::Stream& stream);