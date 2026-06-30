#pragma once
#include <opencv2/core/cuda.hpp>
bool cudaConnectedComponents(
    const cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& labels,
    int& numLabels);