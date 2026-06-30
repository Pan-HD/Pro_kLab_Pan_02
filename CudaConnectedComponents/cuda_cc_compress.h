#pragma once

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

struct CompressLabelResult
{
    cv::cuda::GpuMat label32;   // compressed label image, CV_32SC1
    // unsigned int maxLabel;      // number of connected components
    int maxLabel = 0;
};

CompressLabelResult compressLabelsUF_CUDA(
    const cv::cuda::GpuMat& label32,
    int width,
    int height
);