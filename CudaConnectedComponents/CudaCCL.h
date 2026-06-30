#pragma once

#include <opencv2/core/cuda.hpp>

class CudaCCL
{
public:

    // Phase 1
    static int labelMarkersCUDA(
        const cv::cuda::GpuMat& binary,
        cv::cuda::GpuMat& labels);

    // Phase 2
    static int compressLabelsCUDA(
        cv::cuda::GpuMat& labels);
};