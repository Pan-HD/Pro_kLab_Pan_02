#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

#include <npp.h>
#include <nppi.h>

//namespace cuda_cc
//{
//    // Binary(8UC1)
//    //      ¡ý
//    // Label(32SC1)
//    cv::cuda::GpuMat executeLabelCUDA(
//        const cv::cuda::GpuMat& srcMask);
//}

namespace cuda_cc
{

    cv::cuda::GpuMat executeLabelCUDA(
        const cv::cuda::GpuMat& srcMask);

    int executeCompressLabelCUDA(
        cv::cuda::GpuMat& labelImage);

}