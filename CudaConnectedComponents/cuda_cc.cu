#include <fstream>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <mutex>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include "cuda_cc.h"
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

namespace cuda_cc
{

    cv::cuda::GpuMat executeLabelCUDA(
        const cv::cuda::GpuMat& srcMask)
    {
        CV_Assert(!srcMask.empty());
        CV_Assert(srcMask.type() == CV_8UC1);

        //------------------------------------------
        // Image Size
        //------------------------------------------

        NppiSize roi;
        roi.width = srcMask.cols;
        roi.height = srcMask.rows;

        //------------------------------------------
        // Buffer Size
        //------------------------------------------

        int bufferSize = 0;

        NppStatus status =
            nppiLabelMarkersUFGetBufferSize_32u_C1R(
                roi,
                &bufferSize);

        CV_Assert(status == NPP_SUCCESS);

        //------------------------------------------
        // Allocate Buffer
        //------------------------------------------

        Npp8u* pBuffer = nullptr;

        cudaError_t cudaStatus =
            cudaMalloc((void**)&pBuffer, bufferSize);

        CV_Assert(cudaStatus == cudaSuccess);

        //------------------------------------------
        // Allocate Label Image
        //------------------------------------------

        cv::cuda::GpuMat label32(
            srcMask.size(),
            CV_32SC1);

        //------------------------------------------
        // Execute Labeling
        //------------------------------------------

        status =
            nppiLabelMarkersUF_8u32u_C1R(
                (Npp8u*)srcMask.ptr<Npp8u>(),
                (int)srcMask.step,

                (Npp32u*)label32.ptr<Npp32u>(),
                (int)label32.step,

                roi,

                nppiNormInf,

                pBuffer);

        //------------------------------------------
        // Release Buffer
        //------------------------------------------

        cudaFree(pBuffer);

        CV_Assert(status == NPP_SUCCESS);

        return label32;
    }

    int cuda_cc::executeCompressLabelCUDA(
        cv::cuda::GpuMat& labelImage)
    {
        CV_Assert(!labelImage.empty());
        CV_Assert(labelImage.type() == CV_32SC1);

        //------------------------------------------
        // ROI
        //------------------------------------------

        NppiSize roi;

        roi.width = labelImage.cols;
        roi.height = labelImage.rows;

        //------------------------------------------
        // Max Label
        //------------------------------------------

        Mat cpuLabel;

        labelImage.download(cpuLabel);

        double maxLabel = 0;

        minMaxLoc(cpuLabel,
            nullptr,
            &maxLabel);

        if (maxLabel == 0)
            return 0;

        //------------------------------------------
        // Compress
        //------------------------------------------

        NppStatus status =
            nppiCompressMarkerLabelsUF_32u_C1IR(
                (Npp32u*)labelImage.ptr<Npp32u>(),
                (int)labelImage.step,
                roi,
                (Npp32u)maxLabel);

        CV_Assert(status == NPP_SUCCESS);

        //------------------------------------------
        // Count
        //------------------------------------------

        labelImage.download(cpuLabel);

        minMaxLoc(cpuLabel,
            nullptr,
            &maxLabel);

        return (int)maxLabel;
    }

}