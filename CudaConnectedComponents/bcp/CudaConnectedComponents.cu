#include "CudaConnectedComponents.h"

#include <opencv2/core/cuda.hpp>

#include <cuda_runtime.h>

#include <npp.h>
#include <nppi.h>

#include <iostream>

bool cudaConnectedComponents(
    const cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& labels,
    int& numLabels)
{
    if (src.empty())
        return false;

    if (src.type() != CV_8UC1)
        return false;

    //-----------------------------------------
    // 输出 Label Image
    //-----------------------------------------

    labels.create(src.size(), CV_32SC1);

    //-----------------------------------------
    // ROI
    //-----------------------------------------

    NppiSize roi;
    roi.width = src.cols;
    roi.height = src.rows;

    //-----------------------------------------
    // Query Buffer
    //-----------------------------------------

    int bufferSize = 0;

    NppStatus status;

    status = nppiLabelMarkersUFGetBufferSize_8u32u_C1R(
        roi,
        &bufferSize);

    if (status != NPP_SUCCESS)
    {
        std::cout << "Buffer Query Failed : "
            << status << std::endl;
        return false;
    }

    std::cout << "Buffer Size = "
        << bufferSize
        << " bytes"
        << std::endl;

    //-----------------------------------------
    // Allocate Buffer
    //-----------------------------------------

    Npp8u* pBuffer = nullptr;

    cudaError_t cudaStatus =
        cudaMalloc((void**)&pBuffer, bufferSize);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaMalloc failed."
            << std::endl;
        return false;
    }

    //-----------------------------------------
    // Label
    //-----------------------------------------

    status = nppiLabelMarkersUF_8u32u_C1R(
        src.ptr<Npp8u>(),
        (int)src.step,

        (Npp32u*)labels.ptr<int>(),
        (int)labels.step,

        roi,

        nppiNormInf,

        pBuffer);

    cudaDeviceSynchronize();

    if (status != NPP_SUCCESS)
    {
        std::cout << "Label Failed : "
            << status << std::endl;

        cudaFree(pBuffer);
        return false;
    }

    //-----------------------------------------
    // Compress Label
    //-----------------------------------------
    // LabelMarkersUF 输出Label可能不连续
    // Compress以后：
    // 1,2,3,4,...
    //-----------------------------------------

    int compressBuffer = 0;

    status =
        nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
            roi.width * roi.height,
            &compressBuffer);

    if (status == NPP_SUCCESS)
    {
        Npp8u* pCompressBuffer = nullptr;

        cudaMalloc(
            (void**)&pCompressBuffer,
            compressBuffer);

        status =
            nppiCompressMarkerLabelsUF_32u_C1IR(
                (Npp32u*)labels.ptr<int>(),
                (int)labels.step,
                roi,
                roi.width * roi.height,
                &numLabels,
                pCompressBuffer);

        cudaFree(pCompressBuffer);
    }
    else
    {
        numLabels = 0;
    }

    cudaFree(pBuffer);

    return true;
}