#include "CudaCCL.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include <npp.h>
#include <nppi.h>

using namespace cv;

//////////////////////////////////////////////////////////////////////////
// Phase 2
//////////////////////////////////////////////////////////////////////////

int CudaCCL::compressLabelsCUDA(GpuMat& labels)
{
    CV_Assert(labels.type() == CV_32SC1);

    NppiSize roi;
    roi.width = labels.cols;
    roi.height = labels.rows;

    //------------------------------------------------------------------
    // Step1
    // Query Buffer Size
    //------------------------------------------------------------------

    int bufferSize = 0;

    NppStatus status =
        nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
            roi,
            &bufferSize);

    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error(
            "nppiCompressMarkerLabelsGetBufferSize failed.");
    }

    //------------------------------------------------------------------
    // Step2
    // Allocate Temp Buffer
    //------------------------------------------------------------------

    Npp8u* pBuffer = nullptr;

    cudaError_t cudaStatus =
        cudaMalloc((void**)&pBuffer, bufferSize);

    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaMalloc buffer failed.");
    }

    //------------------------------------------------------------------
    // Step3
    // Compress Labels
    //------------------------------------------------------------------

    int newMaxLabel = 0;

    status =
        nppiCompressMarkerLabelsUF_32u_C1IR(
            (Npp32u*)labels.ptr<Npp32s>(),
            labels.step,
            roi,
            &newMaxLabel,
            pBuffer);

    cudaFree(pBuffer);

    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error(
            "nppiCompressMarkerLabelsUF failed.");
    }

    cudaDeviceSynchronize();

    return newMaxLabel;
}