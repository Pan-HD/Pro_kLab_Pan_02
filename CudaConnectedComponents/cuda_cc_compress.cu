#include "cuda_cc_compress.h"

#include <stdexcept>
#include <string>
#include <iostream>

#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(                                        \
            std::string("CUDA Error: ") + cudaGetErrorString(err));      \
    }                                                                   \
} while (0)

#define NPP_CHECK(call)                                                  \
do {                                                                    \
    NppStatus status = (call);                                           \
    if (status != NPP_SUCCESS) {                                         \
        throw std::runtime_error(                                        \
            std::string("NPP Error, status = ") +                       \
            std::to_string((int)status));                                \
    }                                                                   \
} while (0)

CompressLabelResult compressLabelsUF_CUDA(
    const cv::cuda::GpuMat& label32,
    int width,
    int height
)
{
    if (label32.empty())
        throw std::runtime_error("compressLabelsUF_CUDA: label32 is empty.");

    if (label32.type() != CV_32SC1)
        throw std::runtime_error("compressLabelsUF_CUDA: label32 must be CV_32SC1.");

    /*
        NPP CompressMarkerLabelsUF has a strict requirement:

        step MUST be:
            width * sizeof(Npp32u)

        Therefore Phase 1 output must be linear memory, not pitched memory.
    */
    const int requiredStep = width * sizeof(Npp32u);

    if ((int)label32.step != requiredStep)
    {
        throw std::runtime_error(
            "compressLabelsUF_CUDA: label32.step must equal width * sizeof(Npp32u). "
            "Please allocate Phase 1 label image using linear cudaMalloc memory."
        );
    }

    NppiSize roi;
    roi.width = width;
    roi.height = height;

    CompressLabelResult result;
    result.label32 = label32.clone();

    Npp32u* pLabel = reinterpret_cast<Npp32u*>(result.label32.ptr<Npp32s>());

    /*
        For output from nppiLabelMarkersUF, nStartingNumber should be:

            width * height
    */
    int nStartingNumber = width * height;

    int bufferSize = 0;

    NPP_CHECK(
        nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
            nStartingNumber,
            &bufferSize
        )
    );

    Npp8u* pBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&pBuffer, bufferSize));

    int maxLabel = 0;

    NPP_CHECK(
        nppiCompressMarkerLabelsUF_32u_C1IR(
            pLabel,
            requiredStep,
            roi,
            nStartingNumber,
            &maxLabel,
            pBuffer
        )
    );

    CUDA_CHECK(cudaFree(pBuffer));

    result.maxLabel = static_cast<unsigned int>(maxLabel);

    return result;
}