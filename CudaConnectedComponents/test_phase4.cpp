#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

#include "./CudaConnectedComponents/cuda_cc_stats.h"
#include "./CudaConnectedComponents/cuda_cc_filter.h"

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        throw std::runtime_error(std::string("CUDA Error: ") +              \
                                 cudaGetErrorString(err));                  \
    }                                                                      \
} while (0)

#define NPP_CHECK(call)                                                     \
do {                                                                       \
    NppStatus status = (call);                                              \
    if (status != NPP_SUCCESS) {                                            \
        throw std::runtime_error("NPP Error, status = " +                   \
                                 std::to_string((int)status));              \
    }                                                                      \
} while (0)

int main()
{
    try
    {
        const int width = 16;
        const int height = 12;

        cv::Mat hMask(height, width, CV_8UC1, cv::Scalar(0));

        cv::rectangle(hMask, cv::Rect(2, 2, 3, 3), cv::Scalar(255), cv::FILLED);
        cv::rectangle(hMask, cv::Rect(10, 2, 4, 2), cv::Scalar(255), cv::FILLED);
        cv::rectangle(hMask, cv::Rect(6, 8, 5, 3), cv::Scalar(255), cv::FILLED);

        std::cout << "Input Binary:" << std::endl;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
                std::cout << (hMask.at<uchar>(y, x) > 0 ? "1 " : ". ");
            std::cout << std::endl;
        }

        NppiSize roi{ width, height };

        const int srcStep = width * sizeof(Npp8u);
        const int labelStep = width * sizeof(Npp32u);

        Npp8u* dMask = nullptr;
        Npp32u* dLabel = nullptr;

        CUDA_CHECK(cudaMalloc(&dMask, width * height * sizeof(Npp8u)));
        CUDA_CHECK(cudaMalloc(&dLabel, width * height * sizeof(Npp32u)));

        CUDA_CHECK(cudaMemcpy(
            dMask,
            hMask.data,
            width * height * sizeof(Npp8u),
            cudaMemcpyHostToDevice
        ));

        int labelBufferSize = 0;

        NPP_CHECK(
            nppiLabelMarkersUFGetBufferSize_32u_C1R(
                roi,
                &labelBufferSize
            )
        );

        Npp8u* dLabelBuffer = nullptr;
        CUDA_CHECK(cudaMalloc(&dLabelBuffer, labelBufferSize));

        NPP_CHECK(
            nppiLabelMarkersUF_8u32u_C1R(
                dMask,
                srcStep,
                dLabel,
                labelStep,
                roi,
                nppiNormInf,
                dLabelBuffer
            )
        );

        CUDA_CHECK(cudaFree(dLabelBuffer));

        int nStartingNumber = width * height;
        int compressBufferSize = 0;

        NPP_CHECK(
            nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
                nStartingNumber,
                &compressBufferSize
            )
        );

        Npp8u* dCompressBuffer = nullptr;
        CUDA_CHECK(cudaMalloc(&dCompressBuffer, compressBufferSize));

        int maxLabel = 0;

        NPP_CHECK(
            nppiCompressMarkerLabelsUF_32u_C1IR(
                dLabel,
                labelStep,
                roi,
                nStartingNumber,
                &maxLabel,
                dCompressBuffer
            )
        );

        CUDA_CHECK(cudaFree(dCompressBuffer));

        std::cout << "\nMax Label = " << maxLabel << std::endl;

        RegionStat* dStats = nullptr;

        computeRegionStatsCUDA(
            dLabel,
            width,
            height,
            maxLabel,
            &dStats
        );

        std::vector<RegionStat> hStats =
            downloadRegionStatsCUDA(dStats, maxLabel);

        std::cout << "\nRegion Statistics:" << std::endl;
        for (int i = 1; i <= maxLabel; ++i)
        {
            int bw = hStats[i].xmax - hStats[i].xmin + 1;
            int bh = hStats[i].ymax - hStats[i].ymin + 1;
            float aspect = static_cast<float>(bw) / static_cast<float>(bh);

            std::cout << "Label " << i
                << " | Area = " << hStats[i].area
                << " | BBox = ("
                << hStats[i].xmin << ", "
                << hStats[i].ymin << ") - ("
                << hStats[i].xmax << ", "
                << hStats[i].ymax << ")"
                << " | Aspect = " << aspect
                << std::endl;
        }

        const int minArea = 10;
        const int maxArea = -1;
        const float minAspect = 0.5f;
        const float maxAspect = 3.0f;

        unsigned char* dRemoveMask = nullptr;

        computeRemoveMaskCUDA(
            dStats,
            maxLabel,
            minArea,
            maxArea,
            minAspect,
            maxAspect,
            &dRemoveMask
        );

        std::vector<unsigned char> hRemoveMask =
            downloadRemoveMaskCUDA(dRemoveMask, maxLabel);

        std::cout << "\nCC_FILTER Parameters:" << std::endl;
        std::cout << "minArea   = " << minArea << std::endl;
        std::cout << "maxArea   = " << maxArea << std::endl;
        std::cout << "minAspect = " << minAspect << std::endl;
        std::cout << "maxAspect = " << maxAspect << std::endl;

        std::cout << "\nRemove Mask:" << std::endl;

        for (int i = 0; i <= maxLabel; ++i)
        {
            std::cout << "Label " << i
                << " remove = "
                << static_cast<int>(hRemoveMask[i])
                << std::endl;
        }

        freeRemoveMaskCUDA(dRemoveMask);
        freeRegionStatsCUDA(dStats);

        CUDA_CHECK(cudaFree(dMask));
        CUDA_CHECK(cudaFree(dLabel));

        std::cout << "\nPhase 4 Test Success." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nError: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}