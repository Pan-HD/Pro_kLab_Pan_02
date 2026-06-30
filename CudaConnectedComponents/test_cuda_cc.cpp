#include "./CudaConnectedComponents/cuda_cc.h"

using namespace cv;
using namespace cv::cuda;

int main()
{
    Mat binary = imread("binary.png", IMREAD_GRAYSCALE);

    cv::threshold(binary, binary, 128, 255, THRESH_BINARY);

    GpuMat dBinary(binary);

    //------------------------------------
    // Label
    //------------------------------------

    GpuMat dLabel =
        cuda_cc::executeLabelCUDA(dBinary);

    //------------------------------------
    // Download
    //------------------------------------

    Mat label;

    dLabel.download(label);

    double maxLabel;

    minMaxLoc(label,
        nullptr,
        &maxLabel);

    std::cout
        << "Max Label = "
        << maxLabel
        << std::endl;

    return 0;
}