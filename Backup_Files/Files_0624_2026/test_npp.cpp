#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "CudaConnectedComponents.h"

using namespace std;
using namespace cv;

int main()
{
    //-----------------------------------------
    // 建立一个简单Binary Image
    //-----------------------------------------

    Mat img = Mat::zeros(6, 8, CV_8UC1);

    img.at<uchar>(1, 1) = 255;
    img.at<uchar>(1, 2) = 255;

    img.at<uchar>(2, 1) = 255;
    img.at<uchar>(2, 2) = 255;

    img.at<uchar>(4, 5) = 255;
    img.at<uchar>(4, 6) = 255;

    img.at<uchar>(5, 5) = 255;
    img.at<uchar>(5, 6) = 255;

    cout << "Binary Image:" << endl;
    cout << img << endl;

    //-----------------------------------------
    // Upload
    //-----------------------------------------

    cuda::GpuMat gpuSrc;

    gpuSrc.upload(img);

    //-----------------------------------------
    // Connected Components
    //-----------------------------------------

    cuda::GpuMat gpuLabels;

    int numLabels = 0;

    bool ok =
        cudaConnectedComponents(
            gpuSrc,
            gpuLabels,
            numLabels);

    if (!ok)
    {
        cout << "CCL Failed." << endl;
        return -1;
    }

    //-----------------------------------------
    // Download
    //-----------------------------------------

    Mat labels;

    gpuLabels.download(labels);

    cout << endl;
    cout << "Label Image :" << endl;

    for (int y = 0; y < labels.rows; y++)
    {
        for (int x = 0; x < labels.cols; x++)
        {
            cout << labels.at<int>(y, x) << "\t";
        }
        cout << endl;
    }

    cout << endl;
    cout << "Num Labels = "
        << numLabels
        << endl;

    return 0;
}