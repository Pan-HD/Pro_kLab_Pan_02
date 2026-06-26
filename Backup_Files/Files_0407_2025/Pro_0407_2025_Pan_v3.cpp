#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// the name of decision-variables
// ["threshVal", "gaussianSize", "circleOffset", "meidanSize", "dilateTimes_01"]
// ["aspectOffset_01", "contourPixNums_01", "dilateTimes_02", "aspectOffset_02", "contourPixNum_02"]
int info_val_dv[10] = { 18, 23, 11, 3, 1, 5, 7, 2, 0, 7 };


// for Set-01
// int info_val_dv[10] = { 17, 13, 10, 5, 1, 0, 6, 1, 0, 4 };
// int info_val_dv[10] = { 17, 17, 10, 3, 0, 1, 7, 1, 0, 5 };

// for Set-02
// int info_val_dv[10] = { 17, 15, 10, 3, 2, 0, 7, 2, 0, 5 };
// int info_val_dv[10] = { 17, 11, 8, 3, 1, 2, 3, 1, 0, 4 };

// for Com of Set-01 and Set-02
// int info_val_dv[10] = { 18, 19, 9, 3, 1, 3, 7, 1, 0, 6 };

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

/*
  function: Sobel processing
*/
void gradCal(Mat& srcImg, Mat& dstImg) {
	Mat sobelX, sobelY, gradientMagnitude;
	Sobel(srcImg, sobelX, CV_64F, 1, 0, 1);
	Sobel(srcImg, sobelY, CV_64F, 0, 1, 1);
	magnitude(sobelX, sobelY, gradientMagnitude);
	normalize(gradientMagnitude, dstImg, 0, 255, NORM_MINMAX, CV_8U);
}

vector<Vec3f> circleDetect(Mat img, int gaussianSize) {
	Mat blurred;
	// GaussianBlur(img, blurred, Size(gaussianSize, gaussianSize), 0, 0);
	GaussianBlur(img, blurred, Size(gaussianSize, gaussianSize), 0, 0);
	// imgShow("res", blurred);
	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, blurred.rows / 8, 200, 100, 0, 0);
	return circles;
}

int comDistance(int y, int x, Vec3f circle, int circleOffset) {
	int centerX = (int)circle[0];
	int centerY = (int)circle[1];
	int radius = (int)circle[2];
	int distance = (int)sqrt(pow((double)(x - centerX), 2) + pow((double)(y - centerY), 2));
	if (distance > radius) {
		return 0;
	}
	else if (distance > radius - circleOffset && distance <= radius) {
		return 1;
	}
	else {
		return 2;
	}
}

void contourProcess(Mat& metaImg, Mat& resImg, int aspectRatio, int pixNums, vector<Vec3f> circles, int circleOffset) {
	vector<vector<Point>> contours;
	findContours(metaImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(metaImg.size(), CV_8UC1);
	for (const auto& contour : contours) {
		Rect bounding_box = boundingRect(contour);
		double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
		if ((aspect_ratio <= (1 - aspectRatio * 0.1) || aspect_ratio > (1 + aspectRatio * 0.1)) && cv::contourArea(contour) < pixNums) {
			drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
		}
	}
	// imgShow("mask", mask);

	if (circles.size() != 0) {
		for (int y = 0; y < resImg.rows; y++) {
			for (int x = 0; x < resImg.cols; x++) {
				if (comDistance(y, x, circles[0], circleOffset) == 2) {
					if (mask.at<uchar>(y, x) == 255) {
						resImg.at<uchar>(y, x) = 255;
					}
				}
			}
		}
	}
	// imgShow("res", resImg);
}

void imgSingleProcess(Mat& oriImg, Mat& resImg, int arr_val_dv[]) {
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	Mat metaImg;

	Mat edges_s1;
	gradCal(oriImg, edges_s1); // stat-01 -> Sobel
	// imgShow("res", edges_s1);

	Mat biImg;
	threshold(edges_s1, biImg, arr_val_dv[0], 255, THRESH_BINARY); // stat-02 -> threshold
	// imgShow("res", biImg);

	bitwise_not(biImg, biImg);
	// imgShow("res", biImg);

	vector<Vec3f> circles = circleDetect(biImg, arr_val_dv[1]); // GaussianSize

	if (circles.size() != 0) { // stat-03
		for (int y = 0; y < biImg.rows; y++) {
			for (int x = 0; x < biImg.cols; x++) {
				if (comDistance(y, x, circles[0], arr_val_dv[2]) != 2) {
					biImg.at<uchar>(y, x) = 255;
				}
			}
		}
	}
	// imgShow("test", biImg);

	Mat blurImg_mask;
	medianBlur(biImg, blurImg_mask, arr_val_dv[3]);
	// imgShow("test", blurImg_mask);

	for (int idxET = 0; idxET < arr_val_dv[4]; idxET++) {
		erode(blurImg_mask, blurImg_mask, kernel);
	}
	// imgShow("test", blurImg_mask);

	contourProcess(blurImg_mask, biImg, arr_val_dv[5], 100 * arr_val_dv[6], circles, arr_val_dv[2]);
	// imgShow("res", biImg);

	metaImg = biImg.clone();
	for (int idxET = 0; idxET < arr_val_dv[7]; idxET++) {
		erode(metaImg, metaImg, kernel);
	}
	contourProcess(metaImg, biImg, arr_val_dv[8], 100 * arr_val_dv[9], circles, arr_val_dv[2]);
	// imgShow("res", biImg);
	resImg = biImg.clone();
}

int main(void) {
	Mat oriImg = imread("./imgs_0407_2025_v3/input/oriImg_04.png", IMREAD_GRAYSCALE);
	// imgShow("res", oriImg);
	Mat resImg;
	imgSingleProcess(oriImg, resImg, info_val_dv);
	imgShow("res", resImg);
	return 0;
}