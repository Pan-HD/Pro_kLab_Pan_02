#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int threshVal = 17; // [0, 255] -> dv01 - 8bit
int gaussianSize = 17; // (n * 2 + 1) 1, 3, 5, 7, 9, ..., 31 -> dv02 - 4bit
int circleOffset = 9; // [0, 15] -> dv03 - 4bit
int medianSize = 3; // -> dv04 - 4bit
int dilateTimes_t1 = 1; // -> dv05 - 2bit 
int aspectOffset_t1 = 1; // [0, 7] -> dv06 - 3bit
int contourPixNums_t1 = 7; // [0, 7] -> dv07 - 3bit
int dilateTimes_t2 = 1; // [0, 3] -> dv08 - 2bit
int aspectOffset_t2 = 0; // [0, 7] -> dv09 - 3bit
int contourPixNums_t2 = 3; // [0, 7] -> dv10 - 3bit

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

vector<Vec3f> circleDetect(Mat img) {
	Mat blurred;
	GaussianBlur(img, blurred, Size(gaussianSize, gaussianSize), 0, 0);
	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, blurred.rows / 8, 200, 100, 0, 0);
	return circles;
}

int comDistance(int y, int x, Vec3f circle) {
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

void contourProcess(Mat& metaImg, Mat& resImg, int aspectRatio, int pixNums, vector<Vec3f> circles) {
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
				if (comDistance(y, x, circles[0]) == 2) {
					if (mask.at<uchar>(y, x) == 255) {
						resImg.at<uchar>(y, x) = 255;
					}
				}
			}
		}
	}
	// imgShow("res", resImg);
}

int main(void) {
	Mat oriImg = imread("./imgs_1225_v1/input/oriImg_02.png", IMREAD_GRAYSCALE);
	//imgShow("res", oriImg);

	Mat edges_s1;
	gradCal(oriImg, edges_s1); // stat-01 -> Sobel
	//imgShow("res", edges_s1);

	Mat biImg;
	threshold(edges_s1, biImg, threshVal, 255, THRESH_BINARY); // stat-02 -> threshold
	// imgShow("res", biImg);

	vector<Vec3f> circles = circleDetect(biImg); // GaussianSize

	if (circles.size() != 0) { // stat-03
		for (int y = 0; y < biImg.rows; y++) {
			for (int x = 0; x < biImg.cols; x++) {
				if (comDistance(y, x, circles[0]) == 0) { // offset
					biImg.at<uchar>(y, x) = 0;
				}
				else if (comDistance(y, x, circles[0]) == 1) {
					biImg.at<uchar>(y, x) = 255;
				}
				else {
					biImg.at<uchar>(y, x) = biImg.at<uchar>(y, x) == 0 ? 255 : 0;
				}
			}
		}
	}
	//imgShow("test", biImg);

	Mat blurImg_mask;
	medianBlur(biImg, blurImg_mask, 3);

	if (erodeFlag) {
		// For img-04
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		for (int i = 0; i < erodeTimes; i++) {
			erode(blurImg_mask, blurImg_mask, kernel);
		}
	}
	// imgShow("test", blurImg_mask);

	vector<vector<Point>> contours;
	findContours(blurImg_mask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(blurImg_mask.size(), CV_8UC1);
	for (const auto& contour : contours) {
		Rect bounding_box = boundingRect(contour);
		double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
		if ((aspect_ratio < (1 - aspectRatio * 0.1) || aspect_ratio >(1 + aspectRatio * 0.1)) && cv::contourArea(contour) < 100 * contPixNums) {
			drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
		}
	}
	// imgShow("test", mask);

	if (circles.size() != 0) {
		for (int y = 0; y < biImg.rows; y++) {
			for (int x = 0; x < biImg.cols; x++) {
				if (comDistance(y, x, circles[0]) == 2) {
					if (mask.at<uchar>(y, x) == 255) {
						biImg.at<uchar>(y, x) = 255;
					}
				}
			}
		}
	}
	imgShow("res", biImg);
	return 0;
}