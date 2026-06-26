#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// the name of decision-variables
// ["filterSwitchFlag", "fsize", "absoluteFlag", "threshVal"]
// ["dilateTimes", "aspectOffset", "contourPixNum"]
// int info_val_dv[7] = { 1, 17, 0, 9, 3, 0, 2 };
int info_val_dv[7] = { 1, 23, 0, 9, 3, 1, 5 }; // Opt of GA for Rep on 0806

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void differenceProcess(Mat postImg, Mat preImg, Mat& resImg, int absoluteFlag) {
	resImg = Mat::zeros(Size(postImg.cols, postImg.rows), CV_8UC1);
	for (int j = 0; j < postImg.rows; j++)
	{
		for (int i = 0; i < postImg.cols; i++) {
			int diffVal = postImg.at<uchar>(j, i) - preImg.at<uchar>(j, i);
			if (diffVal < 0) {
				if (absoluteFlag != 0) {
					diffVal = abs(diffVal);
				}
				else {
					diffVal = 0;
				}
			}
			resImg.at<uchar>(j, i) = diffVal;
		}
	}
}

void contourProcess(Mat& metaImg, Mat& resImg, int aspectRatio, int pixNums) {
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
	for (int y = 0; y < resImg.rows; y++) {
		for (int x = 0; x < resImg.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
				resImg.at<uchar>(y, x) = 255;
			}
		}
	}
}

void imgSingleProcess(Mat& oriImg, Mat& resImg, int arr_val_dv[]) {
	Mat blurImg;
	Mat diffImg;
	Mat biImg;
	Mat labelImg;
	if (arr_val_dv[0]) {
		medianBlur(oriImg, blurImg, arr_val_dv[1]);
	}
	else {
		blur(oriImg, blurImg, Size(arr_val_dv[1], arr_val_dv[1]));
	}
	differenceProcess(blurImg, oriImg, diffImg, arr_val_dv[2]);
	threshold(diffImg, biImg, arr_val_dv[3], 255, THRESH_BINARY);
	bitwise_not(biImg, biImg);
	Mat maskImg = biImg.clone();
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	for (int idxET = 0; idxET < arr_val_dv[4]; idxET++) {
		erode(maskImg, maskImg, kernel);
	}
	contourProcess(maskImg, biImg, arr_val_dv[5], 100 * arr_val_dv[6]);
	resImg = biImg.clone();
}

//int main(void) {
//	Mat oriImg = imread("./imgs_0327_2025_v1/input/oriImg_01.png", IMREAD_GRAYSCALE);
//	Mat resImg;
//	// imgShow("ori", oriImg);
//	imgSingleProcess(oriImg, resImg, info_val_dv);
//	// imgShow("resImg", resImg);
//	return 0;
//}

int main(void) {
	Mat oriImg[14];
	Mat resImg[14];
	char inputPathName_ori[14][256];
	char outputPathName_res[14][256];

	for (int idxImg = 0; idxImg < 14; idxImg++) {
		if (idxImg < 9) {
			sprintf_s(inputPathName_ori[idxImg], "./imgs_0806_2025_v0_test/input/oriImg_0%d.png", idxImg + 1);
		}
		else {
			sprintf_s(inputPathName_ori[idxImg], "./imgs_0806_2025_v0_test/input/oriImg_%d.png", idxImg + 1);
		}
		oriImg[idxImg] = imread(inputPathName_ori[idxImg], 0);
	}

	for (int idxImg = 0; idxImg < 14; idxImg++) {
		imgSingleProcess(oriImg[idxImg], resImg[idxImg], info_val_dv);
	}

	for (int idxImg = 0; idxImg < 14; idxImg++) {
		if (idxImg < 9) {
			sprintf_s(outputPathName_res[idxImg], "./imgs_0806_2025_v0_test/output/Ver01/resImg_0%d.png", idxImg + 1);
		}
		else {
			sprintf_s(outputPathName_res[idxImg], "./imgs_0806_2025_v0_test/output/Ver01/resImg_%d.png", idxImg + 1);
		}
		imwrite(outputPathName_res[idxImg], resImg[idxImg]);
	}

	return 0;
}