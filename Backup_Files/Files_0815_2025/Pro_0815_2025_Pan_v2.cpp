#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

double calculateF1Score(double precision, double recall) {
	if (precision + recall == 0) return 0.0;
	return 2.0 * (precision * recall) / (precision + recall);
}

// for calculating the fValue of the ind and writting the organized info into group-arr and groupDvInfoArr
double calculateMetrics(Mat metaImg, Mat tarImg) {
	double f1_value;

	threshold(metaImg, metaImg, 127, 255, THRESH_BINARY);
	imgShow("res", metaImg);

	int tp = 0, fp = 0, fn = 0;
	for (int i = 0; i < metaImg.rows; i++) {
		for (int j = 0; j < metaImg.cols; j++) {
			if (metaImg.at<uchar>(i, j) == 0 && tarImg.at<uchar>(i, j) == 0) {
				tp += 1;
			}
			if (metaImg.at<uchar>(i, j) == 0 && tarImg.at<uchar>(i, j) == 255) {
				fp += 1;
			}
			if (metaImg.at<uchar>(i, j) == 255 && tarImg.at<uchar>(i, j) == 0) {
				fn += 1;
			}
		}
	}

	printf("the true positive val is: %d\n", tp);
	printf("the false positive val is: %d\n", fp);
	printf("the false negative val is: %d\n", fn);

	if (tp == 0) tp += 1;
	if (fp == 0) fp += 1;
	if (fn == 0) fn += 1;
	double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
	double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
	return calculateF1Score(precision, recall);
}

int main(void) {
	Mat resImg = imread("./imgs_0815_2025_v1/output/img_07/Gen-1000.png", 0);
	Mat tarImg = imread("./imgs_0815_2025_v1/input/tarImg_07.png", 0);

	//imgShow("res1", resImg);
	//imgShow("res2", tarImg);

	double f1_value = calculateMetrics(resImg, tarImg);
	printf("the f1_val is: %.4f\n", f1_value);

	return 0;
}