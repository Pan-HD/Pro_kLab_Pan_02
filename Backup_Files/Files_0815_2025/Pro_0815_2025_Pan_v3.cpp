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

#define numSets 4 // the num of sets(pairs)
#define idSet 5 // for mark the selected set if the numSets been set of 1
#define POP_SIZE 1

void imgShow(const string& name, const Mat& img);
double calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], int numInd);

// for storing the f-value of every individual in the group
double indFValInfo[POP_SIZE][numSets + 1];
int indFValFlag = 0;

int main(void) {
	Mat imgArr[numSets][2]; // imgArr -> storing all images numSets(numSets pairs) * 2(ori, tar)
	char inputPathName_ori[256];
	char inputPathName_tar[256];

	if (numSets == 1) {
		sprintf_s(inputPathName_ori, "./imgs_0815_2025_v3/input/oriImg_0%d.png", idSet);
		sprintf_s(inputPathName_tar, "./imgs_0815_2025_v3/input/tarImg_0%d.png", idSet);
		for (int j = 0; j < 2; j++) {
			if (j == 0) {
				imgArr[0][j] = imread(inputPathName_ori, 0);
			}
			else {
				imgArr[0][j] = imread(inputPathName_tar, 0);
			}
		}
	}
	else {
		for (int i = 0; i < numSets; i++) {
			sprintf_s(inputPathName_ori, "./imgs_0815_2025_v3/input/oriImg_0%d.png", i + 5);
			sprintf_s(inputPathName_tar, "./imgs_0815_2025_v3/input/tarImg_0%d.png", i + 5);
			for (int j = 0; j < 2; j++) {
				if (j == 0) {
					imgArr[i][j] = imread(inputPathName_ori, 0);
				}
				else {
					imgArr[i][j] = imread(inputPathName_tar, 0);
				}
			}
		}
	}

	Mat metaImg[numSets];
	Mat tarImg[numSets];

	for (int idxSet = 0; idxSet < numSets; idxSet++) {
		metaImg[idxSet] = imgArr[idxSet][0].clone();
		tarImg[idxSet] = imgArr[idxSet][1].clone();
	}

	calculateMetrics(metaImg, tarImg, 0);
	for (int idx = 0; idx <= numSets; idx++) {
		printf("idx-%d: %.4f\n", idx, indFValInfo[0][idx]);
	}

	return 0;
}

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
double calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], int numInd) {
	double f1_score[numSets];
	for (int idxSet = 0; idxSet < numSets; idxSet++) {
		int tp = 0, fp = 0, fn = 0;
		for (int i = 0; i < metaImg_g[idxSet].rows; i++) {
			for (int j = 0; j < metaImg_g[idxSet].cols; j++) {
				// if the metaImg processed without threshFunc, then return min-score directly.
				if (metaImg_g[idxSet].at<uchar>(i, j) != 0 && metaImg_g[idxSet].at<uchar>(i, j) != 255) {
					return 0.01;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 0) {
					tp += 1;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 255) {
					fp += 1;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 255 && tarImg_g[idxSet].at<uchar>(i, j) == 0) {
					fn += 1;
				}
			}
		}
		if (tp == 0) tp += 1;
		if (fp == 0) fp += 1;
		if (fn == 0) fn += 1;
		double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
		double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
		f1_score[idxSet] = calculateF1Score(precision, recall);
	}
	double sum_f1 = 0.0;

	indFValFlag = 1;

	for (int idxSet = 0; idxSet < numSets; idxSet++) {
		if (indFValFlag) { // in the last generation
			indFValInfo[numInd][idxSet] = f1_score[idxSet];
		}
		sum_f1 += f1_score[idxSet];
	}

	if (indFValFlag) {
		indFValInfo[numInd][numSets] = sum_f1;
	}

	return sum_f1;
}