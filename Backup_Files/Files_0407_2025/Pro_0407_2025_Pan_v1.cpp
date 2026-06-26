#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// the name of decision-variables
// ["filterSwitchFlag", "fsize", "absoluteFlag", "threshVal"]
// ["pixelLabelingMethod", "linear", "erodeDilateSequence", "erodeDilateTimes"]
int info_val_dv[8] = { 1, 13, 0, 33, 0, 26, 1, 0 };

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

void labeling(Mat img, Mat& resImg, int connectivity, int linear) {
	Mat img_con;
	Mat stats, centroids;

	int label_x, label_y;
	int label_longer;
	int label_areaall;
	double label_cal;

	int i, j, label_num;

	label_num = cv::connectedComponentsWithStats(img, img_con, stats, centroids, connectivity, CV_32S);
	// colors: for storing the color of background and every connected areas 
	vector<Vec3b>colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);
	colors[1] = Vec3b(255, 255, 255);

	for (i = 2; i <= label_num; i++)
	{
		// colors[i] = Vec3b(0, 0, 0);
		label_areaall = stats.at<int>(i, CC_STAT_AREA);
		label_x = stats.at<int>(i, CC_STAT_WIDTH);
		label_y = stats.at<int>(i, CC_STAT_HEIGHT);

		label_longer = label_x > label_y ? label_x : label_y;
		label_cal = label_longer * label_longer;

		// (int)(label_cal / label_areaall) < linear -> detected area is not a fold-area
		//  In fold-detect task: discard -> colors[i] = Vec3b(255, 255, 255);
		if ((int)(label_cal / label_areaall) < linear + 127)
		{
			colors[i] = Vec3b(0, 0, 0); // in spot-detect task
		}
		else {
			colors[i] = Vec3b(255, 255, 255); // in spot-detect task
		}
	}

	// CV_8UC3: 3 channels
	Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
	for (j = 0; j < img_con.rows; j++) {
		for (i = 0; i < img_con.cols; i++)
		{
			int label = img_con.at<int>(j, i);
			CV_Assert(0 <= label && label <= label_num); // make sure the num of label is leagal
			img_color.at<Vec3b>(j, i) = colors[label];
		}
	}
	cvtColor(img_color, img_color, COLOR_RGB2GRAY);
	resImg = img_color.clone();
}

void Morphology(Mat img, Mat& resImg, int isDilFirst) {
	Mat dst;
	dst.create(img.size(), img.type());
	if (isDilFirst) {
		dilate(img, dst, Mat());
		erode(dst, dst, Mat());
	}
	else {
		erode(img, dst, Mat());
		dilate(dst, dst, Mat());
	}
	resImg = dst.clone();
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

	if (!arr_val_dv[4])
	{
		labeling(biImg, labelImg, 4, arr_val_dv[5]);
	}
	else
	{
		labeling(biImg, labelImg, 8, arr_val_dv[5]);
	}

	// erodeDilateSequence: 0 -> dilate first, 1 -> erode first
	if (!arr_val_dv[6]) // 0 -> dilate first
	{
		if (arr_val_dv[7] != 0) {
			for (int idx_edt = 0; idx_edt < arr_val_dv[7]; idx_edt++)
			{
				Morphology(labelImg, labelImg, 1);
			}
		}
	}
	else // 1 -> erode first
	{
		if (arr_val_dv[7] != 0) {
			for (int idx_edt = 0; idx_edt < arr_val_dv[7]; idx_edt++)
			{
				Morphology(labelImg, labelImg, 0);
			}
		}
	}
	resImg = labelImg.clone();
}

int main(void) {
	Mat oriImg = imread("./imgs_0321_2025_v1/input/oriImg_02.png", IMREAD_GRAYSCALE);
	Mat resImg;
	imgSingleProcess(oriImg, resImg, info_val_dv);
	imgShow("res", resImg);
	return 0;
}