#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// the declaration of 8 decision variables (with the sum-fVal of 3.5645)
int fsize = 27;
int binary = 63;
int linear = 5;
int filterswitch_flag = 1;
int erodedilate_times = 5;
int erodedilate_sequence = 1;
int abusolute_flag = 0;
int pixellabelingmethod = 1;

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void differenceProcess(Mat postImg, Mat preImg, Mat &resImg) {
	resImg = Mat::zeros(Size(postImg.cols, postImg.rows), CV_8UC1);
	for (int j = 0; j < postImg.rows; j++)
	{
		for (int i = 0; i < postImg.cols; i++) {
			int diffVal = postImg.at<uchar>(j, i) - preImg.at<uchar>(j, i);
			if (diffVal < 0) {
				if (abusolute_flag != 0) {
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

void labeling(Mat img, Mat &resImg, int connectivity) {
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

vector<Vec3f> circleDetect(Mat img, int gaussianSize) {
	Mat blurred;
	// GaussianBlur(img, blurred, Size(gaussianSize, gaussianSize), 0, 0);
	GaussianBlur(img, blurred, Size(gaussianSize, gaussianSize), 0, 0);
	// imgShow("res", blurred);

	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, blurred.rows / 8, 200, 100, 0, 0);
	return circles;
}

void Morphology(Mat img, Mat &resImg, int isDilFirst) {
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

int main(void) {
	Mat oriImg = imread("./imgs_0321_2025_v1/input/oriImg_02.png", IMREAD_GRAYSCALE);

	// imgShow("ori", oriImg);

	Mat blurImg;
	Mat diffImg;
	Mat biImg;
	Mat labelImg;

	// bluring
	if (filterswitch_flag) {
		medianBlur(oriImg, blurImg, fsize); // marked
	}
	else {
		blur(oriImg, blurImg, Size(fsize, fsize));
	}
	// imgShow("res", blurImg);

	differenceProcess(blurImg, oriImg, diffImg);
	// imgShow("res", diffImg);

	threshold(diffImg, biImg, binary, 255, THRESH_BINARY);
	// imgShow("res", biImg);

	if (!pixellabelingmethod)
	{
		labeling(biImg, labelImg, 4);
	}
	else
	{
		labeling(biImg, labelImg, 8);
	}
	// imgShow("res", labelImg);

	bitwise_not(labelImg, labelImg);
	// imgShow("res", labelImg);

	vector<Vec3f> circles = circleDetect(oriImg, 11); // GaussianSize
	if (circles.size() != 0) { // (int)circles[0][2]
		for (int y = 0; y < oriImg.rows; y++) {
			for (int x = 0; x < oriImg.cols; x++) {
				int distance = (int)sqrt(pow((double)(x - (int)circles[0][0]), 2) + pow((double)(y - (int)circles[0][1]), 2));
				if (distance > (int)circles[0][2])
					labelImg.at<uchar>(y, x) = 0;
			}
		}
	}
	// imgShow("res", labelImg);

	// Morphology
	if (!erodedilate_sequence)
	{
		if (erodedilate_times != 0) {
			for (int idx_edt = 0; idx_edt < erodedilate_times; idx_edt++)
			{
				Morphology(labelImg, labelImg, 1);
			}
		}
	}
	else
	{
		if (erodedilate_times != 0) {
			for (int idx_edt = 0; idx_edt < erodedilate_times; idx_edt++)
			{
				Morphology(labelImg, labelImg, 0);
			}
		}
	}
	// imgShow("res", labelImg);

	return 0;
}