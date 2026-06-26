#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/*
int main() {
	// 读取两张图片
	Mat img1 = imread("binary_image.png", IMREAD_GRAYSCALE); // 教師画像
	Mat img2 = imread("heikin.png", IMREAD_GRAYSCALE);     

	if (img1.empty() || img2.empty()) {
		cerr << "Error: Can't read one of the images." << endl;
		return -1;
	}

	// 确保两张图片大小一致
	if (img1.size() != img2.size()) {
		cerr << "Error: Images must have the same dimensions." << endl;
		return -1;
	}

	// 阈值处理，将图像二值化（确保所有像素为0或255）
	Mat binary_img1, binary_img2;
	threshold(img1, binary_img1, 127, 255, THRESH_BINARY);
	threshold(img2, binary_img2, 127, 255, THRESH_BINARY);

	// 计算真阳性（TP）、假阳性（FP）、假阴性（FN）
	int TP = 0, FP = 0, FN = 0;

	for (int i = 0; i < binary_img1.rows; i++) {
		for (int j = 0; j < binary_img1.cols; j++) {
			uchar pixel1 = binary_img1.at<uchar>(i, j);
			uchar pixel2 = binary_img2.at<uchar>(i, j);

			if (pixel1 == 255 && pixel2 == 255) {
				TP++; // True Positive (TP) - Both images have white pixels
			}
			else if (pixel1 == 0 && pixel2 == 255) {
				FP++; // False Positive (FP) - Ground truth is black, compared image is white
			}
			else if (pixel1 == 255 && pixel2 == 0) {
				FN++; // False Negative (FN) - Ground truth is white, compared image is black
			}
		}
	}

	// 计算精确率（Precision）、再现率（Recall）、F值（F1-score）
	double precision = TP + FP > 0 ? TP / double(TP + FP) : 0.0;
	double recall = TP + FN > 0 ? TP / double(TP + FN) : 0.0;
	double f1_score = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

	cout << "適合率(Precision): " << precision << endl;
	cout << "再現率(Recall): " << recall << endl;
	cout << "F値: " << f1_score << endl;

	return 0;
}
*/