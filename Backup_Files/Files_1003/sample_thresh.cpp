#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

Mat imgResize(Mat img, int width, int height) {
	Mat resImg;
	Size size(width, height);
	resize(img, resImg, size);
	return resImg;
}

void func1(Mat);

int main(void) {
	Mat img = imread("./imgs_test/01.png");
	Mat img_g = imread("./imgs_test/01.png", 0);
	func1(img_g);
}

void func1(Mat img_g) {
	Mat img_blur;
	GaussianBlur(img_g, img_blur, Size(3, 3), 0);
	//vector<Mat> images = { img_g, img_blur };
	//Mat res;
	//hconcat(images, res);
	//imgShow("blur", res);
	Mat img_bi;
	threshold(img_blur, img_bi, 195, 255, THRESH_BINARY);
	vector<Mat> images = { img_g, img_bi };
	Mat res;
	hconcat(images, res);
	imgShow("bi", res);
	imwrite("./imgs_test/target.png", img_bi);
}