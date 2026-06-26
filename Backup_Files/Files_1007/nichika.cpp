#include <opencv2/opencv.hpp>
#include <iostream>
/*
int main() {

	cv::Mat gray_image = cv::imread("12345.png", cv::IMREAD_GRAYSCALE);


	if (gray_image.empty()) {
		std::cerr << "エラー" << std::endl;
		return -1;
	}

	cv::Mat binary_image;
	int threshold_value = 190;
	cv::threshold(gray_image, binary_image, threshold_value, 255, cv::THRESH_BINARY);

	if (!cv::imwrite("1234567.png", binary_image)) {
		std::cerr << "できない" << std::endl;
		return -1;
	}

	std::cout << "できた！ 'binary_image.png'が保存されました．" << std::endl;

	return 0;
}
*/