#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip> // for std::setprecision

/*
	Function: 精度と再現率の調和平均を取ることで、両者のバランスを取った指標として機能します。
			  調和平均を取ることで、片方だけが高くてもF値は高くなりません。
*/
double calculateF1Score(double precision, double recall) {
	if (precision + recall == 0) return 0.0;
	return 2.0 * (precision * recall) / (precision + recall);
}

/*
	Function:
	Params: image1 -> oriImg, image2 -> tarImg
*/
void calculateMetrics(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat* mask = nullptr) {
	// tp: True Positive, 正しく検出のピクセル数
	// fp: False Positive, 誤検出のピクセル数
	// fn: False Negative, 未検出のピクセル数
	int tp = 0, fp = 0, fn = 0;
	int mask_pixel_count = 0;  // マスク中で値が255のピクセル数をカウントする変数

	for (int y = 0; y < image1.rows; y++) {
		for (int x = 0; x < image1.cols; x++) {
			// マスクが存在する場合、マスクの白い部分のみ計算
			if (mask && mask->at<uchar>(y, x) != 255) {
				continue;
			}

			if (mask && mask->at<uchar>(y, x) == 255) {
				mask_pixel_count++;
			}

			bool isImage1White = (image1.at<uchar>(y, x) == 255);
			bool isImage2White = (image2.at<uchar>(y, x) == 255);

			if (!isImage1White && !isImage2White) {
				tp++; // 真陽性
			}
			else if (!isImage1White && isImage2White) {
				fp++; // 偽陽性
			}
			else if (isImage1White && !isImage2White) {
				fn++; // 偽陰性
			}
		}
	}

	// precision: 検出部分の中、正確な部分の割合
	double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
	// recall: 検出すべき部分の中、実際の検出部分の割合
	double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
	double f1_score = calculateF1Score(precision, recall);

	// 結果を表示
	if (mask) {
		std::cout << "マスク部分の結果:" << std::endl;
		std::cout << "マスク中の255のピクセル数: " << mask_pixel_count << std::endl;
	}
	else {
		std::cout << "全体画像の結果:" << std::endl;
	}

	std::cout << std::fixed << std::setprecision(20); // 固定表示と小数点以下20桁に設定
	std::cout << "適応率 (Precision): " << precision << std::endl;
	std::cout << "再現率 (Recall): " << recall << std::endl;
	std::cout << "F値 (F1 Score): " << f1_score << std::endl;
	std::cout << "正解数 (True Positives): " << tp << std::endl;
	std::cout << "欠損画素数 (False Negatives): " << fn << std::endl;
}

int main() {
	// 画像を読み込む: image1 -> oriImg, image2 -> tarImg, image3 -> maskImg
	cv::Mat image1 = cv::imread("./imgs_1007_v1/sedai89エリート.png", cv::IMREAD_GRAYSCALE);
	cv::Mat image2 = cv::imread("./imgs_1007_v1/kyoushi1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask = cv::imread("./imgs_1007_v1/mask.png", cv::IMREAD_GRAYSCALE);

	if (image1.empty() || image2.empty() || mask.empty()) {
		std::cerr << "画像の読み込みに失敗しました" << std::endl;
		return -1;
	}

	// 全体画像の結果を計算・表示
	calculateMetrics(image1, image2);

	// マスク部分の結果を計算・表示
	calculateMetrics(image1, image2, &mask);

	return 0;
}
