#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/*
// 線状度判定の関数
bool isLinear(const Mat& stats, int index, double threshold) {
	int width = stats.at<int>(index, CC_STAT_WIDTH);
	int height = stats.at<int>(index, CC_STAT_HEIGHT);
	double ratio = static_cast<double>(max(width, height)) / min(width, height);
	return ratio >= threshold;
}

// 画素ラベリング（4近傍）
Mat labeling_new4(Mat img_sabun, double linear) {
	Mat img_con;
	Mat stats, centroids;
	int label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 4, 4);
	vector<Vec3b> colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);

	for (int i = 1; i < label_num; i++) {
		colors[i] = Vec3b(255, 255, 255);
		int label_areaall = stats.at<int>(i, CC_STAT_AREA);
		int label_x = stats.at<int>(i, CC_STAT_WIDTH);
		int label_y = stats.at<int>(i, CC_STAT_HEIGHT);
		int label_longer = max(label_x, label_y);
		double label_cal = label_longer * label_longer;
		if (label_cal / label_areaall < linear) {
			colors[i] = Vec3b(0, 0, 0);
		}
	}

	Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
	for (int j = 0; j < img_con.rows; j++) {
		for (int i = 0; i < img_con.cols; i++) {
			int label = img_con.at<int>(j, i);
			CV_Assert(0 <= label && label <= label_num);
			img_color.at<Vec3b>(j, i) = colors[label];
		}
	}

	return img_color;
}

// 画素ラベリング（8近傍）
Mat labeling_new8(Mat img_sabun, double linear) {
	Mat img_con;
	Mat stats, centroids;
	int label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 8, 4);
	vector<Vec3b> colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);

	for (int i = 1; i < label_num; i++) {
		colors[i] = Vec3b(255, 255, 255);
		int label_areaall = stats.at<int>(i, CC_STAT_AREA);
		int label_x = stats.at<int>(i, CC_STAT_WIDTH);
		int label_y = stats.at<int>(i, CC_STAT_HEIGHT);
		int label_longer = max(label_x, label_y);
		double label_cal = label_longer * label_longer;
		if (label_cal / label_areaall < linear) {
			colors[i] = Vec3b(0, 0, 0);
		}
	}

	Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
	for (int j = 0; j < img_con.rows; j++) {
		for (int i = 0; i < img_con.cols; i++) {
			int label = img_con.at<int>(j, i);
			CV_Assert(0 <= label && label <= label_num);
			img_color.at<Vec3b>(j, i) = colors[label];
		}
	}

	return img_color;
}

// 差分処理
Mat sabun(Mat input1, Mat input2) {
	Mat output = Mat::zeros(Size(input2.cols, input2.rows), CV_8UC3); // 3チャンネルに変換
	cvtColor(output, output, COLOR_RGB2GRAY); // グレースケール

	for (int j = 0; j < input1.rows; j++) {
		for (int i = 0; i < input1.cols; i++) {
			output.at<unsigned char>(j, i) = input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i);
			if (input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i) < 0) {
				output.at<unsigned char>(j, i) = 0;
			}
		}
	}
	return output;
}

// オープニングとクロージング処理（交互に膨張と収縮）
Mat dilate_erode_alternate(Mat src1, int iterations) {
	Mat dst = src1.clone();
	for (int i = 0; i < iterations; i++) {
		dilate(dst, dst, Mat(), Point(-1, -1), 1);  // 膨張処理
		erode(dst, dst, Mat(), Point(-1, -1), 1);   // 収縮処理
	}
	return dst;
}

// オープニングとクロージング処理（交互に収縮と膨張）
Mat erode_dilate_alternate(Mat src1, int iterations) {
	Mat dst = src1.clone();
	for (int i = 0; i < iterations; i++) {
		erode(dst, dst, Mat(), Point(-1, -1), 1);   // 収縮処理
		dilate(dst, dst, Mat(), Point(-1, -1), 1);  // 膨張処理
	}
	return dst;
}

// 画像の読み込みと処理
void processImage(const std::string &imagePath, int filterType, int filterSize, int binaryThreshold, int erodeDilateOrder, int erodeDilateTimes, double linearThreshold, int neighborType) {
	Mat img = imread(imagePath, IMREAD_GRAYSCALE);
	if (img.empty()) {
		cerr << "エラー：画像を読み込めませんでした！" << endl;
		return;
	}

	// 画像サイズの調整
	resize(img, img, Size(500, 388));

	// フィルタ処理
	Mat img_filtered;
	if (filterType == 1) {
		blur(img, img_filtered, Size(filterSize, filterSize));  // 平均フィルタ
	}
	else if (filterType == 0) {
		medianBlur(img, img_filtered, filterSize);  // メディアンフィルタ
	}
	else if (filterType == 2) {
		GaussianBlur(img, img_filtered, Size(filterSize, filterSize), 0);  // ガウシアンフィルタ
	}

	// 差分処理
	Mat diff = sabun(img, img_filtered);

	// 二値化
	Mat img_binary;
	threshold(diff, img_binary, binaryThreshold, 255, THRESH_BINARY);

	// 画素ラベリングとノイズ除去
	Mat img_labeled;
	if (neighborType == 0) {
		img_labeled = labeling_new4(img_binary, linearThreshold);
	}
	else {
		img_labeled = labeling_new8(img_binary, linearThreshold);
	}

	// 収縮と膨張処理
	Mat img_processed;
	if (erodeDilateOrder == 0) {
		img_processed = dilate_erode_alternate(img_labeled, erodeDilateTimes);  // 交互に膨張-収縮
	}
	else {
		img_processed = erode_dilate_alternate(img_labeled, erodeDilateTimes);  // 交互に収縮-膨張
	}

	// 反転して皺纹变为黑色
	bitwise_not(img_processed, img_processed);

	// 処理結果の画像表示
	imshow("処理結果", img_processed);

	// 処理結果の保存
	size_t pos = imagePath.find_last_of("/\\");
	string directory = (pos == string::npos) ? "" : imagePath.substr(0, pos + 1);
	string orderType = (erodeDilateOrder == 0) ? "dilate_erode_" : "erode_dilate_";
	string savePath = directory + "processed_" + orderType + imagePath.substr(pos + 1);
	imwrite(savePath, img_processed);
	cout << "処理結果を保存しました: " << savePath << endl;

	waitKey(0);
	destroyAllWindows();
}
int main() {
	string imagePath;
	cout << "画像名を入力してください: ";
	cin >> imagePath;

	// フィルタの種類とサイズの手動入力
	int filterType;
	cout << "フィルタの種類を選んでください (0: メディアンフィルタ, 1: 平均フィルタ, 2: ガウシアンフィルタ): ";
	cin >> filterType;

	int filterSize;
	cout << "フィルタのサイズを入力してください: ";
	cin >> filterSize;

	// 膨張と収縮の順序の手動入力
	int erodeDilateOrder;
	cout << "膨張と収縮の順序を選んでください (0: 膨張-収縮, 1: 収縮-膨張): ";
	cin >> erodeDilateOrder;

	// 膨張と収縮の回数の手動入力
	int erodeDilateTimes;
	cout << "膨張と収縮の回数を入力してください: ";
	cin >> erodeDilateTimes;

	// 二値化の閾値の手動入力
	int binaryThreshold;
	cout << "二値化の閾値を入力してください: ";
	cin >> binaryThreshold;

	// 線状度の閾値の手動入力
	double linearThreshold;
	cout << "線状度の閾値を入力してください: ";
	cin >> linearThreshold;

	// 近傍の種類の手動入力
	int neighborType;
	cout << "近傍の種類を選んでください (0: 4近傍, 1: 8近傍): ";
	cin >> neighborType;

	// 画像処理の実行
	processImage(imagePath, filterType, filterSize, binaryThreshold, erodeDilateOrder, erodeDilateTimes, linearThreshold, neighborType);

	return 0;
}
*/