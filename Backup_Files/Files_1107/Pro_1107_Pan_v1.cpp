#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

int main() {
	// Mat img = imread("./imgs_1107_v1/oriImg_01.png");
	Mat img_g = imread("./imgs_1107_v1/oriImg_05.png", 0);
	Mat img_bi;

	threshold(img_g, img_bi, 177, 255, THRESH_BINARY);

	imgShow("bi", img_bi);
	imwrite("./imgs_1107_v1/tarImg_05.png", img_bi);

	return 0;
}