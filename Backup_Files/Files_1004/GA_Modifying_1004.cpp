#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define chLen 20 // 個体の染色体の長さ
#define num_ind 100 // 個体数

/*
	個体のデータ構造
*/
typedef struct gene {
	int ch[chLen];
	float adapt;
	float value;
}gene;

/*
    決定変数：
	fsize: ブラー用 6bit
	binary：二値化の閾値 8bit
	(linear: 必要ではないと思う) 5bit
	filterswitch_flag: 1bit
	erodedilate_times: 3bit
	erodedilate_sequence: 1bit -> test for its performance
	(abusolute_flag: 必要ではないと思う) 1bit
	pixellabelingmethod: 1bit
*/

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

/*
	役割：生成した個体群の初期化 -> 各個体に対してその遺伝子・・・
		  g: 個体群を保存する配列, gene g[kotai]
*/
void make(gene* g) // 
{
	int i = 0;
	int j = 0;

	for (j = 0; j < num_ind; j++) {
		// fprintf(fp,"個体:%2d番目",j+1); // 個体番号の表示
		for (i = 0; i < chLen; i++) { // chLen: 遺伝子の長さ
			if (rand() > (RAND_MAX + 1) / 2) g[j].ge[i] = 1;
			else g[j].ge[i] = 0;
			// 今回の初期化情報をfpとstdoutに書き込む。
			// fprintf(fp, "%d", g[j].ge[i]);  // 個体の表示
			// printf("%d", g[j].ge[i]);
		}
		puts("\n");
		// fprintf(fp, "\n");
	}
}

int main(void) {
	Mat img_ori = imread("./imgs_Pro_GA/oriImg.png");
	Mat img_mask = imread("./imgs_Pro_GA/maskImg.png");

	//vector<Mat> images = { img_ori, img_mask };
	//Mat res;
	//hconcat(images, res);
	//imgShow("ori", res);

	gene g[num_ind]; // 個体群


	return 0;
}