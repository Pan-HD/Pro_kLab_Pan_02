#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<conio.h> //getch()
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/opencv.hpp"
#define sedai 100//世代数
#define kotai 100//個体数
#define	length 26//遺伝子長(=ビット数) 
#define cross 0.8//交叉率(70～90%)
#define mut 0.1//突然変異率(0.1～5%)
#define RAND_MAX 32767//乱数の最大値
#define maxnoise 999999//ノイズ総数の限度
#define hs 200;//補正の基準値
using namespace cv;
using namespace std;
using namespace std;
short **gazo;//画像処理に用いる画像の配列
short **gazoold;//入力画像を保管する配列
short **gazo2;//画像処理に用いる画像2の配列
short **gazoold2;//入力画像2を保管する配列
short **gazo3;
short **sabun_g;//差分処理後の画像の画素値
int x, y;//入力画像1のX,Y方向の大きさ
int x2, y2;//入力画像2のX,Y方向の大きさ
int **label;//画素ごとのラベル
int label_num[maxnoise];//ラベルごとの点の数
int label_sum[maxnoise];//ラベルごとの総画素値
int lx1[maxnoise];//ラベルのX方向の大きさを求めるための変数
int lx2[maxnoise];//ラベルのX方向の大きさを求めるための変数
int ly1[maxnoise];//ラベルのY方向の大きさを求めるための変数
int ly2[maxnoise];//ラベルのY方向の大きさを求めるための変数
int label_area[maxnoise];//ラベルごとの正方形の面積
int table[maxnoise][2];//ルックアップテーブル
int tmp = 0;//カウンターの初期化
float sum1 = 0.0;//合計値
int ekotai = 0;
int esedai = 0;
float evalu = 0.0;
FILE *fp, *fp2, *fp3, *fp4, *fp5, *fp6;//ファイル定義
int fsize = 0;//メディアンフィルタ
int binary = 0;//2値化しきい値
int ccc = 0;//最低画素数
double linear = 0.0;//線状度
int eee = 0;//線状度*キズ濃度3
int abusolute_flag = 0;//線状度*キズ総画素数
int erodedilate_sequence = 0;//膨張や収縮の順番　
int filterswitch_flag;//フォルダを切り替えるフラグ
int erodedilate_times;//膨張収縮処理の繰り返す回数
int pixellabelingmethod = 0;//8か4か

Mat img_label, img_bitwise, img_output;//ラベルは連通区域を検出した画像　img_bitwiseは白黒反転した画像
Mat img1, img2;//二つの入力画像 img1は元画像の切り抜き，img2は教師画像に切り抜き
Mat img1_before, img2_before;//前回実験の切り抜き
Mat img1ths;
Mat img_origin, img_origincopy, img_target, img_targetcopy;
Point sp(-1, -1);
Point ep(-1, -1);


typedef struct gene {
public:
	int ge[length];//染色体
	float tekioudo;//適応度
	float valu;//評価
}gene;

gene h[kotai][8];//GENEという構造体構築方法でhという構造体を定義した

void make(gene *g)//初期個体群の生成
{
	int i = 0;
	int j = 0;

	for (j = 0; j < kotai; j++) {
		//fprintf(fp,"個体:%2d番目",j+1);//個体番号の表示
		for (i = 0; i < length; i++) {
			if (rand() > (RAND_MAX + 1) / 2) g[j].ge[i] = 1;
			else g[j].ge[i] = 0;

			fprintf(fp, "%d", g[j].ge[i]); //個体の表示
			printf("%d", g[j].ge[i]);
		}
		puts("\n");
		fprintf(fp, "\n");
	}
}

void phenotype(gene *g)//表現系計算(2進数を10進数に)
{
	int i = 0, j = 0, k = 0;

	for (j = 0; j < kotai; j++) {
		for (i = 0; i < 6; i++) {
			h[j][i].tekioudo = 0.0;//初期化
		}
	}

	for (j = 0; j < kotai; j++) {
		i = 6;
		for (k = 0; k < 6; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][0].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i); //メディアンフィルタ
			}
		}

		i = 8;
		for (k = 6; k < 14; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][1].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//2値化しきい値
			}
		}


		i = 5;
		for (k = 14; k < 19; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][2].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//線状度閾値
			}
		}

		i = 1;
		for (k = 19; k < 20; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][3].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//使用するフィルタの種類
			}
		}

		i = 3;
		for (k = 20; k < 23; k++) {
			i--;


			if (g[j].ge[k] == 1) {
				h[j][4].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//ラベリング閾値
			}
		}

		i = 1;
		for (k = 23; k < 24; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][5].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//膨張や収縮の順番

			}
		}

		i = 1;
		for (k = 24; k < 25; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][6].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//絶対値の選択

			}
		}

		i = 1;
		for (k = 25; k < 26; k++) {
			i--;
			if (g[j].ge[24] == 1) {
				h[j][7].tekioudo = 1;//近傍8画素もしくは4画素の選択
			}
			else
			{
				h[j][7].tekioudo = 0;//近傍8画素もしくは4画素の選択
			}
		}

	}
}

void fitness(gene *g, gene *elite, int se)//適応度の計算(エリート保存)
{
	int i = 0, j = 0;
	double ave = 0.0;
	double deviation = 0.0;
	double variance = 0.0;

	sum1 = 0;//初期化

	//エリート保存
	for (i = 0; i < kotai; i++) {
		sum1 += h[i][0].valu;
		if (h[i][0].valu > elite[1].valu) {
			elite[1].valu = h[i][0].valu;//エリート入れ換え
			for (j = 0; j < length; j++) {
				elite[1].ge[j] = g[i].ge[j];
			}
		}
	}

	float min_value = 1.1;
	elite[2].valu = 1.1;
	for (i = 0; i < kotai; i++) {
		if (h[i][0].valu < min_value) {
			min_value = h[i][0].valu;
		}
	}

	elite[3].valu = 0.0;//初期化
	for (i = 0; i < kotai; i++) {
		elite[3].valu += h[i][0].valu;
	}

	ave = (double)(elite[3].valu) / (double)kotai;

	for (i = 0; i < kotai; i++)
	{
		double diff = h[i][0].valu - ave;
		variance += diff * diff;
	}

	deviation = sqrt(variance / kotai);


	//fprintf(fp4, "%d	%.2f	%.2f	平均：%.2f\n", se, elite[1].valu, elite[2].valu, ave);
	fprintf(fp4, "%d	最大：%.2f	最小：%.2f　平均値：%.2f  平均偏差値：%.2f\n", se, elite[1].valu, min_value, ave, deviation);
	printf("%d	%.2f	%.2f	平均：%.2f  平均偏差値：%.2f\n", se, elite[1].valu, min_value, ave, deviation);

}

void elite_back(gene *g, gene *elite) {//エリート個体と適応度最小個体を交換

	int i = 0, j = 0;
	float ave = 0.0;
	float min1 = 1.0;

	tmp = 0;//カウンターの初期化

	for (i = 0; i < kotai; i++) {//最小値探索
		if (h[i][0].valu < min1) {
			min1 = h[i][0].valu;
			tmp = i;
		}
	}

	for (j = 0; j < length; j++) {
		g[tmp].ge[j] = elite[1].ge[j];//最小値とエリートを交換
	}

	h[tmp][0].valu = elite[1].valu;//エリートの評価値と交換
	ave = sum1 / kotai;//合計値の計算
}

int roulette()//ルーレット選択
{
	int i = 0, r = 0;
	int num = 0;
	float sum = 0.0;
	float *p;

	p = (float*)malloc(sizeof(int)*kotai);

	sum = 0;
	for (i = 0; i < kotai; i++) {
		sum += h[i][0].valu;//すべての合計
	}
	for (i = 0; i < kotai; i++) {
		p[i] = h[i][0].valu / sum;//適応度(％)
	}

	sum = 0;
	r = rand();
	for (i = 0; i < kotai; i++) {
		sum += RAND_MAX * p[i];//1
		if (r <= sum) {
			num = i;
			break;
		}
	}
	if (num < 0)	num = roulette();//エラーのための処理
	free(p);
	return(num);
}

void crossover(gene *g) {//一点交叉
	gene g2[1000] = { 0 };
	int num = 0;
	int n1 = 0;
	int n2 = 0;
	int p = 0;
	int i, j;

	for (num = 0; num < kotai; num += 2) {
		n1 = rand() % 10;
		n2 = rand() % 10;
		if (rand() <= RAND_MAX * cross) {//交叉確率を満たす場合
			n1 = roulette();
			n2 = roulette();
			//乱数の範囲指定公式：(int)( rand() * (最大値 - 最小値 + 1.0) / (1.0 + RAND_MAX) )
			p = (int)(rand()*((length - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);

			//子A
			for (i = 0; i < p; i++) {
				g2[num].ge[i] = g[n1].ge[i];
			}
			for (i = p; i < length; i++) {
				g2[num].ge[i] = g[n2].ge[i];
			}

			//子B
			for (i = 0; i < p; i++) {
				g2[num + 1].ge[i] = g[n2].ge[i];
			}
			for (i = p; i < length; i++) {
				g2[num + 1].ge[i] = g[n1].ge[i];
			}
		}
		else {
			for (i = 0; i < length; i++) {
				n1 = roulette();
				n2 = roulette();
				g2[num].ge[i] = g[n1].ge[i];
				g2[num + 1].ge[i] = g[n2].ge[i];
			}
		}
	}

	for (j = 0; j < kotai; j++) {
		for (i = 0; i < length; i++) {
			g[j].ge[i] = g2[j].ge[i];//g[]を更新
		}
	}
}

void mutation(gene *g)//突然変異
{
	int num = 0;
	int r = 0;
	int i = 0;
	int p = 0;
	for (num = 0; num < kotai; num++) {
		if (rand() <= RAND_MAX * mut) {//突然変異確率を満たす場合，1つの遺伝子を選択
			p = (int)(rand()*((length - 1) + 1.0) / (1.0 + RAND_MAX));
			for (i = 0; i < length; i++) {//1と0を逆転
				if (i == p) {
					if (g[num].ge[i] == 0) g[num].ge[i] = 1;
					else				g[num].ge[i] = 0;
				}
			}
			p = 0;
		}
	}
}

void import_image() {

	img_bitwise.copyTo(img_output);
}

void import_para(int ko) {//パラメータの出力
	fsize = 0;
	binary = 0;
	ccc = 0;
	linear = 0.0;
	filterswitch_flag = 0;
	abusolute_flag = 0;

	fsize = (int)(3 + 2 * h[ko][0].tekioudo);
	binary = (int)(1 * h[ko][1].tekioudo);
	//ccc = (int)(4 * h[ko][2].tekioudo);
	linear = (double)(1.0 + 0.5*h[ko][2].tekioudo);
	filterswitch_flag = (int)(h[ko][3].tekioudo);
	erodedilate_times = (int)(h[ko][4].tekioudo);
	erodedilate_sequence = (int)(h[ko][5].tekioudo);
	abusolute_flag = (int)(h[ko][6].tekioudo);
	pixellabelingmethod = (int)(h[ko][7].tekioudo);

	fprintf(fp2, "fsize：%5d	binary：%5d	filterswitch_flag：%5d	linear：%2.2f	erodedilate_times:%7d　 erodedilate_sequence:%d pixellabelingmethod:%7d\n", fsize, binary, filterswitch_flag, linear, erodedilate_times, erodedilate_sequence, pixellabelingmethod);
}

void noiz_kessonnew(int ko, int se) {
	int i = 0, j = 0;
	int m = 0, n = 0, tm = 0, tn = 0;
	float ks = 0.0, nz = 0.0;
	float v = 0.0;
	int flg = 0;
	h[ko][0].valu = 0.0;//初期化
	for (j = 0; j < y2; j++) {
		for (i = 0; i < x2; i++) {
			if (img2.at<unsigned char>(j, i) == 0)
			{
				tm++;
			}
			else tn++;
		}
	}
	//ノイズ率，欠損率
	for (j = 0; j < y2; j++) {
		for (i = 0; i < x2; i++) {
			if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 255) {//ノイズ率
				m++;
			}
			if (img2.at<unsigned char>(Point(i, j)) == 255 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) {//欠損率
				n++;
			}
		}
	}
	if (m == 0)
	{
		m = 1;
	}
	if (n == 0)
	{
		n = 1;
	}

	ks = (float)((float)m / (float)tm);
	//printf("ks:%f\n", ks);
	nz = (float)((float)n / (float)tn);
	h[ko][0].valu = 1.0 - sqrt((0.2*ks*ks) + (0.8*nz*nz));
	v = 1.0 - sqrt((0.2*ks*ks) + (0.8*nz*nz));

	if (ko == 0) {
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
		fprintf(fp5, "世代：%d 個体：%d\n", se, ko);
	}
	if ((ko != 0) /*&& (v > evalu)*/) {
		flg = 1;
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
		//printf("v:%f\n", v);
		fprintf(fp5, "世代：%d 個体：%d\n", se, ko);
	}
	fprintf(fp3, "nz * 100:%.4f	ks * 100：%.2f\n", nz * 100, ks * 100);
}

void noiz_kessonFvalue(int ko, int se) {
	int i = 0, j = 0;
	int TP = 0, FP = 0, TN = 0, FN = 0;
	int m = 0, n = 0, tm = 0, tn = 0;
	float ks = 0.0, nz = 0.0;
	float precision = 0.0, recall = 0.0;//precision適合率，recall再現率

	float v = 0.0;
	int flg = 0;
	h[ko][0].valu = 0.0;//初期化
	//ノイズ率，欠損率
	for (j = 0; j < y2; j++) {
		for (i = 0; i < x2; i++) {
			if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) {//TruePositive
				TP++;
			}
			else if (img2.at<unsigned char>(Point(i, j)) == 255 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) {//FalsePositive
				FP++;
			}
			else if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 255) {//FalseNegative
				FN++;
			}
		}
	}
	if (TP == 0)
	{
		TP = 1;
	}
	if (FP == 0)
	{
		FP = 1;
	}
	if (FN == 0)
	{
		FN = 1;
	}

	precision = ((float)TP / ((float)TP + (float)FP));
	recall = (float)((float)TP / ((float)TP + (float)FN));
	v = (2 * precision*recall) / (precision + recall);
	h[ko][0].valu = (2 * precision*recall) / (precision + recall);
	v = (2 * precision*recall) / (precision + recall);

	if (ko == 0) {
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
	}

	if ((ko != 0) /*&& (v > evalu)*/) {
		flg = 1;
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
		//printf("v:%f\n", v);
		fprintf(fp5, "世代：%d 個体：%d\n", se, ko);
		fprintf(fp5, "v:%f\n	\n", v);
	}
	fprintf(fp3, "precision:%.4f	recall：%.6f\n  Fvalue：%.2f\n", precision, recall, v);
	if (v < 0.51)
	{
		fprintf(fp6, "NO");
	}
	else
	{
		fprintf(fp6, "YES");
	}
}

int tyuuou()//画像の中央値を求める
{
	int i = 0, j = 0, n = 0;
	int sum = 0;
	int c = 0;
	for (j = 0; j <= y - 1; j++) {
		for (i = 0; i <= x - 1; i++) {
			sum += gazoold[i][j];
			c++;
		}
	}
	return (sum / (float)c);
}

void lookup(int look, int min) {
	int tmp;
	if (table[look][1] > min) {
		tmp = table[look][1];
		table[look][1] = min;
		lookup(tmp, min);
	}
}
/*
Mat labeling_new4(Mat img_sabun, double linear) {//近傍4画素で画素の塊をラベリング方法
	Mat img_con;
	Mat stats, centroids;//連通区域の属性
	int i, j, label_num;//連通区域の数

	int label_x, label_y;
	int label_longer;
	double label_cal;
	int label_areaall;//ラベルを付けた区域の画素数

	label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 4, 4);
	vector<Vec3b>colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);

	for (i = 1; i < label_num; i++)
	{
		colors[i] = Vec3b(255, 255, 255);
		label_areaall = stats.at<int>(i, CC_STAT_AREA);

		label_x = stats.at<int>(i, CC_STAT_WIDTH);
		label_y = stats.at<int>(i, CC_STAT_HEIGHT);

		if (label_x > label_y)label_longer = label_x;
		else label_longer = label_y;

		label_cal = label_longer * label_longer;//より長い辺の二乗を計算する

		if (label_cal / label_areaall < linear)
		{
			colors[i] = Vec3b(0, 0, 0);
		}
	}

	Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
	for (j = 0; j < img_con.rows; j++) {
		for (i = 0; i < img_con.cols; i++)
		{
			int label = img_con.at<int>(j, i);
			CV_Assert(0 <= label && label <= label_num);
			img_color.at<Vec3b>(j, i) = colors[label];
		}
	}

	return img_color;
}
Mat labeling_new8(Mat img_sabun, double linear) {//近傍8画素で画素の塊をラベリング方法
	Mat img_con;
	Mat stats, centroids;//連通区域の属性
	int i, j, label_num;//連通区域の数

	int label_x, label_y;
	int label_longer;
	double label_cal;
	int label_areaall;//ラベルを付けた区域の画素数
	label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 8, 4);
	vector<Vec3b>colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);

	for (i = 1; i < label_num; i++)
	{
		colors[i] = Vec3b(255, 255, 255);


		label_areaall = stats.at<int>(i, CC_STAT_AREA);


		label_x = stats.at<int>(i, CC_STAT_WIDTH);
		label_y = stats.at<int>(i, CC_STAT_HEIGHT);

		if (label_x > label_y)label_longer = label_x;
		else label_longer = label_y;
		label_cal = label_longer * label_longer;//より長い辺の二乗を計算する
		if (label_cal / label_areaall < linear)
		{
			colors[i] = Vec3b(0, 0, 0);
		}
	}
	Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
	for (j = 0; j < img_con.rows; j++) {
		for (i = 0; i < img_con.cols; i++)
		{
			int label = img_con.at<int>(j, i);
			CV_Assert(0 <= label && label <= label_num);
			img_color.at<Vec3b>(j, i) = colors[label];
		}
	}
	return img_color;
}

static void on_draw(int event, int x, int y, int flags, void* userdata) {
	Mat image = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN) {     //マウスを左クリックする時の座標を記録する
		sp.x = x;
		sp.y = y;
		std::cout << "start point:" << sp << std::endl;			//そのときの座標を出力

	}
	else if (event == EVENT_LBUTTONUP) {
		ep.x = x;							//数値を記録する
		ep.y = y;
		int dx = abs(ep.x - sp.x);				//widthを計算
		int dy = abs(ep.y - sp.y);				// heightを計算
		Rect box;
		if (dx > 0 && dy > 0) {					//選定してるかどうかを判定
			if ((ep.x - sp.x) > 0 && (ep.y - sp.y) > 0) {			//左上から初めて　右下まで選定
				box.x = sp.x;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);   //処理区域を選定
			}
			else if ((ep.x - sp.x) > 0 && (ep.y - sp.y) < 0) {		//左下から初めて　右上まで選定
				box.x = sp.x;
				box.y = sp.y - dy;
				box.width = dx;
				box.height = dy;
				cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			}
			else if ((ep.x - sp.x) < 0 && (ep.y - sp.y) > 0) {			//右上から初めて　左下まで
				box.x = sp.x - dx;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			}
			else {												//右下から初めて　左上まで選定以右下角为起点，左上角为结点框选
				{
					box.x = sp.x - dx;
					box.y = sp.y - dy;
					box.width = dx;
					box.height = dy;
					cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				}
			}
			img_origincopy.copyTo(image);
			img_targetcopy.copyTo(img_target);
			imshow("鼠标绘制", image);
			imshow("ROI区域_origin", image(box));
			imshow("ROI区域_target", img_target(box));
			imwrite("origin_out.jpg", image(box));
			imwrite("target_out.jpg", img_target(box));
			img1 = image(box);
			img2 = img_target(box);
			//图片保存功能，根据需要自行修改
			//ready for next drawing
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {			//マウスの移動イベント　選択区域の線は動的に見えるように
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = abs(ep.x - sp.x);
			int dy = abs(ep.y - sp.y);
			Rect box;
			if ((ep.x - sp.x) > 0 && (ep.y - sp.y) > 0) {
				box.x = sp.x;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);		//把img_origincopy中的图片复制给image,达到一种清屏的效果	 不明白的可以自行注释这两条代码运行看看就懂了	
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);			//清屏后立刻重新款选
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);			//清屏后立刻重新款选

			}
			else if ((ep.x - sp.x) > 0 && (ep.y - sp.y) < 0) {
				box.x = sp.x;
				box.y = sp.y - dy;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);

			}
			else if ((ep.x - sp.x) < 0 && (ep.y - sp.y) > 0) {
				box.x = sp.x - dx;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);
			}
			else {
				box.x = sp.x - dx;
				box.y = sp.y - dy;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);

			}
			imshow("鼠标绘制", image);
			imshow("鼠标绘制", img_target);
		}
	}
}

void mouse_drawing_demo(Mat& image, Mat& image_target) {
	namedWindow("鼠标绘制", WINDOW_FREERATIO);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	imshow("鼠标绘制", image);
	img_origincopy = image.clone();			//复制image图片
	img_targetcopy = image_target.clone();	//复制image图片
}

Mat sabun(Mat input1, Mat input2) {
	int i, j;
	Mat output;
	output = cv::Mat::zeros(cv::Size(input2.cols, input2.rows), CV_8UC3);//8UC3は3チャンネルに変えるタイプだ
	cvtColor(output, output, COLOR_RGB2GRAY);//グレースケール

	for (j = 0; j < input1.rows; j++)
	{
		for (i = 0; i < input1.cols; i++) {
			output.at<unsigned char>(j, i) = input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i);
			if (input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i) < 0)
			{
				output.at<unsigned char>(j, i) = 0;
			}
		}
	}
	return output;
}

Mat dilate_erode(Mat src1) {//オープニングクロージング処理
	Mat dst;
	dst.create(src1.size(), src1.type());
	//クロージング処理後，オープニング処理．更に膨張処理
	dilate(src1, dst, Mat());//膨張処理
	erode(dst, dst, Mat());//収縮処理

	return dst;
}

Mat erode_dilate(Mat src1) {//オープニングクロージング処理
	Mat dst;
	dst.create(src1.size(), src1.type());
	//クロージング処理後，オープニング処理．更に膨張処理
	erode(dst, dst, Mat());//収縮処理
	dilate(src1, dst, Mat());//膨張処理

	return dst;
}

int main(int argc, char *argv[])
{
	Mat img1_median, img1_sabun;//二つの入力画像
	unsigned char r1, g1, b1, r2, g2, b2;
	short count = 0;
	short tyuu;
	short v2;
	int i = 0, j = 0, k = 0;//for文用変数
	int picture_loading_mode = 0;//新たな切り抜きをつくるのが前の画像のまま使うのが　0なら新しい切り抜きを作成　1なら
	int num;
	char filename1[256];//ファイル名入力用配列
	char filename2[256];//ファイル名入力用配列
	char filename3[256];//ファイル名入力用配列
	char decision_data[256];//ファイル名入力用配列 決定木用

	char sw = 0;
	clock_t start, end;
	start = clock();

	if ((fp = fopen("個体の遺伝子型.txt", "w")) == NULL) {
		printf("error:ファイルをオープンできません。\n");
	}
	if ((fp2 = fopen("parameter.txt", "w")) == NULL) {
		printf("error:ファイルをオープンできません。\n");
	}
	if ((fp3 = fopen("noiz_kesson.txt", "w")) == NULL) {
		printf("error:ファイルをオープンできません。\n");
	}
	if ((fp4 = fopen("max_ave_min_deviation.txt", "w")) == NULL) {
		printf("error:ファイルをオープンできません。\n");
	}
	if ((fp5 = fopen("max_para.txt", "w")) == NULL) {
		printf("error:ファイルをオープンできません。\n");
	}
	if ((fp6 = fopen("decision_tree_dataset.txt", "w")) == NULL) {
		printf("error:ファイルをオープンできません。\n");
	}

	srand((unsigned)time(NULL));//時刻による乱数の初期化
	gene g[kotai];
	gene elite[10];//エリート保存数
	elite[1].valu = 0.0;//エリート個体の初期化
	img_origin = imread("シワ1.jpg");
	img_target = imread("mask1.jpg");
	puts("前回実験の切り抜きを使いますが？0/1\n");
	puts("0いいえ　新たな区域で実験する\n");
	puts("1はい　前の画像を使いたい\n");
	scanf("%d", &picture_loading_mode);
	printf("picture_loading_mode:%d\n", picture_loading_mode);

	if (picture_loading_mode == 0)
	{
		if (img_origin.empty())
		{
			printf("画像を読み込むことが失敗した");
			return -1;
		}
		mouse_drawing_demo(img_origin, img_target);
		waitKey(0);
		destroyAllWindows();
		cvtColor(img1, img1ths, COLOR_RGB2GRAY);
		img1_before = img1.clone();
		img2_before = img2.clone();
		imwrite("image1.png", img1_before);
		imwrite("kyoushi1.png", img2_before);
	}
	else
	{
		img1 = imread("image1.png");
		img2 = imread("kyoushi1.png");
		cvtColor(img1, img1ths, COLOR_RGB2GRAY);
		if (img1.empty())
		{
			printf("画像を読み込みできない");
			return -1;
		}
	}
	x = img1.cols;//x方向の画像サイズ
	y = img1.rows;//y方向の画像サイズ
	x2 = img2.cols;//x方向の画像サイズ
	y2 = img2.rows;//y方向の画像サイズ
	make(g);//初期個体群の生成

	for (num = 1; num <= sedai; num++) {//世代
		printf("======= 第%d世代 =======\n", num);
		phenotype(g);//表現系計算(2進数を10進数に)
		fprintf(fp2, "======= 第%d世代 =======\n", num);

		for (k = 0; k < kotai; k++) {
			import_para(k);
			fprintf(fp6, "%d,%d,%2.2f", fsize, binary, linear);
			sprintf_s(decision_data, "%d", fsize);
			sprintf_s(decision_data, ",%d", binary);

			if (filterswitch_flag == 0)
			{
				medianBlur(img1ths, img1_median, fsize);
				fprintf(fp6, ",median-filter");

			}
			else
			{
				blur(img1ths, img1_median, Size(fsize, fsize));
				fprintf(fp6, ",average-filter");
			}

			if (abusolute_flag == 0)
			{
				fprintf(fp6, ",abusolute");
			}
			else
			{
				fprintf(fp6, ",non-abusolute");
			}

			img1_sabun = sabun(img1ths, img1_median);

			waitKey(0);
			threshold(img1_sabun, img1_sabun, binary, 255, THRESH_BINARY);//二値化処理
			if (pixellabelingmethod == 0)
			{
				//printf("pixelmehod's value:%d\n", pixellabelingmethod);
				img_label = labeling_new4(img1_sabun, linear);//ラベリング+ノイズ除去
				fprintf(fp6, ",4pixel");
			}
			else
			{
				img_label = labeling_new8(img1_sabun, linear);//ラベリング+ノイズ除去
				fprintf(fp6, ",8pixel");

			}
			pixellabelingmethod = 0;//０に戻す
			bitwise_not(img_label, img_bitwise);//白黒反転処理
			if (erodedilate_sequence == 0)
			{
				fprintf(fp6, ",dilatefirst");
				if (erodedilate_times != 0) {
					for (i = 0; i < erodedilate_times; i++)
					{
						img_bitwise = dilate_erode(img_bitwise);
					}
				}
				fprintf(fp6, ",%d,", erodedilate_times);
			}
			else if (erodedilate_sequence == 1)
			{
				fprintf(fp6, ",erodefirst");
				if (erodedilate_times != 0) {
					for (i = 0; i < erodedilate_times; i++)
					{
						img_bitwise = erode_dilate(img_bitwise);
					}
				}
				fprintf(fp6, ",%d,", erodedilate_times);
			}
			noiz_kessonFvalue(k, num);//評価
			fprintf(fp6, "\n");
		}
		//output();//画像出力
		sprintf(filename3, "sedai%dエリート.png", esedai);//出力画像名入力
		imwrite(filename3, img_output);
		esedai = 0;
		ekotai = 0;
		evalu = 0.0;
		fitness(g, elite, num);//適応度の計算(エリート保存)
		crossover(g);//一点交叉
		mutation(g);//突然変異
		elite_back(g, elite);//エリート個体と適応度最小個体を交換
	}

	printf("========= 終了 =========\n");
	//ファイル終了
	fclose(fp);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
	fclose(fp5);
	fclose(fp6);
	//処理にかかった時間
	end = clock();
	int minute = 0;//かかった分
	double second = 0;//かかった秒
	minute = ((double)(end - start) / CLOCKS_PER_SEC) / 60;
	second = ((double)(end - start) / CLOCKS_PER_SEC) - 60 * minute;
	printf("%d分%.3f秒\n", minute, second);
	//メモリ開放
	free(gazo);
	free(gazoold);
	free(gazo2);
	free(gazoold2);
	free(sabun_g);
	free(label);
	_getch();
	return(0);
}*/