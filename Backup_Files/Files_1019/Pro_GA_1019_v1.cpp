/*
    To do list:
        01. 錠剤のマスク画像の作成
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define RAND_MAX 32767 // 乱数の最大値
#define chLen 21 // 個体の染色体の長さ
#define num_ind 100 // the nums of individuals in the group
#define num_gen 100 // the nums of generation of the GA algorithm
#define cross 0.8 // 交叉率
#define mut 0.05 // 突然変異率

typedef struct {
    int ch[chLen]; // defining chromosomes by ch-array
    float fitness; // the fitness of 7 decision variables
    float f_value; // the harmonic mean of precision and recall of the individual
}gene;

//typedef struct {
//    double precision;
//    double recall;
//    double f_value;
//}eliteValue;

// for mask-generate and target-generate
int circlePoint_x = 0;
int circlePoint_y = 0;
int circleRadius = 0;
// for storing the fitness value of 7 decision variables
gene h[num_ind][7];
// the input 3 images in the beginning
Mat oriImg;
Mat maskImg;
Mat tarImg;
// the declaration of 7 decision variables
int fsize = 0;
int binary = 0;
int abusolute_flag = 0;
int erodedilate_sequence = 0;
int filterswitch_flag;
int erodedilate_times;
int pixellabelingmethod = 0;
// 
FILE* fp;

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
}

/*
    Func: Independent function, for generating mask picture.
          Only for pictures of pills(錠剤)
*/
void maskGenerate() {
    Mat img_ori = imread("./imgs_Pro_GA/oriImg.png");
    Mat img_ori_g = imread("./imgs_Pro_GA/oriImg.png", 0);
    GaussianBlur(img_ori_g, img_ori_g, cv::Size(9, 9), 2, 2);
    vector<Vec3f> circles;

    HoughCircles(img_ori_g, circles, HOUGH_GRADIENT, 1,
        img_ori_g.rows / 16,
        100, 30, 0, 0
    );

    Mat maskImg = cv::Mat::zeros(img_ori.rows, img_ori.cols, CV_8UC1);
    circlePoint_x = circles[0][0];
    circlePoint_y = circles[0][1];
    circleRadius = circles[0][2];

    for (int y = 0; y < img_ori.rows; y++) {
        for (int x = 0; x < img_ori.cols; x++) {
            int dx = x - circlePoint_x;
            int dy = y - circlePoint_y;
            if (dx * dx + dy * dy <= circleRadius * circleRadius) {
                maskImg.at<uchar>(y, x) = 255;
            }
        }
    }
    imwrite("./imgs_Pro_GA/maskImg.png", maskImg);
}

/*
    Func: Independent function, for generating target picture.
          Only for pictures of pills(錠剤)
*/
void targetGenerate() {
    Mat img_ori = imread("./imgs_Pro_GA/oriImg.png");
    Mat img_ori_g = imread("./imgs_Pro_GA/oriImg.png", 0);
    Mat img_mask2tar = imread("./imgs_Pro_GA/maskImg.png");

    GaussianBlur(img_ori_g, img_ori_g, cv::Size(9, 9), 2, 2);
    Mat img_ori_bi;
    threshold(img_ori_g, img_ori_bi, 205, 255, THRESH_BINARY);

    vector<Vec3f> circles;
    HoughCircles(img_ori_bi, circles, HOUGH_GRADIENT, 1,
        1,
        100, 20, 0, 0
    );

    //cout << "inner radius: " << circles[0][2] << endl;

    int squ_out = circleRadius * circleRadius;
    int squ_in = circles[0][2] * circles[0][2] - 300;

    for (int y = 0; y < img_ori.rows; y++) {
        for (int x = 0; x < img_ori.cols; x++) {
            int dx = x - circlePoint_x;
            int dy = y - circlePoint_y;
            if (dx * dx + dy * dy >= squ_in && dx * dx + dy * dy <= squ_out) {
                img_ori_bi.at<uchar>(y, x) = 255;
            }
        }
    }
    imwrite("./imgs_Pro_GA/tarImg.png", img_ori_bi);
}

/*
    Func: Initializing the elements of the chromosomes of every individual
*/
void make(gene* g)
{
    for (int j = 0; j < num_ind; j++) {
        for (int i = 0; i < chLen; i++) {
            if (rand() > (RAND_MAX + 1) / 2) g[j].ch[i] = 1;
            else g[j].ch[i] = 0;
        }
    }
}

void phenotype(gene* g)
{
    int i = 0, j = 0, k = 0;
    // initializing the fitness in h-array by assigning 0
    for (j = 0; j < num_ind; j++) {
        for (i = 0; i < 7; i++) {
            h[j][i].fitness = 0.0;
        }
    }

    for (j = 0; j < num_ind; j++) {
        // fsize - 6 bits
        i = 6;
        for (k = 0; k < 6; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][0].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // binary - 8 bits
        i = 8;
        for (k = 6; k < 14; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][1].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // filterswich_flag - 1 bit
        i = 1;
        for (k = 14; k < 15; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][2].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // erodedilate_times - 3 bits
        i = 3;
        for (k = 15; k < 18; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][3].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // erodedilate_sequence - 1 bit
        i = 1;
        for (k = 18; k < 19; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][4].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // abusolute_flag - 1 bit
        i = 1;
        for (k = 19; k < 20; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][5].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);

            }
        }
        // pixellabelingmethod - 1bit
        i = 1;
        for (k = 20; k < 21; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][6].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);

            }
        }
    }
}

void import_para(int ko) {
    fsize = 0;
    binary = 0;
    filterswitch_flag = 0;
    abusolute_flag = 0;
    fsize = (int)(3 + 2 * h[ko][0].fitness);
    binary = (int)(1 * h[ko][1].fitness);
    filterswitch_flag = (int)(h[ko][2].fitness);
    erodedilate_times = (int)(h[ko][3].fitness);
    erodedilate_sequence = (int)(h[ko][4].fitness);
    abusolute_flag = (int)(h[ko][5].fitness);
    pixellabelingmethod = (int)(h[ko][6].fitness);
}

Mat sabun(Mat input1, Mat input2) {
    int i, j;
    int res; // the diff value between input2 and input1
    Mat output;
    output = Mat::zeros(Size(input2.cols, input2.rows), CV_8UC3);
    cvtColor(output, output, COLOR_RGB2GRAY);
    for (j = 0; j < input1.rows; j++)
    {
        for (i = 0; i < input1.cols; i++) {
            res = input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i);
            if (abusolute_flag) {
                output.at<unsigned char>(j, i) = res >= 0 ? res : (-1) * res;
            }
            else {
                output.at<unsigned char>(j, i) = res >= 0 ? res : 0;
            }
        }
    }
    return output;
}

Mat labeling_new4(Mat img_sabun) {
    Mat img_con;
    Mat stats, centroids;
    int i, j, label_num;

    int label_x, label_y;
    int label_longer;
    double label_cal;
    int label_areaall;
    label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 4, 4);
    vector<Vec3b>colors(label_num + 1); // ラベルごとにどの色を使うかを格納するためのものです
    colors[0] = Vec3b(0, 0, 0); // 背景の色を黒に設定
    for (i = 1; i < label_num; i++) // ラベルごとの処理
    {
        colors[i] = Vec3b(255, 255, 255);

        //label_areaall = stats.at<int>(i, CC_STAT_AREA);
        //label_x = stats.at<int>(i, CC_STAT_WIDTH);
        //label_y = stats.at<int>(i, CC_STAT_HEIGHT);
    }
    // CV_8UC3：3チャンネル・・・
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

Mat labeling_new8(Mat img_sabun) { // 近傍8画素で画素の塊をラベリング方法
    Mat img_con;
    Mat stats, centroids; // 連通区域の属性
    int i, j, label_num; // 連通区域の数

    int label_x, label_y;
    int label_longer;
    double label_cal;
    int label_areaall; // ラベルを付けた区域の画素数
    label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 8, 4);
    vector<Vec3b>colors(label_num + 1);
    colors[0] = Vec3b(0, 0, 0);

    for (i = 1; i < label_num; i++)
    {
        colors[i] = Vec3b(255, 255, 255);
        //label_areaall = stats.at<int>(i, CC_STAT_AREA);
        //label_x = stats.at<int>(i, CC_STAT_WIDTH);
        //label_y = stats.at<int>(i, CC_STAT_HEIGHT);
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

Mat dilate_erode(Mat src1) { // オープニングクロージング処理
    Mat dst;
    dst.create(src1.size(), src1.type());
    // クロージング処理後，オープニング処理．更に膨張処理
    dilate(src1, dst, Mat()); // 膨張処理
    erode(dst, dst, Mat()); // 収縮処理

    return dst;
}

Mat erode_dilate(Mat src1) { // オープニングクロージング処理
    Mat dst;
    dst.create(src1.size(), src1.type());
    // クロージング処理後，オープニング処理．更に膨張処理
    erode(dst, dst, Mat()); // 収縮処理
    dilate(src1, dst, Mat()); // 膨張処理

    return dst;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculateMetrics(Mat image1, Mat image2, Mat mask, int numInd) {
    // tp: True Positive, 正しく検出のピクセル数
    // fp: False Positive, 誤検出のピクセル数
    // fn: False Negative, 未検出のピクセル数
    int tp = 0, fp = 0, fn = 0;
    int mask_pixel_count = 0;  // マスク中で値が255のピクセル数をカウントする変数

    for (int y = 0; y < image1.rows; y++) {
        for (int x = 0; x < image1.cols; x++) {
            // マスクが存在する場合、マスクの白い部分のみ計算
            if (mask.at<uchar>(y, x) != 255) {
                continue;
            }

            if (mask.at<uchar>(y, x) == 255) {
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
    h[numInd][0].f_value = f1_score;

    // 結果を表示
    //if (mask) {
    //    std::cout << "マスク部分の結果:" << std::endl;
    //    std::cout << "マスク中の255のピクセル数: " << mask_pixel_count << std::endl;
    //}
    //else {
    //    std::cout << "全体画像の結果:" << std::endl;
    //}

    //std::cout << std::fixed << std::setprecision(20); // 固定表示と小数点以下20桁に設定
    //std::cout << "適応率 (Precision): " << precision << std::endl;
    //std::cout << "再現率 (Recall): " << recall << std::endl;
    //std::cout << "F値 (F1 Score): " << f1_score << std::endl;
    //std::cout << "正解数 (True Positives): " << tp << std::endl;
    //std::cout << "欠損画素数 (False Negatives): " << fn << std::endl;
}

void fitness(gene* g, gene* elite, int se) // 適応度の計算(エリート保存)
{
    int i = 0, j = 0;
    double ave = 0.0;
    double deviation = 0.0;
    double variance = 0.0;

    int sum1 = 0; // 初期化

    // エリート保存
    for (i = 0; i < num_ind; i++) {
        sum1 += h[i][0].f_value;
        if (h[i][0].f_value > elite[1].f_value) {
            // エリート入れ換え
            elite[1].f_value = h[i][0].f_value;
            for (j = 0; j < chLen; j++) {
                elite[1].ch[j] = g[i].ch[j];
            }
        }
    }

    //float min_value = 1.1;
    //elite[2].f_value = 1.1;
    //for (i = 0; i < num_ind; i++) {
    //    if (h[i][0].f_value < min_value) {
    //        min_value = h[i][0].f_value;
    //    }
    //}

    //elite[3].f_value = 0.0; // 初期化
    //for (i = 0; i < num_ind; i++) {
    //    elite[3].f_value += h[i][0].f_value;
    //}

    //ave = (double)(elite[3].f_value) / (double)num_ind;

    //for (i = 0; i < num_ind; i++)
    //{
    //    double diff = h[i][0].f_value - ave;
    //    variance += diff * diff;
    //}

    //deviation = sqrt(variance / num_ind);


    // fprintf(fp4, "%d	%.2f	%.2f	平均：%.2f\n", se, elite[1].valu, elite[2].valu, ave);
    // fprintf(fp4, "%d	最大：%.2f	最小：%.2f　平均値：%.2f  平均偏差値：%.2f\n", se, elite[1].valu, min_value, ave, deviation);
    // printf("%d	%.2f	%.2f	平均：%.2f  平均偏差値：%.2f\n", se, elite[1].valu, min_value, ave, deviation);

}

int roulette() // ルーレット選択
{
    int i = 0, r = 0;
    int num = 0;
    float sum = 0.0;
    float* p;

    p = (float*)malloc(sizeof(int) * num_ind);

    sum = 0;
    for (i = 0; i < num_ind; i++) {
        sum += h[i][0].f_value; // すべての合計
    }
    for (i = 0; i < num_ind; i++) {
        p[i] = h[i][0].f_value / sum; // 個体適応度 / 群体適応度
    }

    sum = 0;
    r = rand();
    for (i = 0; i < num_ind; i++) {
        sum += RAND_MAX * p[i]; // 1
        if (r <= sum) {
            num = i;
            break;
        }
    }
    if (num < 0)	num = roulette(); // エラーのための処理
    free(p);
    return(num);
}

void crossover(gene* g) {
    gene g2[1000] = { 0 }; // 新しい個体群を一時的に格納するための配列
    int num = 0; // 処理中の個体番号。2つずつ処理します。
    // n1, n2: 交叉するために選ばれた2つの親のインデックス
    int n1 = 0;
    int n2 = 0;
    int p = 0; // 交叉を行う遺伝子の位置
    int i, j;

    for (num = 0; num < num_ind; num += 2) {
        n1 = rand() % 10;
        n2 = rand() % 10;
        if (rand() <= RAND_MAX * cross) { // 交叉確率を満たす場合
            n1 = roulette();
            n2 = roulette();
            // 乱数の範囲指定公式：(int)( rand() * (最大値 - 最小値 + 1.0) / (1.0 + RAND_MAX) )
            p = (int)(rand() * ((chLen - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);
            // g[n1], g[n2]: 2つの親　g2[num], g2[num+1]: 2つの子
            // 交叉仕組み：親1の最初のpビットを継承し、残りのビットを親2から継承します
            // 子A
            for (i = 0; i < p; i++) {
                g2[num].ch[i] = g[n1].ch[i];
            }
            for (i = p; i < chLen; i++) {
                g2[num].ch[i] = g[n2].ch[i];
            }

            // 子B
            for (i = 0; i < p; i++) {
                g2[num + 1].ch[i] = g[n2].ch[i];
            }
            for (i = p; i < chLen; i++) {
                g2[num + 1].ch[i] = g[n1].ch[i];
            }
        }
        else {
            for (i = 0; i < chLen; i++) {
                n1 = roulette();
                n2 = roulette();
                g2[num].ch[i] = g[n1].ch[i];
                g2[num + 1].ch[i] = g[n2].ch[i];
            }
        }
    }

    // 新しい個体群g2を、元の個体群gに上書きして次世代の個体群とします
    for (j = 0; j < num_ind; j++) {
        for (i = 0; i < chLen; i++) {
            g[j].ch[i] = g2[j].ch[i]; // g[]を更新
        }
    }
}

void mutation(gene* g) // 突然変異
{
    int num = 0;
    int r = 0;
    int i = 0;
    int p = 0;
    for (num = 0; num < num_ind; num++) {
        if (rand() <= RAND_MAX * mut) { // 突然変異確率を満たす場合，1つの遺伝子を選択
            p = (int)(rand() * ((chLen - 1) + 1.0) / (1.0 + RAND_MAX));
            for (i = 0; i < chLen; i++) { // 1と0を逆転
                if (i == p) {
                    if (g[num].ch[i] == 0) g[num].ch[i] = 1;
                    else				g[num].ch[i] = 0;
                }
            }
            p = 0;
        }
    }
}

void elite_back(gene* g, gene* elite) { // エリート個体とvalu最小個体を交換

    int i = 0, j = 0;
    float ave = 0.0;
    float min1 = 1.0;

    int tmp = 0; // カウンターの初期化

    for (i = 0; i < num_ind; i++) { // 最小値探索
        if (h[i][0].f_value < min1) {
            min1 = h[i][0].f_value;
            tmp = i;
        }
    }

    for (j = 0; j < chLen; j++) {
        g[tmp].ch[j] = elite[1].ch[j]; // 最小値とエリートを交換
    }

    h[tmp][0].f_value = elite[1].f_value; // エリートの評価値と交換
    // ave = sum1 / kotai; // 合計値の計算
}

int main(void) {
    oriImg = imread("./imgs_Pro_GA/oriImg.png");
    maskImg = imread("./imgs_Pro_GA/maskImg.png");
    tarImg = imread("./imgs_Pro_GA/tarImg.png");
    Mat blurImg;
    Mat diffImg;
    Mat labelImg;
    if (maskImg.empty() || tarImg.empty())
    {
        maskGenerate();
        targetGenerate();
        Mat maskImg = imread("./imgs_Pro_GA/maskImg.png");
        Mat tarImg = imread("./imgs_Pro_GA/tarImg.png");
    }
    // Initializing ・・・
    srand((unsigned)time(NULL));
    gene g[num_ind]; // For storing the group of individuals 
    gene elite[10]; // For storing the elite individual of each generation
    elite[1].f_value = 0.0;

    //eliteValue eValue; // for storing the value-info of elite individual of each generation
    //eValue.precision = 0.0;
    //eValue.recall = 0.0;
    //eValue.f_value = 0.0;

    make(g); // Initializing the group of individuals
    //if ((fp = fopen("process_log.txt", "w")) == NULL) { // for working log
    //    printf("Opening Failed!\n");
    //}

    // Main Part
    for (int numGen = 1; numGen <= num_gen; numGen++) { // the loop of generation
        cout << "-------generation: " << numGen << "---------" << endl;
        // in the start of each generation, first calculate the fitness of the decision variable and storing it in h-array 
        phenotype(g);
        for (int numInd = 0; numInd < num_ind; numInd++) { // the loop of individual
            import_para(numInd); // read in the fitness of h-array and write it to the global variables
            // the process of bluring
            if (filterswitch_flag == 0)
            {
                medianBlur(oriImg, blurImg, fsize);
            }
            else
            {
                blur(oriImg, blurImg, Size(fsize, fsize));
            }
            // the differential process of oriImg and blurImg
            diffImg = sabun(oriImg, blurImg);
            threshold(diffImg, diffImg, binary, 255, THRESH_BINARY);
            if (pixellabelingmethod == 0)
            {
                labelImg = labeling_new4(diffImg);
            }
            else
            {
                labelImg = labeling_new8(diffImg);
            }
            pixellabelingmethod = 0;
            cv::bitwise_not(labelImg, labelImg);
            if (erodedilate_sequence == 0)
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = dilate_erode(labelImg);
                    }
                }
            }
            else if (erodedilate_sequence == 1)
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = erode_dilate(labelImg);
                    }
                }
            }
            calculateMetrics(labelImg, tarImg, maskImg, numInd);
        }
        fitness(g, elite, numGen);
        crossover(g);
        mutation(g);
        elite_back(g, elite);
    }
    vector<Mat> images = { oriImg, tarImg, labelImg };
    Mat res;
    hconcat(images, res);
    imgShow("res_p1", res);
    // imwrite("./imgs/final_opt.jpg", res);
    return 0;
}