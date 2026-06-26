#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define RAND_MAX 32767 // the max value of random number
#define chLen 17 // the length of chromosome
#define num_ind 100 // the nums of individuals in the group
#define num_gen 200 // the nums of generation of the GA algorithm
#define cross 0.95 // the rate of cross
#define mut 0.25 // the rate of mutation

// the declaration of 7 decision variables
int fsize = 0;
int binary = 0;
int abusolute_flag = 0;
int erodedilate_sequence = 0;
int filterswitch_flag;
int erodedilate_times;
int pixellabelingmethod = 0;

typedef struct {
    int ch[chLen]; // defining chromosomes by ch-array
    // float fitness; // the fitness of 7 decision variables
    int fitness;
    float f_value; // the harmonic mean of precision and recall of the individual
}gene;

// for storing the fitness value of 7 decision variables
gene h[num_ind][7];

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
}

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
            h[j][i].fitness = 0;
        }
    }

    for (j = 0; j < num_ind; j++) {
        // fsize - 2 bits
        i = 2;
        for (k = 0; k < 2; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][0].fitness += (int)pow(2.0, (double)i);
            }
        }
        // binary - 8 bits
        i = 8;
        for (k = 2; k < 10; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][1].fitness += (int)pow(2.0, (double)i);
            }
        }
        // filterswich_flag - 1 bit
        i = 1;
        for (k = 10; k < 11; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][2].fitness += (int)pow(2.0, (double)i);
            }
        }
        // erodedilate_times - 3 bits
        i = 3;
        for (k = 11; k < 14; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][3].fitness += (int)pow(2.0, (double)i);
            }
        }
        // erodedilate_sequence - 1 bit
        i = 1;
        for (k = 14; k < 15; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][4].fitness += (int)pow(2.0, (double)i);
            }
        }
        // abusolute_flag - 1 bit
        i = 1;
        for (k = 15; k < 16; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][5].fitness += (int)pow(2.0, (double)i);
            }
        }
        // pixellabelingmethod - 1bit
        i = 1;
        for (k = 16; k < 17; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][6].fitness += (int)pow(2.0, (double)i);
            }
        }
    }
}

void import_para(int ko) {
    fsize = 0;
    binary = 0;
    filterswitch_flag = 0;
    abusolute_flag = 0;
    fsize = 1 + 2 * h[ko][0].fitness;
    binary = h[ko][1].fitness;
    filterswitch_flag = h[ko][2].fitness;
    erodedilate_times = h[ko][3].fitness;
    erodedilate_sequence = h[ko][4].fitness;
    abusolute_flag = h[ko][5].fitness;
    pixellabelingmethod = h[ko][6].fitness;
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

Mat labeling(Mat img, int connectivity) {
    Mat img_con;
    Mat stats, centroids;
    int i, j, label_num;

    //int label_x, label_y;
    //int label_longer;
    //double label_cal;
    //int label_areaall;

    label_num = cv::connectedComponentsWithStats(img, img_con, stats, centroids, 8, CV_32S);
    vector<Vec3b>colors(label_num + 1); // ラベルごとにどの色を使うかを格納するためのものです
    colors[0] = Vec3b(0, 0, 0); // 背景の色を黒に設定
    colors[1] = Vec3b(255, 255, 255);
    for (i = 2; i < label_num; i++) // ラベルごとの処理
    {
        // colors[i] = Vec3b(255, 255, 255);
        colors[i] = Vec3b(0, 0, 0);
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

Mat Morphology(Mat img, int isDilFirst) {
    Mat dst;
    dst.create(img.size(), img.type());
    if (isDilFirst) {
        dilate(img, dst, Mat());
        erode(dst, dst, Mat());
    }
    else {
        erode(dst, dst, Mat());
        dilate(img, dst, Mat());
    }

    return dst;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculateMetrics(Mat metaImg, Mat tarImg, Mat maskImg, int numInd, int numGen) {
    Mat metaImg_g;
    Mat tarImg_g;
    Mat maskImg_g;
    cvtColor(metaImg, metaImg_g, cv::COLOR_BGR2GRAY);
    cvtColor(tarImg, tarImg_g, cv::COLOR_BGR2GRAY);
    cvtColor(maskImg, maskImg_g, cv::COLOR_BGR2GRAY);

    int tp = 0, fp = 0, fn = 0;

    for (int i = 0; i < maskImg_g.rows; i++) {
        for (int j = 0; j < maskImg_g.cols; j++) {
            if (maskImg_g.at<uchar>(i, j) == 0) {
                continue;
            }
            if (metaImg_g.at<uchar>(i, j) == 0 && tarImg_g.at<uchar>(i, j) == 0) {
                tp += 1;
            }
            if (metaImg_g.at<uchar>(i, j) == 0 && tarImg_g.at<uchar>(i, j) == 255) {
                fp += 1;
            }
            if (metaImg_g.at<uchar>(i, j) == 255 && tarImg_g.at<uchar>(i, j) == 0) {
                fn += 1;
            }
        }
    }
    // cout << "----gen: " << numGen << ", ind: " << numInd << ", tp fp fn: " << tp << " " << fp << " " << fn << "-------" << endl;
    if (tp == 0) tp += 1;
    if (fp == 0) fp += 1;
    if (fn == 0) fn += 1;
    double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
    double f1_score = calculateF1Score(precision, recall);
    h[numInd][0].f_value = f1_score;
}

void fitness(gene* g, gene* elite, int se) // 適応度の計算(エリート保存)
{
    int i = 0, j = 0;
    //double ave = 0.0;
    //double deviation = 0.0;
    //double variance = 0.0;

    // int sum1 = 0; // 初期化

    // エリート保存
    for (i = 0; i < num_ind; i++) {
        // sum1 += h[i][0].f_value;
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
    //float* p;
    //p = (float*)malloc(sizeof(int) * num_ind);

    float p[num_ind];

    //sum = 0;
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
    gene g2[num_ind]; // 新しい個体群を一時的に格納するための配列
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

void print_h() {
    cout << "fsize, binary, filterFlag, eroDilTimes, eroDilSeq, abuFlag, labelMethod" << endl;
    for (int i = 0; i < num_ind; i++) {
        for (int j = 0; j < 7; j++) {
            printf("%d  ", h[i][j].fitness);
        }
        printf("\n");
    }
}

int main(void) {
    // preparation of images
    Mat oriImg = imread("./imgs_Pro_GA/oriImg.png");
    Mat oriImg_g = imread("./imgs_Pro_GA/oriImg.png", 0);
    Mat maskImg = imread("./imgs_Pro_GA/maskImg.png");
    Mat tarImg = imread("./imgs_Pro_GA/tarImg.png");
    Mat blurImg;
    Mat diffImg;
    Mat biImg;
    Mat labelImg;

    // Initializing ・・・
    srand((unsigned)time(NULL));
    gene g[num_ind]; // For storing the group of individuals 
    gene elite[10]; // For storing the elite individual of each generation
    elite[1].f_value = 0.0;
    make(g);

    //char imgFileName[] = "./imgs_Pro_GA/Output/"

    // Main Part
    for (int numGen = 1; numGen <= num_gen; numGen++) {
        //for (int numGen = 1; numGen <= 1; numGen++) {
        cout << "-------generation: " << numGen << "---------" << endl;
        phenotype(g);
        for (int numInd = 0; numInd < num_ind; numInd++) {
            //for (int numInd = 0; numInd < 1; numInd++) {
                // global value of decision variables been written
            import_para(numInd);
            // bluring
            if (filterswitch_flag)
            {
                medianBlur(oriImg_g, blurImg, fsize);
            }
            else
            {
                blur(oriImg_g, blurImg, Size(fsize, fsize));
            }

            //diffImg = sabun(oriImg_g, blurImg);
            threshold(blurImg, biImg, binary, 255, THRESH_BINARY);

            if (!pixellabelingmethod)
            {
                // biImg with 1-channel has been changed to 3-channel
                labelImg = labeling(biImg, 4);
            }
            else
            {
                labelImg = labeling(biImg, 8);
            }

            //vector<Mat> images_01 = { labelImg, tarImg};
            //Mat res01;
            //hconcat(images_01, res01);
            //imgShow("res_01", res01);

            // bitwise_not(labelImg, labelImg);

            // Morphology
            if (!erodedilate_sequence)
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = Morphology(labelImg, 1);
                    }
                }
            }
            else
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = Morphology(labelImg, 0);
                    }
                }
            }
            calculateMetrics(labelImg, tarImg, maskImg, numInd, numGen);
        }
        fitness(g, elite, numGen);
        crossover(g);
        mutation(g);
        elite_back(g, elite);
        cout << "binary: " << binary << "  f_value: " << elite[1].f_value << endl;
    }
    vector<Mat> images = { labelImg, tarImg, maskImg };
    Mat res;
    hconcat(images, res);
    imgShow("res_p1", res);
}