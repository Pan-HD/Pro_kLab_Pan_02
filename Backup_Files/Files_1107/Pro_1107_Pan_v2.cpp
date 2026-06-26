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
#define num_gen 100 // the nums of generation of the GA algorithm
#define cross 0.8 // the rate of cross
#define mut 0.05 // the rate of mutation

// FILE* file = nullptr; // output of params

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
    int fitness;
    double f_value; // the harmonic mean of precision and recall of the individual
}gene;

// the info of each generation, including the info of elite individual and the info of the group
typedef struct {
    // double elitePrecision;
    // double eliteRecall;
    double eliteFValue;
    double genMinFValue; // the min value of each gen, (max value is eliteFValue)
    double genAveFValue;
    double genDevFValue;
}genInfoType;

// for storing the fitness value of 7 decision variables
gene h[num_ind][7];

// for storing the info of each generation
genInfoType genInfo[num_gen];

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

Mat labeling(Mat img, int connectivity) {
    Mat img_con;
    Mat stats, centroids;
    int i, j, label_num;

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
    if (tp == 0) tp += 1;
    if (fp == 0) fp += 1;
    if (fn == 0) fn += 1;
    double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
    double f1_score = calculateF1Score(precision, recall);
    h[numInd][0].f_value = f1_score;
}

void fitness(gene* g, gene* elite, int se) // for storing the info of elite individual
{
    int i = 0, j = 0;
    double minFValue = h[0][0].f_value;
    double maxFValue = h[0][0].f_value;
    double aveFValue = 0.0;
    double deviation = 0.0;
    double variance = 0.0;
    double sumFValue = 0.0;
    int maxFValueIndex = 0;

    for (i = 0; i < num_ind; i++) {
        sumFValue += h[i][0].f_value;
        if (h[i][0].f_value > maxFValue) {
            maxFValue = h[i][0].f_value;
            maxFValueIndex = i;
        }
        if (h[i][0].f_value < minFValue) {
            minFValue = h[i][0].f_value;
        }
    }

    elite[1].f_value = maxFValue;
    for (j = 0; j < chLen; j++) {
        elite[1].ch[j] = g[maxFValueIndex].ch[j];
    }
    aveFValue = sumFValue / num_ind;
    genInfo[se - 1].eliteFValue = h[maxFValueIndex][0].f_value;
    genInfo[se - 1].genMinFValue = minFValue;
    genInfo[se - 1].genAveFValue = aveFValue;
    for (i = 0; i < num_ind; i++)
    {
        double diff = h[i][0].f_value - aveFValue;
        variance += diff * diff;
    }
    deviation = sqrt(variance / num_ind);
    genInfo[se - 1].genDevFValue = deviation;
}

int roulette()
{
    int i = 0, r = 0;
    int num = 0;
    double sum = 0.0;
    double p[num_ind];

    for (i = 0; i < num_ind; i++) {
        sum += h[i][0].f_value;
    }
    for (i = 0; i < num_ind; i++) {
        p[i] = h[i][0].f_value / sum;
    }

    sum = 0;
    r = rand();
    for (i = 0; i < num_ind; i++) {
        sum += RAND_MAX * p[i];
        if (r <= sum) {
            num = i;
            break;
        }
    }
    if (num < 0)	num = roulette(); // エラーのための処理
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
        if (rand() <= RAND_MAX * cross) {
            n1 = roulette();
            n2 = roulette();
            p = (int)(rand() * ((chLen - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);
            for (i = 0; i < p; i++) {
                g2[num].ch[i] = g[n1].ch[i];
            }
            for (i = p; i < chLen; i++) {
                g2[num].ch[i] = g[n2].ch[i];
            }

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
    double ave = 0.0;
    double min1 = 1.0;
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
}

void singleProcess(Mat oriImg_g, Mat tarImg, Mat maskImg, int num_img) {
    Mat blurImg;
    Mat diffImg;
    Mat biImg;
    Mat labelImg;
    char imgName_pro[256];
    char imgName_final[256];
    char paramAddName[256];

    sprintf_s(paramAddName, "./imgs_1107_v2/output/img_0%d/params.txt", num_img);

    FILE* file = nullptr;
    errno_t err = fopen_s(&file, paramAddName, "a");
    if (err != 0 || file == nullptr) {
        perror("Cannot open the file");
        return;
    }

    srand((unsigned)time(NULL));
    gene g[num_ind]; // For storing the group of individuals 
    gene elite[10]; // For storing the elite individual of each generation
    elite[1].f_value = 0.0;
    make(g);

    for (int numGen = 1; numGen <= num_gen; numGen++) {
        cout << "-------generation: " << numGen << "---------" << endl;
        phenotype(g);
        for (int numInd = 0; numInd < num_ind; numInd++) {
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
        cout << "fsize: " << fsize << endl;

        if (numGen % 10 == 0) {
            sprintf_s(imgName_pro, "./imgs_1107_v2/output/img_0%d/Gen-%d.png", num_img, numGen);
            imwrite(imgName_pro, labelImg);
        }
    }

    for (int i = 0; i < num_gen; i++) {
        cout << "gen-" << i + 1 << "->" << "eliV: " << genInfo[i].eliteFValue << "  genMin: " << genInfo[i].genMinFValue << "  genAve: " << genInfo[i].genAveFValue << "  genDev: " << genInfo[i].genDevFValue << endl;
        fprintf(file, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
    }
    vector<Mat> images = { labelImg, tarImg, maskImg };
    Mat res;
    hconcat(images, res);
    // imgShow("res_p1", res);
    sprintf_s(imgName_final, "./imgs_1107_v2/output/img_0%d/imgs_final.png", num_img);
    imwrite(imgName_final, res);
    fclose(file);
}

int main(void) {
    Mat imgArr[5][3];
    char inputPathName_ori[256];
    char inputPathName_tar[256];
    char inputPathName_mask[256];

    for (int i = 0; i < 5; i++) {
        sprintf_s(inputPathName_ori, "./imgs_1107_v2/input/oriImg_0%d.png", i + 1);
        sprintf_s(inputPathName_tar, "./imgs_1107_v2/input/tarImg_0%d.png", i + 1);
        if (i == 1) {
            sprintf_s(inputPathName_mask, "./imgs_1107_v2/input/maskImg_02.png");
        }
        else {
            sprintf_s(inputPathName_mask, "./imgs_1107_v2/input/maskImg_general.png");
        }
        for (int j = 0; j < 3; j++) {
            if (j == 0) {
                imgArr[i][j] = imread(inputPathName_ori, 0);
            }
            else if (j == 1) {
                imgArr[i][j] = imread(inputPathName_tar);
            }
            else {
                imgArr[i][j] = imread(inputPathName_mask);
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        cout << "---------img_0" << i + 1 << "-----------" << endl;
        singleProcess(imgArr[i][0], imgArr[i][1], imgArr[i][2], i + 1);
    }

    return 0;
}