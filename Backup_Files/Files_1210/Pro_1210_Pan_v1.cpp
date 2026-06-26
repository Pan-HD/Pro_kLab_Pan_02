#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define RAND_MAX 32767 // the max value of random number
#define chLen 26 // the length of chromosome
#define num_ind 100 // the nums of individuals in the group
#define num_gen 100 // the nums of generation of the GA algorithm
#define cross 0.8 // the rate of cross
#define mut 0.05 // the rate of mutation
#define numSets 4 // the num of sets(pairs)

// for storing the index of the individual with max f-value
int curMaxFvalIdx = 0;

// the declaration of 8 decision variables
int fsize = 0;
int binary = 0;
int linear = 0;
int filterswitch_flag;
int erodedilate_times;
int erodedilate_sequence = 0;
int abusolute_flag = 0;
int pixellabelingmethod = 0;

typedef struct {
    int ch[chLen]; // defining chromosomes by ch-array
    int fitness;
    double f_value; // the harmonic mean of precision and recall of the individual
}gene;

// the info of each generation, including the info of elite individual and the info of the group
typedef struct {
    double eliteFValue;
    double genMinFValue; // the min value of each gen, (max value is eliteFValue)
    double genAveFValue;
    double genDevFValue;
}genInfoType;

// for storing the fitness value of 8 decision variables
gene h[num_ind][8];

// for storing the info of each generation
genInfoType genInfo[num_gen];

// for storing the f-value of every individual in the group
double indFvalInfo[num_ind][numSets + 1];

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
        for (i = 0; i < 8; i++) {
            h[j][i].fitness = 0;
        }
    }

    for (j = 0; j < num_ind; j++) {
        // fsize - 6 bits
        i = 6;
        for (k = 0; k < 6; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][0].fitness += (int)pow(2.0, (double)i);
            }
        }
        // binary - 8 bits
        i = 8;
        for (k = 6; k < 14; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][1].fitness += (int)pow(2.0, (double)i);
            }
        }

        // linear - 5 bit
        i = 5;
        for (k = 14; k < 19; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][2].fitness += (int)pow(2.0, (double)i);
            }
        }

        // filterswich_flag - 1 bit
        i = 1;
        for (k = 19; k < 20; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][3].fitness += (int)pow(2.0, (double)i);
            }
        }
        // erodedilate_times - 3 bits
        i = 3;
        for (k = 20; k < 23; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][4].fitness += (int)pow(2.0, (double)i);
            }
        }
        // erodedilate_sequence - 1 bit
        i = 1;
        for (k = 23; k < 24; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][5].fitness += (int)pow(2.0, (double)i);
            }
        }
        // abusolute_flag - 1 bit
        i = 1;
        for (k = 24; k < 25; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][6].fitness += (int)pow(2.0, (double)i);
            }
        }
        // pixellabelingmethod - 1bit
        i = 1;
        for (k = 25; k < 26; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][7].fitness += (int)pow(2.0, (double)i);
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
    linear = (int)(1.0 + 0.5 * h[ko][2].fitness);
    filterswitch_flag = h[ko][3].fitness;
    erodedilate_times = h[ko][4].fitness;
    erodedilate_sequence = h[ko][5].fitness;
    abusolute_flag = h[ko][6].fitness;
    pixellabelingmethod = h[ko][7].fitness;
}

Mat labeling(Mat img, int connectivity) {
    Mat img_con;
    Mat stats, centroids;

    int label_x, label_y;
    int label_longer;
    int label_areaall;
    double label_cal;

    int i, j, label_num;

    label_num = cv::connectedComponentsWithStats(img, img_con, stats, centroids, connectivity, CV_32S);
    // colors: for storing the color of background and every connected areas 
    vector<Vec3b>colors(label_num + 1);
    colors[0] = Vec3b(0, 0, 0);
    colors[1] = Vec3b(255, 255, 255);

    for (i = 2; i <= label_num; i++)
    {
        // colors[i] = Vec3b(0, 0, 0);
        label_areaall = stats.at<int>(i, CC_STAT_AREA);
        label_x = stats.at<int>(i, CC_STAT_WIDTH);
        label_y = stats.at<int>(i, CC_STAT_HEIGHT);

        label_longer = label_x > label_y ? label_x : label_y;
        label_cal = label_longer * label_longer;

        // (int)(label_cal / label_areaall) < linear -> detected area is not a fold-area
        //  In fold-detect task: discard -> colors[i] = Vec3b(255, 255, 255);
        if ((int)(label_cal / label_areaall) < linear + 127)
        {
            colors[i] = Vec3b(0, 0, 0); // in spot-detect task
        }
        else {
            colors[i] = Vec3b(255, 255, 255); // in spot-detect task
        }
    }

    // CV_8UC3F3 channels
    Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
    for (j = 0; j < img_con.rows; j++) {
        for (i = 0; i < img_con.cols; i++)
        {
            int label = img_con.at<int>(j, i);
            CV_Assert(0 <= label && label <= label_num); // make sure the num of label is leagal
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

void calculateMetrics(Mat metaImg[], Mat tarImg[], Mat maskImg[], int numInd, int numGen) {
    Mat metaImg_g[numSets];
    Mat tarImg_g[numSets];
    Mat maskImg_g[numSets];
    for (int i = 0; i < numSets; i++) {
        cvtColor(metaImg[i], metaImg_g[i], cv::COLOR_BGR2GRAY);
        cvtColor(tarImg[i], tarImg_g[i], cv::COLOR_BGR2GRAY);
        cvtColor(maskImg[i], maskImg_g[i], cv::COLOR_BGR2GRAY);
    }

    double f1_score[numSets];

    for (int k = 0; k < numSets; k++) { // k: the index of the set being processed
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < maskImg_g[k].rows; i++) {
            for (int j = 0; j < maskImg_g[k].cols; j++) {
                if (maskImg_g[k].at<uchar>(i, j) == 0) {
                    continue;
                }
                if (metaImg_g[k].at<uchar>(i, j) == 0 && tarImg_g[k].at<uchar>(i, j) == 0) {
                    tp += 1;
                }
                if (metaImg_g[k].at<uchar>(i, j) == 0 && tarImg_g[k].at<uchar>(i, j) == 255) {
                    fp += 1;
                }
                if (metaImg_g[k].at<uchar>(i, j) == 255 && tarImg_g[k].at<uchar>(i, j) == 0) {
                    fn += 1;
                }
            }
        }
        if (tp == 0) tp += 1;
        if (fp == 0) fp += 1;
        if (fn == 0) fn += 1;
        double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
        f1_score[k] = calculateF1Score(precision, recall);
    }
    double sum_f1 = 0.0;
    for (int i = 0; i < numSets; i++) {
        indFvalInfo[numInd][i] = f1_score[i];
        sum_f1 += f1_score[i];
    }

    // h[numInd][0].f_value = sum_f1 / numSets;
    h[numInd][0].f_value = sum_f1;
    indFvalInfo[numInd][numSets] = sum_f1;
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
    curMaxFvalIdx = maxFValueIndex;
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
    if (num < 0)	num = roulette();
    return(num);
}

void crossover(gene* g) {
    gene g2[num_ind];
    int num = 0;
    int n1 = 0;
    int n2 = 0;
    int p = 0;
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
            g[j].ch[i] = g2[j].ch[i];
        }
    }
}

void mutation(gene* g)
{
    int num = 0;
    int r = 0;
    int i = 0;
    int p = 0;
    for (num = 0; num < num_ind; num++) {
        if (rand() <= RAND_MAX * mut) {
            p = (int)(rand() * ((chLen - 1) + 1.0) / (1.0 + RAND_MAX));
            for (i = 0; i < chLen; i++) {
                if (i == p) {
                    if (g[num].ch[i] == 0) g[num].ch[i] = 1;
                    else				g[num].ch[i] = 0;
                }
            }
            p = 0;
        }
    }
}

void elite_back(gene* g, gene* elite) {
    int i = 0, j = 0;
    double ave = 0.0;
    double min1 = 1.0;
    int tmp = 0;
    for (i = 0; i < num_ind; i++) {
        if (h[i][0].f_value < min1) {
            min1 = h[i][0].f_value;
            tmp = i;
        }
    }
    for (j = 0; j < chLen; j++) {
        g[tmp].ch[j] = elite[1].ch[j];
    }
    h[tmp][0].f_value = elite[1].f_value;
}

void multiProcess(Mat imgArr[][3]) {
    Mat blurImg[numSets];
    Mat diffImg[numSets];
    Mat biImg[numSets];
    Mat labelImg[numSets];

    char imgName_pro[numSets][256];
    char imgName_final[numSets][256];

    // for recording the f_value
    FILE* fl_fValue = nullptr;
    errno_t err = fopen_s(&fl_fValue, "./imgs_1209_v1/output/f_value.txt", "a");
    if (err != 0 || fl_fValue == nullptr) {
        perror("Cannot open the file");
        return;
    }

    // for recording the decision varibles
    FILE* fl_params = nullptr;
    errno_t err1 = fopen_s(&fl_params, "./imgs_1209_v1/output/params.txt", "a");
    if (err1 != 0 || fl_params == nullptr) {
        perror("Cannot open the file");
        return;
    }

    FILE* fl_maxFval = nullptr;
    errno_t err2 = fopen_s(&fl_maxFval, "./imgs_1209_v1/output/maxFvalInfo_final.txt", "a");
    if (err2 != 0 || fl_maxFval == nullptr) {
        perror("Cannot open the file");
        return;
    }

    srand((unsigned)time(NULL));
    gene g[num_ind]; // For storing the group of individuals
    gene elite[10]; // For storing the elite individual of each generation
    elite[1].f_value = 0.0;
    make(g); // Initializing the info of chrom of 100 individuals

    for (int numGen = 1; numGen <= num_gen; numGen++) {
        cout << "-------generation: " << numGen << "---------" << endl;
        phenotype(g);
        for (int numInd = 0; numInd < num_ind; numInd++) {
            import_para(numInd);
            for (int i = 0; i < numSets; i++) {
                // bluring
                if (filterswitch_flag) {
                    medianBlur(imgArr[i][0], blurImg[i], fsize);
                }
                else {
                    blur(imgArr[i][0], blurImg[i], Size(fsize, fsize));
                }
                threshold(blurImg[i], biImg[i], binary, 255, THRESH_BINARY);
                if (!pixellabelingmethod)
                {
                    // biImg with 1-channel has been changed to 3-channel
                    labelImg[i] = labeling(biImg[i], 4);
                }
                else
                {
                    labelImg[i] = labeling(biImg[i], 8);
                }
                // Morphology
                if (!erodedilate_sequence)
                {
                    if (erodedilate_times != 0) {
                        for (int i = 0; i < erodedilate_times; i++)
                        {
                            labelImg[i] = Morphology(labelImg[i], 1);
                        }
                    }
                }
                else
                {
                    if (erodedilate_times != 0) {
                        for (int i = 0; i < erodedilate_times; i++)
                        {
                            labelImg[i] = Morphology(labelImg[i], 0);
                        }
                    }
                }
            }
            Mat tarImg[numSets];
            Mat maskImg[numSets];
            for (int i = 0; i < numSets; i++) {
                tarImg[i] = imgArr[i][1];
                maskImg[i] = imgArr[i][2];
            }
            calculateMetrics(labelImg, tarImg, maskImg, numInd, numGen);
        }

        fitness(g, elite, numGen);
        crossover(g);
        mutation(g);
        elite_back(g, elite);
        printf("f_value: %.4f, binary: %d\n", elite[1].f_value, binary);
        if (numGen % 10 == 0) {
            for (int i = 0; i < numSets; i++) {
                sprintf_s(imgName_pro[i], "./imgs_1209_v1/output/img_0%d/Gen-%d.png", i + 1, numGen);
                imwrite(imgName_pro[i], labelImg[i]);
            }
        }
    }
    for (int i = 0; i < numSets; i++) {
        vector<Mat> images = { labelImg[i], imgArr[i][1], imgArr[i][2] };
        Mat res;
        hconcat(images, res);
        sprintf_s(imgName_final[i], "./imgs_1209_v1/output/img_0%d/imgs_final.png", i + 1);
        imwrite(imgName_final[i], res);
    }

    for (int i = 0; i < num_gen; i++) {
        fprintf(fl_fValue, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
    }
    fprintf(fl_params, "%d %d %d %d %d %d %d %d\n", fsize, binary, linear, filterswitch_flag, erodedilate_times, erodedilate_sequence, abusolute_flag, pixellabelingmethod);
    for (int i = 0; i <= numSets; i++) {
        fprintf(fl_maxFval, "%.4f ", indFvalInfo[curMaxFvalIdx][i]);
    }
    fprintf(fl_maxFval, "\n");

    fclose(fl_fValue);
    fclose(fl_params);
    fclose(fl_maxFval);
}

int main(void) {
    Mat imgArr[numSets][3]; // imgArr -> storing all images 4(4 pairs) * 3(ori, tar, mask)
    char inputPathName_ori[256];
    char inputPathName_tar[256];
    char inputPathName_mask[256];

    for (int i = 0; i < numSets; i++) {
        sprintf_s(inputPathName_ori, "./imgs_1209_v1/input/oriImg_0%d.png", i + 1);
        sprintf_s(inputPathName_tar, "./imgs_1209_v1/input/tarImg_0%d.png", i + 1);
        sprintf_s(inputPathName_mask, "./imgs_1209_v1/input/maskImg_general.png");

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
    multiProcess(imgArr);
    return 0;
}