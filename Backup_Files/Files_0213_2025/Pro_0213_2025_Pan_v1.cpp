#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define numSets 1 // the num of sets(pairs)
#define idSet 2 // for mark the selected set if the numSets been set of 1

// been modified (#define numDV 9 // the nums of decision-variables)
#define numDV 7 // the nums of decision-variables

// been modified (#define chLen 28 // the length of chromosome)
#define chLen 28 // the length of chromosome

#define num_ind 100 // the nums of individuals in the group
#define num_gen 150 // the nums of generation of the GA algorithm
#define cross 0.8 // the rate of cross
#define mut 0.15 // the rate of mutation
#define gaussianBlurFlag 1
#define medianBlurFlag 1


// for storing the index of the individual with max f-value
int curMaxFvalIdx = 0;

// the allocated nums of bit of the decision-variables
// been modified (int info_dv_arr[numDV] = { 5, 4, 4, 1, 2, 3, 3, 2, 3 };)
int info_dv_arr[numDV] = { 8, 4, 4, 4, 2, 3, 3 };

// the declaration of 7 decision variables
int threshVal = 17; // [0, 255] -> dv01 - 8bit
int gaussianSize = 17; // (n * 2 + 1) 3, 5, 7, 9, ..., 31 -> dv02 - 4bit
int circleOffset = 9; // [0, 15] -> dv03 - 4bit
int medianSize = 3; // (n * 2 + 1) 3, 5, 7, 9, ..., 31 -> dv04 - 4bit
int dilateTimes = 1; // -> dv05 - 2bit
int aspectOffset = 1; // [0, 7] -> dv06 - 3bit
int contourPixNums = 7; // [0, 7] -> dv07 - 3bit

// been modified 
//int dilateTimes_s2 = 1; // [0, 3] -> dv08 - 2bit
//int contourPixNums_s2 = 3; // [0, 7] -> dv09 - 3bit

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

// for storing the fitness value of 7 decision variables
gene h[num_ind][numDV];

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

/*
  function: Convert the decision variable information in the chromosome corresponding to each individual
            from binary to decimal and store it in the h array
*/
void phenotype(gene* g)
{
    int i = 0, j = 0;
    // initializing the fitness in h-array by assigning 0
    for (j = 0; j < num_ind; j++) {
        for (i = 0; i < numDV; i++) {
            h[j][i].fitness = 0;
        }
    }

    for (int idxInd = 0; idxInd < num_ind; idxInd++) { // the loop of inds
        int curIdx_chrom = 0;
        for (int idx_dv = 0; idx_dv < numDV; idx_dv++) {
            int len_curDv = info_dv_arr[idx_dv];
            int sum_val = 0;
            for (int idx = curIdx_chrom + len_curDv - 1; idx >= curIdx_chrom; idx--) {
                sum_val += g[idxInd].ch[idx] * (int)pow(2.0, (double)(len_curDv - (idx - curIdx_chrom) - 1));
            }
            h[idxInd][idx_dv].fitness = sum_val;
            curIdx_chrom += len_curDv;
        }
    }
}

void import_para(int ko) {
    // dv01
    threshVal = h[ko][0].fitness;
    // dv02
    //if (h[ko][1].fitness >= 0 && h[ko][1].fitness <= 8) {
    //    gaussianSize = 9 + 2 * h[ko][1].fitness;
    //}
    //else {
    //    do {
    //        gaussianSize = rand() % 17 + 9;
    //    } while (gaussianSize % 2 == 0);
    //}
    gaussianSize = h[ko][1].fitness * 2 + 1;
    // dv03
    circleOffset = h[ko][2].fitness;
    // dv04
    medianSize = h[ko][3].fitness * 2 + 1;
    // dv05
    dilateTimes = h[ko][4].fitness;
    // dv06
    aspectOffset = h[ko][5].fitness;
    // dv07
    contourPixNums = h[ko][6].fitness;
    // been modified
    //dilateTimes_s2 = h[ko][7].fitness;
    //contourPixNums_s2 = h[ko][8].fitness;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], Mat maskImg_g[], int numInd, int numGen) {
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

/*
  function: Sobel processing
*/
void gradCal(Mat& srcImg, Mat& dstImg) {
    Mat sobelX, sobelY, gradientMagnitude;
    Sobel(srcImg, sobelX, CV_64F, 1, 0, 1);
    Sobel(srcImg, sobelY, CV_64F, 0, 1, 1);
    magnitude(sobelX, sobelY, gradientMagnitude);
    normalize(gradientMagnitude, dstImg, 0, 255, NORM_MINMAX, CV_8U);
}

vector<Vec3f> circleDetect(Mat img) {
    Mat blurred;

    // been modified
    if (!gaussianBlurFlag) {
        blurred = img.clone();
    }
    else {
        GaussianBlur(img, blurred, Size(gaussianSize, gaussianSize), 0, 0);
    }

    // GaussianBlur(img, blurred, Size(gaussianSize, gaussianSize), 0, 0);
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, blurred.rows / 8, 200, 100, 0, 0);
    return circles;
}

int comDistance(int y, int x, Vec3f circle) {
    int centerX = (int)circle[0];
    int centerY = (int)circle[1];
    int radius = (int)circle[2];
    int distance = (int)sqrt(pow((double)(x - centerX), 2) + pow((double)(y - centerY), 2));
    if (distance > radius) {
        return 0;
    }
    else if (distance > radius - circleOffset && distance <= radius) {
        return 1;
    }
    else {
        return 2;
    }
}

void contourProcess(Mat& metaImg, Mat& resImg, int aspectRatio, int pixNums, vector<Vec3f> circles, int aspectFlag) {
    vector<vector<Point>> contours;
    findContours(metaImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Mat mask = Mat::zeros(metaImg.size(), CV_8UC1);
    for (const auto& contour : contours) {
        Rect bounding_box = boundingRect(contour);
        double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
        if (aspectFlag) {
            if ((aspect_ratio < (1 - aspectRatio * 0.1) || aspect_ratio >(1 + aspectRatio * 0.1)) && cv::contourArea(contour) < pixNums) {
                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
            }
        }
        else { // just need to focus on the proportions(areas), since the probability of the aspect_ratio equals "1.0000"
            // when the second time to find contours (without long-slender ribon inside)
            if (cv::contourArea(contour) < pixNums) {
                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
            }
        }

    }
    // imgShow("mask", mask);

    if (circles.size() != 0) {
        for (int y = 0; y < resImg.rows; y++) {
            for (int x = 0; x < resImg.cols; x++) {
                if (comDistance(y, x, circles[0]) == 2) {
                    if (mask.at<uchar>(y, x) == 255) {
                        resImg.at<uchar>(y, x) = 255;
                    }
                }
            }
        }
    }
    // imgShow("res", resImg);
}

void multiProcess(Mat imgArr[][3]) {
    Mat edges_s1[numSets];
    Mat biImg[numSets];
    Mat blurImg_mask[numSets];
    Mat metaImg[numSets];

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    char imgName_pro[numSets][256];
    char imgName_final[numSets][256];

    // for recording the f_value of every generation (max, min, ave, dev)
    FILE* fl_fValue = nullptr;
    errno_t err = fopen_s(&fl_fValue, "./imgs_0213_2025_v1/output/f_value.txt", "a");
    if (err != 0 || fl_fValue == nullptr) {
        perror("Cannot open the file");
        return;
    }

    // for recording the decision varibles
    FILE* fl_params = nullptr;
    errno_t err1 = fopen_s(&fl_params, "./imgs_0213_2025_v1/output/params.txt", "a");
    if (err1 != 0 || fl_params == nullptr) {
        perror("Cannot open the file");
        return;
    }

    // for recording the f_value of elite-ind in last gen (setX1, setX2, ..., Max)
    FILE* fl_maxFval = nullptr;
    errno_t err2 = fopen_s(&fl_maxFval, "./imgs_0213_2025_v1/output/maxFvalInfo_final.txt", "a");
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
                gradCal(imgArr[i][0], edges_s1[i]);
                threshold(edges_s1[i], biImg[i], threshVal, 255, THRESH_BINARY);
                vector<Vec3f> circles = circleDetect(biImg[i]);
                if (circles.size() != 0) {
                    for (int y = 0; y < biImg[i].rows; y++) {
                        for (int x = 0; x < biImg[i].cols; x++) {
                            if (comDistance(y, x, circles[0]) == 0) {
                                biImg[i].at<uchar>(y, x) = 0;
                            }
                            else if (comDistance(y, x, circles[0]) == 1) {
                                biImg[i].at<uchar>(y, x) = 255;
                            }
                            else {
                                biImg[i].at<uchar>(y, x) = biImg[i].at<uchar>(y, x) == 0 ? 255 : 0;
                            }
                        }
                    }
                }

                // been modified
                if (!medianBlurFlag) {
                    blurImg_mask[i] = biImg[i].clone();
                }
                else {
                    medianBlur(biImg[i], blurImg_mask[i], medianSize);
                }

                // medianBlur(biImg[i], blurImg_mask[i], medianSize);

                for (int idxET = 0; idxET < dilateTimes; idxET++) {
                    erode(blurImg_mask[i], blurImg_mask[i], kernel);
                }

                contourProcess(blurImg_mask[i], biImg[i], aspectOffset, 100 * contourPixNums, circles, 1);

                // been modified
                //metaImg[i] = biImg[i].clone();
                //for (int idxET = 0; idxET < dilateTimes_s2; idxET++) {
                //    erode(metaImg[i], metaImg[i], kernel);
                //}
                //contourProcess(metaImg[i], biImg[i], 0, 100 * contourPixNums_s2, circles, 0);

            }
            Mat tarImg[numSets];
            Mat maskImg[numSets];
            for (int i = 0; i < numSets; i++) {
                tarImg[i] = imgArr[i][1];
                maskImg[i] = imgArr[i][2];
            }
            calculateMetrics(biImg, tarImg, maskImg, numInd, numGen);
        }

        fitness(g, elite, numGen);
        crossover(g);
        mutation(g);
        elite_back(g, elite);
        printf("f_value: %.4f\n", elite[1].f_value);

        if (numGen % 10 == 0) {
            if (numSets == 1) {
                sprintf_s(imgName_pro[0], "./imgs_0213_2025_v1/output/img_0%d/Gen-%d.png", idSet, numGen);
                imwrite(imgName_pro[0], biImg[0]);
            }
            else {
                for (int i = 0; i < numSets; i++) {
                    sprintf_s(imgName_pro[i], "./imgs_0213_2025_v1/output/img_0%d/Gen-%d.png", i + 1, numGen);
                    imwrite(imgName_pro[i], biImg[i]);
                }
            }
        }
    }

    if (numSets == 1) {
        vector<Mat> images = { biImg[0], imgArr[0][1], imgArr[0][2] };
        Mat res;
        hconcat(images, res);
        sprintf_s(imgName_final[0], "./imgs_0213_2025_v1/output/img_0%d/imgs_final.png", idSet);
        imwrite(imgName_final[0], res);
    }
    else {
        for (int i = 0; i < numSets; i++) {
            vector<Mat> images = { biImg[i], imgArr[i][1], imgArr[i][2] };
            Mat res;
            hconcat(images, res);
            sprintf_s(imgName_final[i], "./imgs_0213_2025_v1/output/img_0%d/imgs_final.png", i + 1);
            imwrite(imgName_final[i], res);
        }
    }

    for (int i = 0; i < num_gen; i++) {
        fprintf(fl_fValue, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
    }

    // been modified
    // (fprintf(fl_params, "%d %d %d %d %d %d %d %d %d\n", threshVal, gaussianSize, circleOffset, dilateFlag, dilateTimes, aspectOffset, contourPixNums, dilateTimes_s2, contourPixNums_s2);)
    fprintf(fl_params, "Stat: %d%d, deci-vari: %d %d %d %d %d %d %d\n", gaussianBlurFlag, medianBlurFlag, threshVal, gaussianSize, circleOffset, medianSize, dilateTimes, aspectOffset, contourPixNums);

    for (int i = 0; i <= numSets; i++) {
        fprintf(fl_maxFval, "%.4f ", indFvalInfo[curMaxFvalIdx][i]);
    }
    fprintf(fl_maxFval, "\n");

    fclose(fl_fValue);
    fclose(fl_params);
    fclose(fl_maxFval);
}

int main(void) {
    Mat imgArr[numSets][3]; // imgArr -> storing all images 2(2 pairs) * 3(ori, tar, mask)
    char inputPathName_ori[256];
    char inputPathName_tar[256];
    char inputPathName_mask[256];

    if (numSets == 1) {
        sprintf_s(inputPathName_ori, "./imgs_0213_2025_v1/input/oriImg_0%d.png", idSet);
        sprintf_s(inputPathName_tar, "./imgs_0213_2025_v1/input/tarImg_0%d.png", idSet);
        sprintf_s(inputPathName_mask, "./imgs_0213_2025_v1/input/maskImg_general.png");
        for (int j = 0; j < 3; j++) {
            if (j == 0) {
                imgArr[0][j] = imread(inputPathName_ori, 0);
            }
            else if (j == 1) {
                imgArr[0][j] = imread(inputPathName_tar, 0);
            }
            else {
                imgArr[0][j] = imread(inputPathName_mask, 0);
            }
        }
    }
    else {
        for (int i = 0; i < numSets; i++) {
            sprintf_s(inputPathName_ori, "./imgs_0213_2025_v1/input/oriImg_0%d.png", i + 1);
            sprintf_s(inputPathName_tar, "./imgs_0213_2025_v1/input/tarImg_0%d.png", i + 1);
            sprintf_s(inputPathName_mask, "./imgs_0213_2025_v1/input/maskImg_general.png");

            for (int j = 0; j < 3; j++) {
                if (j == 0) {
                    imgArr[i][j] = imread(inputPathName_ori, 0);
                }
                else if (j == 1) {
                    imgArr[i][j] = imread(inputPathName_tar, 0);
                }
                else {
                    imgArr[i][j] = imread(inputPathName_mask, 0);
                }
            }
        }
    }
    multiProcess(imgArr);
    return 0;
}