#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define chLen 26 // the length of chromosome
#define numSets 4 // the num of sets(pairs)
#define tempMax 3.5645 // this value would be compared with all the f1_values, better values would be writed

int fitnessArr[8];

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
    int curCase;
    double fValuArr[numSets + 1];
    int paramArr[8]; // for storing the decision variables in the best case
}bestCaseType;

bestCaseType bestCaseInfo;

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
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
        erode(img, dst, Mat());
        dilate(dst, dst, Mat());
    }
    return dst;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculateMetrics(FILE* file, Mat metaImg[], Mat tarImg[], Mat maskImg[], int curCase) {
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
        sum_f1 += f1_score[i];
    }
    // Modifying - ing
    if (sum_f1 >= tempMax) {
        bestCaseInfo.curCase = curCase;

        for (int i = 0; i < numSets; i++) {
            bestCaseInfo.fValuArr[i] = f1_score[i];
        }
        bestCaseInfo.fValuArr[numSets] = sum_f1;

        bestCaseInfo.paramArr[0] = fsize;
        bestCaseInfo.paramArr[1] = binary;
        bestCaseInfo.paramArr[2] = linear;
        bestCaseInfo.paramArr[3] = filterswitch_flag;
        bestCaseInfo.paramArr[4] = erodedilate_times;
        bestCaseInfo.paramArr[5] = erodedilate_sequence;
        bestCaseInfo.paramArr[6] = abusolute_flag;
        bestCaseInfo.paramArr[7] = pixellabelingmethod;

        fprintf(file, "-------curCase: %d-------\n", curCase);
        // f1_value(4+1)
        for (int i = 0; i <= numSets; i++) {
            fprintf(file, "%.4f ", bestCaseInfo.fValuArr[i]);
        }
        fprintf(file, "\n");
        // 8 deci-varis ...
        for (int i = 0; i < 8; i++) {
            fprintf(file, "%d ", bestCaseInfo.paramArr[i]);
        }
        fprintf(file, "\n");
    }
}


void import_para() {
    fsize = 0;
    binary = 0;
    filterswitch_flag = 0;
    abusolute_flag = 0;
    fsize = 1 + 2 * fitnessArr[0];
    binary = fitnessArr[1];
    linear = (int)(1.0 + 0.5 * fitnessArr[2]);
    filterswitch_flag = fitnessArr[3];
    erodedilate_times = fitnessArr[4];
    erodedilate_sequence = fitnessArr[5];
    abusolute_flag = fitnessArr[6];
    pixellabelingmethod = fitnessArr[7];
}

void phenotype(int chromArr[])
{
    int i;
    for (int idx = 0; idx < 8; idx++) {
        fitnessArr[idx] = 0;
    }

    i = 6;
    for (int k = 0; k < 6; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[0] += (int)pow(2.0, (double)i);
        }
    }

    i = 8;
    for (int k = 6; k < 14; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[1] += (int)pow(2.0, (double)i);
        }
    }

    i = 5;
    for (int k = 14; k < 19; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[2] += (int)pow(2.0, (double)i);
        }
    }

    i = 1;
    for (int k = 19; k < 20; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[3] += (int)pow(2.0, (double)i);
        }
    }

    i = 3;
    for (int k = 20; k < 23; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[4] += (int)pow(2.0, (double)i);
        }
    }

    i = 1;
    for (int k = 23; k < 24; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[5] += (int)pow(2.0, (double)i);
        }
    }

    i = 1;
    for (int k = 24; k < 25; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[6] += (int)pow(2.0, (double)i);
        }
    }

    i = 1;
    for (int k = 25; k < 26; k++) {
        i--;
        if (chromArr[k] == 1) {
            fitnessArr[7] += (int)pow(2.0, (double)i);
        }
    }
}

void singleTimeProcess(FILE* file, int chromArr[], Mat imgArr[][3], int curCase) {
    phenotype(chromArr);
    import_para();

    Mat blurImg[numSets];
    Mat diffImg[numSets];
    Mat biImg[numSets];
    Mat labelImg[numSets];

    for (int i = 0; i < numSets; i++) {
        // bluring
        if (filterswitch_flag) {
            medianBlur(imgArr[i][0], blurImg[i], fsize);
        }
        else {
            blur(imgArr[i][0], blurImg[i], Size(fsize, fsize));
        }

        // binary
        threshold(blurImg[i], biImg[i], binary, 255, THRESH_BINARY);

        // labeling
        if (!pixellabelingmethod)
        {
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
                for (int idx_edt = 0; idx_edt < erodedilate_times; idx_edt++)
                {
                    labelImg[i] = Morphology(labelImg[i], 1);
                }
            }
        }
        else
        {
            if (erodedilate_times != 0) {
                for (int idx_edt = 0; idx_edt < erodedilate_times; idx_edt++)
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
    calculateMetrics(file, labelImg, tarImg, maskImg, curCase);
}

void arrIteration(int arr[], int len) {
    int fullFlag = 1;
    for (int i = 0; i < len; i++) {
        if (arr[i] == 0) {
            fullFlag = 0;
            break;
        }
    }
    if (fullFlag) return;

    for (int idx = len - 1; idx >= 0; idx--) {
        if (arr[idx] == 0) {
            arr[idx] = 1;
            break;
        }
        arr[idx] = 0;
    }
}

int main(void) {
    FILE* file = nullptr;
    errno_t err = fopen_s(&file, "./imgs_1211_v2/output/Output_BestCase.txt", "a");
    if (err != 0 || file == nullptr) {
        perror("Cannot open the file");
        return -1;
    }

    // Initializing the bestCaseInfo
    bestCaseInfo.curCase = 0;
    for (int i = 0; i <= numSets; i++) {
        bestCaseInfo.fValuArr[i] = 0.0;
    }
    for (int i = 0; i < 8; i++) {
        bestCaseInfo.paramArr[i] = 0;
    }

    Mat imgArr[numSets][3]; // imgArr -> storing all images 4(4 pairs) * 3(ori, tar, mask)
    char inputPathName_ori[256];
    char inputPathName_tar[256];
    char inputPathName_mask[256];

    for (int i = 0; i < numSets; i++) {
        sprintf_s(inputPathName_ori, "./imgs_1211_v2/input/oriImg_0%d.png", i + 1);
        sprintf_s(inputPathName_tar, "./imgs_1211_v2/input/tarImg_0%d.png", i + 1);
        sprintf_s(inputPathName_mask, "./imgs_1211_v2/input/maskImg_general.png");

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

    int chromArr[chLen];
    for (int i = 0; i < chLen; i++) {
        chromArr[i] = 0;
    }
    int count = (int)pow(2.0, (double)chLen);
    int halfCount = count / 2;
    for (int caseIdx = 0; caseIdx < halfCount; caseIdx++) {
        // for (int caseIdx = halfCount; caseIdx < count; caseIdx++) {
        printf("curCase: %d\n", caseIdx);
        singleTimeProcess(file, chromArr, imgArr, caseIdx);
        arrIteration(chromArr, chLen);
    }
    fclose(file);
    return 0;
}
