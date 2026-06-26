#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

#define mutateRate 0.3

using namespace cv;
using namespace std;
// 定义基因树节点类型
enum OperationType {
	BLUR_TYPE,    // 滤波类型（中值滤波 or 均值滤波）
	KERNEL_SIZE,  // 滤波核大小
	THRESHOLD     // 二值化阈值
};
// 基因树节点结构
struct GeneNode {
	OperationType opType;
	float param;
	GeneNode* left;
	GeneNode* right;
};

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

// **应用基因树到图像处理**
Mat applyGeneTree(const Mat& image, GeneNode* root, int showFlag) {
	Mat result = image.clone();
	int filterType = 0; // 0: average, 1: Median
	int kernelSize = 3;
	int thresholdValue = 128;
	// **遍历整棵基因树，提取参数**
	queue<GeneNode*> q;
	q.push(root);
	while (!q.empty()) {
		GeneNode* current = q.front();
		q.pop();
		if (current->opType == BLUR_TYPE) {
			filterType = static_cast<int>(current->param);
		}
		else if (current->opType == KERNEL_SIZE) {
			kernelSize = static_cast<int>(current->param);
			if (kernelSize % 2 == 0) kernelSize += 1;  // 确保 kernel size 为奇数
		}
		else if (current->opType == THRESHOLD) {
			thresholdValue = static_cast<int>(current->param);
			thresholdValue = max(0, min(thresholdValue, 255)); // 限制范围
		}
		if (current->left) q.push(current->left);
		if (current->right) q.push(current->right);
	}

	if (showFlag) {
		printf("Final----kernelSize: %d, thresholdValue: %d\n", kernelSize, thresholdValue);
	}

	if (showFlag) {
		imgShow("00-res", result);
	}

	// **执行滤波**
	if (filterType == 1) {
		medianBlur(result, result, kernelSize);
	}
	else {
		blur(result, result, Size(kernelSize, kernelSize));
	}

	if (showFlag) {
		imgShow("01-blur", result);
	}

	// **执行二值化**
	// threshold(result, result, 127, 255, THRESH_BINARY);
	threshold(result, result, thresholdValue, 255, THRESH_BINARY);
	if (showFlag) {
		imgShow("02-thresh", result);
	}

	return result;
}
// **计算精准率、召回率、F 值**
float calculatePrecision(int TP, int FP) { return (TP + FP == 0) ? 0 : static_cast<float>(TP) / (TP + FP); }
float calculateRecall(int TP, int FN) { return (TP + FN == 0) ? 0 : static_cast<float>(TP) / (TP + FN); }
float calculateFValue(float precision, float recall) { return (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall); }

double calculateF1Score(double precision, double recall) {
	if (precision + recall == 0) return 0.0;
	return 2.0 * (precision * recall) / (precision + recall);
}

// **评估基因树**
float evaluateGeneTree(const Mat& image, GeneNode* node, const Mat& targetImage) {
	Mat processedImage = applyGeneTree(image, node, 0); // processedImg, tarImg

	int tp = 0, fp = 0, fn = 0;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (processedImage.at<uchar>(i, j) == 0 && targetImage.at<uchar>(i, j) == 0) {
				tp += 1;
			}
			if (processedImage.at<uchar>(i, j) == 0 && targetImage.at<uchar>(i, j) == 255) {
				fp += 1;
			}
			if (processedImage.at<uchar>(i, j) == 255 && targetImage.at<uchar>(i, j) == 0) {
				fn += 1;
			}
		}
	}
	if (tp == 0) tp += 1;
	if (fp == 0) fp += 1;
	if (fn == 0) fn += 1;
	double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
	double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
	float score = (float)calculateF1Score(precision, recall);

	return score;
	//Mat diff;
	//absdiff(processedImage, targetImage, diff);
	//Mat binaryDiff;
	//threshold(diff, binaryDiff, 128, 255, THRESH_BINARY);
	//int TP = 0, FP = 0, FN = 0;
	//for (int i = 0; i < binaryDiff.rows; i++) {
	//	for (int j = 0; j < binaryDiff.cols; j++) {
	//		uchar pixelDiff = binaryDiff.at<uchar>(i, j);
	//		if (pixelDiff == 0 && targetImage.at<uchar>(i, j) == 0) TP++;
	//		else if (pixelDiff == 0 && targetImage.at<uchar>(i, j) == 255) FP++;
	//		else if (pixelDiff == 255 && targetImage.at<uchar>(i, j) == 0) FN++;
	//	}
	//}
	//return calculateFValue(calculatePrecision(TP, FP), calculateRecall(TP, FN));
}
// **交叉与变异**
GeneNode* crossover(GeneNode* parent1, GeneNode* parent2) {
	if (!parent1 || !parent2) return nullptr;
	GeneNode* child = new GeneNode(*parent1);
	child->param = (rand() % 2 == 0) ? parent1->param : parent2->param;
	child->left = crossover(parent1->left, parent2->left);
	child->right = crossover(parent1->right, parent2->right);
	return child;
}
void mutate(GeneNode* node) {
	if (!node) return;
	if (rand() <= RAND_MAX * mutateRate) {
		if (node->opType == BLUR_TYPE) node->param = rand() % 2;
		else if (node->opType == KERNEL_SIZE) node->param = (rand() % 5) * 2 + 3;
		else if (node->opType == THRESHOLD) node->param = rand() % 256;
	}
	mutate(node->left);
	mutate(node->right);
}
// **初始化种群**
vector<GeneNode*> initializePopulation(int populationSize) {
	vector<GeneNode*> population;
	for (int i = 0; i < populationSize; i++) {
		GeneNode* filterNode = new GeneNode{ BLUR_TYPE, static_cast<float>(rand() % 2), nullptr, nullptr };
		GeneNode* kernelNode = new GeneNode{ KERNEL_SIZE, static_cast<float>((rand() % 5) * 2 + 3), nullptr, nullptr };
		GeneNode* thresholdNode = new GeneNode{ THRESHOLD, static_cast<float>(rand() % 256), nullptr, nullptr };
		filterNode->left = kernelNode;
		filterNode->right = thresholdNode;

		// kernelNode->left = thresholdNode;

		population.push_back(filterNode);
	}
	return population;
}
// **遗传编程**
GeneNode* geneticAlgorithm(const Mat& image, const Mat& targetImage, int generations, int populationSize) {
	vector<GeneNode*> population = initializePopulation(populationSize);
	vector<float> fitnessScores(populationSize);
	int bestIndex = 0;
	for (int gen = 0; gen < generations; gen++) {
		for (int i = 0; i < populationSize; i++) {
			fitnessScores[i] = evaluateGeneTree(image, population[i], targetImage);
		}
		bestIndex = max_element(fitnessScores.begin(), fitnessScores.end()) - fitnessScores.begin();
		GeneNode* bestIndividual = population[bestIndex];
		cout << "Generation " << gen + 1 << ": Best F-value = " << fitnessScores[bestIndex]
			<< ", Filter Type: " << (bestIndividual->param == 1 ? "Median" : "Average")
			<< ", Kernel Size: " << bestIndividual->left->param
			<< ", Threshold: " << bestIndividual->right->param << endl;

		if (gen != generations - 1) {
			GeneNode* child = crossover(bestIndividual, population[rand() % populationSize]);
			mutate(child);
			population[rand() % populationSize] = child;
		}
		//GeneNode* child = crossover(bestIndividual, population[rand() % populationSize]);
		//mutate(child);
		//population[rand() % populationSize] = child;
	}
	return population[bestIndex];
}
int main() {
	srand(static_cast<unsigned int>(time(0)));
	Mat image = imread("./imgs_0326_2025_v1/input/oriImg_01.png", IMREAD_GRAYSCALE);
	Mat targetImage = imread("./imgs_0326_2025_v1/input/tarImg_01.png", IMREAD_GRAYSCALE);
	GeneNode* bestSolution = geneticAlgorithm(image, targetImage, 100, 20);
	Mat processedImage = applyGeneTree(image, bestSolution, 1);
	imshow("Best Solution", processedImage);
	waitKey(0);
	return 0;
}