#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define sysRunTimes 1
#define numSets 8 // the num of sets(pairs)
#define idSet 1 // for mark the selected set if the numSets been set of 1
#define POP_SIZE 200 // 100 -> 200
#define GENERATIONS 10000
#define OFFSPRING_COUNT 32 // 16 -> 32
#define MUTATION_RATE 0.9
#define NUM_TYPE_FUNC 19
#define MAX_DEPTH 12 // { 0, 1, 2, ... }, 10 -> 12

void imgShow(const string& name, const Mat& img);
void multiProcess(Mat imgArr[][2]);

enum FilterType { // type-terminal and type-function
	TERMINAL_INPUT,
	GAUSSIAN_BLUR,
	MED_BLUR,
	BLUR,
	BILATERAL_FILTER,
	SOBEL_X,
	SOBEL_Y,
	CANNY,
	DIFF_PROCESS,
	THRESHOLD_9,
	THRESHOLD_31,
	THRESHOLD_63,
	THRESHOLD_127,
	ERODE,
	DILATE,
	CON_PRO_SINGLE_TIME,
	BITWISE_AND,
	BITWISE_OR,
	BITWISE_NOT,
	BITWISE_XOR,
};


struct TreeNode {
	FilterType type;
	vector<shared_ptr<TreeNode>> children; // Array of Children
};

struct genType {
	shared_ptr<TreeNode> eliteTree;
	double eliteFValue;
	double genMinFValue;
	double genAveFValue;
	double genDevFValue;
};

// for storing the f-value of every individual in the group
double indFValInfo[POP_SIZE][numSets + 1];
int curMaxFvalIdx = 0;
double curThreshFVal = 3.00;

int main(void) {
	Mat imgArr[numSets][2]; // imgArr -> storing all images numSets(numSets pairs) * 2(ori, tar)
	char inputPathName_ori[256];
	char inputPathName_tar[256];

	if (numSets == 1) {
		sprintf_s(inputPathName_ori, "./imgs_0820_2025_v1/input/oriImg_0%d.png", idSet);
		sprintf_s(inputPathName_tar, "./imgs_0820_2025_v1/input/tarImg_0%d.png", idSet);
		for (int j = 0; j < 2; j++) {
			if (j == 0) {
				imgArr[0][j] = imread(inputPathName_ori, 0);
			}
			else {
				imgArr[0][j] = imread(inputPathName_tar, 0);
			}
		}
	}
	else {
		for (int i = 0; i < numSets; i++) {
			sprintf_s(inputPathName_ori, "./imgs_0820_2025_v1/input/oriImg_0%d.png", i + 1);
			sprintf_s(inputPathName_tar, "./imgs_0820_2025_v1/input/tarImg_0%d.png", i + 1);
			for (int j = 0; j < 2; j++) {
				if (j == 0) {
					imgArr[i][j] = imread(inputPathName_ori, 0);
				}
				else {
					imgArr[i][j] = imread(inputPathName_tar, 0);
				}
			}
		}
	}

	multiProcess(imgArr);
	return 0;
}

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

random_device rd;
mt19937 rng(rd());
uniform_real_distribution<> prob(0.0, 1.0);

shared_ptr<TreeNode> generateRandomTree(int depth = 0, int maxDepth = MAX_DEPTH) {
	if (depth >= maxDepth || prob(rng) < 0.1) {
		return make_shared<TreeNode>(TreeNode{ TERMINAL_INPUT, {} });
	}

	FilterType t = static_cast<FilterType>(1 + (rng() % NUM_TYPE_FUNC));
	auto node = make_shared<TreeNode>(TreeNode{ t, {} });

	int numChildren = 1;

	if (t == DIFF_PROCESS || t == BITWISE_AND || t == BITWISE_OR || t == BITWISE_XOR) numChildren = 2;

	for (int i = 0; i < numChildren; ++i) {
		// only the process of diff, with 2 inputs
		// the inputs of diff: (1) random inp, (2) TERMINAL_INPUT
		if (i == 0) {
			node->children.push_back(generateRandomTree(depth + 1, maxDepth));
		}
		else {
			// node->children.push_back(make_shared<TreeNode>(TreeNode{ TERMINAL_INPUT, {} }));
			node->children.push_back(generateRandomTree(depth + 1, maxDepth));
		}
	}
	return node;
}


shared_ptr<TreeNode> cloneTree(const shared_ptr<TreeNode>& node) {
	if (!node) return nullptr;
	auto newNode = make_shared<TreeNode>(TreeNode{ node->type, {} });
	for (auto& child : node->children) {
		newNode->children.push_back(cloneTree(child));
	}
	return newNode;
}

Mat gauBlurFunc(const Mat& img) {
	Mat out;
	GaussianBlur(img, out, Size(5, 5), 1.5);
	return out;
}

Mat medBlurFunc(const Mat& img) {
	Mat out;
	medianBlur(img, out, 19);
	return out;
}

Mat blurFunc(const Mat& img) {
	Mat out;
	blur(img, out, Size(19, 19));
	return out;
}

Mat biLateralFilterFunc(const Mat& img) {
	Mat out;
	bilateralFilter(img, out, 9, 75, 75);
	return out;
}

Mat sobelXFunc(const Mat& img) {
	Mat out;
	Sobel(img, out, CV_8U, 1, 0, 3);
	return out;
}

Mat sobelYFunc(const Mat& img) {
	Mat out;
	Sobel(img, out, CV_8U, 0, 1, 3);
	return out;
}

Mat cannyFunc(const Mat& img) {
	Mat out;
	Canny(img, out, 100, 200);
	return out;
}

Mat diffProcess(const Mat postImg, const Mat preImg) {
	int absoluteFlag = 0;
	Mat resImg = Mat::zeros(Size(postImg.cols, postImg.rows), CV_8UC1);
	for (int j = 0; j < postImg.rows; j++)
	{
		for (int i = 0; i < postImg.cols; i++) {
			int diffVal = postImg.at<uchar>(j, i) - preImg.at<uchar>(j, i);
			if (diffVal < 0) {
				if (absoluteFlag != 0) {
					diffVal = abs(diffVal);
				}
				else {
					diffVal = 0;
				}
			}
			resImg.at<uchar>(j, i) = diffVal;
		}
	}
	return resImg;
}

Mat threshFunc_9(const Mat& img) {
	Mat out;
	threshold(img, out, 9, 255, THRESH_BINARY);
	return out;
}

Mat threshFunc_31(const Mat& img) {
	Mat out;
	threshold(img, out, 31, 255, THRESH_BINARY);
	return out;
}

Mat threshFunc_63(const Mat& img) {
	Mat out;
	threshold(img, out, 63, 255, THRESH_BINARY);
	return out;
}

Mat threshFunc_127(const Mat& img) {
	Mat out;
	threshold(img, out, 127, 255, THRESH_BINARY);
	return out;
}

Mat erodeFunc(const Mat& img) {
	Mat out;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	erode(img, out, kernel);
	return out;
}

Mat dilateFunc(const Mat& img) {
	Mat out;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(img, out, kernel);
	return out;
}

Mat conPro_singleTime(const Mat& img) {
	Mat out = Mat::zeros(img.size(), CV_8UC1);
	Mat maskImg = img.clone();
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	for (int idxET = 0; idxET < 3; idxET++) {
		erode(maskImg, maskImg, kernel);
	}
	vector<vector<Point>> contours;
	findContours(maskImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(maskImg.size(), CV_8UC1);
	for (const auto& contour : contours) {
		Rect bounding_box = boundingRect(contour);
		double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
		if ((aspect_ratio <= (1 - 1 * 0.1) || aspect_ratio > (1 + 1 * 0.1)) && cv::contourArea(contour) < 100 * 2) {
			drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
		}
	}
	for (int y = 0; y < out.rows; y++) {
		for (int x = 0; x < out.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
				out.at<uchar>(y, x) = 255;
			}
		}
	}
	return out;
}

Mat bitWiseAndFunc(const Mat postImg, const Mat preImg) {
	Mat out;
	bitwise_and(postImg, preImg, out);
	return out;
}

Mat bitWiseOrFunc(const Mat postImg, const Mat preImg) {
	Mat out;
	bitwise_or(postImg, preImg, out);
	return out;
}

Mat bitWiseNotFunc(const Mat& img) {
	Mat out;
	bitwise_not(img, out);
	return out;
}

Mat bitWiseXorFunc(const Mat postImg, const Mat preImg) {
	Mat out;
	bitwise_xor(postImg, preImg, out);
	return out;
}

Mat executeTree(const shared_ptr<TreeNode>& node, Mat& input) { // ind-tree, img
	switch (node->type) {
	case TERMINAL_INPUT:
		return input.clone();
	case GAUSSIAN_BLUR:
		return gauBlurFunc(executeTree(node->children[0], input));
	case MED_BLUR:
		return medBlurFunc(executeTree(node->children[0], input));
	case BLUR:
		return blurFunc(executeTree(node->children[0], input));
	case BILATERAL_FILTER:
		return biLateralFilterFunc(executeTree(node->children[0], input));
	case SOBEL_X:
		return sobelXFunc(executeTree(node->children[0], input));
	case SOBEL_Y:
		return sobelYFunc(executeTree(node->children[0], input));
	case CANNY:
		return cannyFunc(executeTree(node->children[0], input));
	case DIFF_PROCESS:
		return diffProcess(executeTree(node->children[0], input), executeTree(node->children[1], input));
	case THRESHOLD_9:
		return threshFunc_9(executeTree(node->children[0], input));
	case THRESHOLD_31:
		return threshFunc_31(executeTree(node->children[0], input));
	case THRESHOLD_63:
		return threshFunc_63(executeTree(node->children[0], input));
	case THRESHOLD_127:
		return threshFunc_127(executeTree(node->children[0], input));
	case ERODE:
		return erodeFunc(executeTree(node->children[0], input));
	case DILATE:
		return dilateFunc(executeTree(node->children[0], input));
	case CON_PRO_SINGLE_TIME:
		return conPro_singleTime(executeTree(node->children[0], input));
	case BITWISE_AND:
		return bitWiseAndFunc(executeTree(node->children[0], input), executeTree(node->children[1], input));
	case BITWISE_OR:
		return bitWiseOrFunc(executeTree(node->children[0], input), executeTree(node->children[1], input));
	case BITWISE_NOT:
		return bitWiseNotFunc(executeTree(node->children[0], input));
	case BITWISE_XOR:
		return bitWiseXorFunc(executeTree(node->children[0], input), executeTree(node->children[1], input));
	default:
		return input;
	}
}

using NodeWithParent = pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>>;

void collectNodesWithParents(const shared_ptr<TreeNode>& node,
	const shared_ptr<TreeNode>& parent,
	vector<NodeWithParent>& result) {
	if (!node) return;
	result.emplace_back(node, parent);
	for (auto& child : node->children) {
		collectNodesWithParents(child, node, result);
	}
}

bool isTerminal(FilterType type) {
	return (type == TERMINAL_INPUT);
}

bool isBinaryFilter(FilterType type) {
	return (type == DIFF_PROCESS || type == BITWISE_AND || type == BITWISE_OR || type == BITWISE_XOR);
}

int getTreeMaxDepth(const std::shared_ptr<TreeNode>& node, int depth = 0) {
	if (!node) return depth;
	if (node->children.empty()) return depth;
	int maxChildDepth = depth;
	for (auto& child : node->children) {
		maxChildDepth = std::max(maxChildDepth, getTreeMaxDepth(child, depth + 1));
	}
	return maxChildDepth;
}

/*
	Function:
	(1) Adjust the children for type
	(2) Limit the max-depth of the tree
*/
void adjustChildrenForType(shared_ptr<TreeNode>& node, int currentDepth, int maxDepth, int overFlag = 0) {
	int remainingDepth = maxDepth - currentDepth;
	if (isTerminal(node->type)) {
		node->children.clear();
	}
	else {
		// if the remainingDepth go down to 1, the child node of current node can only been set to the terminal node
		if (remainingDepth <= 1) {
			// 01 - cleanning the children array
			node->children.clear();
			// 02 - set push back times and judge the type of the node, if isBinaryFilter, change the times
			int cntPushBack = 1;
			if (isBinaryFilter(node->type)) {
				cntPushBack = 2;
			}
			// 03 - push back with the TERMINAL_INPUT
			for (int idx = 0; idx < cntPushBack; idx++) {
				node->children.push_back(make_shared<TreeNode>(TreeNode{ TERMINAL_INPUT, {} }));
			}
		}
		else {
			int requiredChildren = isBinaryFilter(node->type) ? 2 : 1;
			// current tree with over-depth, needs to cut down the branch
			if (overFlag) node->children.clear();
			while ((int)node->children.size() < requiredChildren) {
				node->children.push_back(generateRandomTree(currentDepth + 1, maxDepth));
			}
			while ((int)node->children.size() > requiredChildren) {
				node->children.pop_back();
			}
		}
	}
}

void confirmDepth(shared_ptr<TreeNode>& root, int maxDepth = MAX_DEPTH) {
	int finalDepth = getTreeMaxDepth(root);
	if (finalDepth > maxDepth) {
		adjustChildrenForType(root, 0, maxDepth, 1);
	}
}

void crossover(shared_ptr<TreeNode>& a, shared_ptr<TreeNode>& b) {
	vector<NodeWithParent> nodesA, nodesB;
	collectNodesWithParents(a, nullptr, nodesA);
	collectNodesWithParents(b, nullptr, nodesB);

	vector<NodeWithParent> validA, validB;
	for (const auto& np : nodesA) {
		if (np.second) validA.push_back(np);
	}
	for (const auto& np : nodesB) {
		if (np.second) validB.push_back(np);
	}
	// Crossover only works on the situation: except root node, any child node exists.
	if (validA.empty() || validB.empty()) return;

	int idxA = rng() % validA.size();
	int idxB = rng() % validB.size();

	shared_ptr<TreeNode> nodeA = validA[idxA].first;
	shared_ptr<TreeNode> parentA = validA[idxA].second;

	shared_ptr<TreeNode> nodeB = validB[idxB].first;
	shared_ptr<TreeNode> parentB = validB[idxB].second;

	auto& childrenA = parentA->children;
	auto itA = find(childrenA.begin(), childrenA.end(), nodeA);

	auto& childrenB = parentB->children;
	auto itB = find(childrenB.begin(), childrenB.end(), nodeB);

	if (itA != childrenA.end() && itB != childrenB.end()) {
		swap(*itA, *itB);
	}

	confirmDepth(a);
	confirmDepth(b);
}

void mutate(std::shared_ptr<TreeNode>& root, int maxDepth = MAX_DEPTH) {
	using std::shared_ptr;
	using std::make_shared;

	std::vector<NodeWithParent> nodesRoot;
	collectNodesWithParents(root, nullptr, nodesRoot);
	if (nodesRoot.empty()) return;

	const size_t pick = rng() % nodesRoot.size();
	auto& target = nodesRoot[pick].first;
	auto& targetParent = nodesRoot[pick].second;

	int idxTargetInParent = -1;
	int currentDepth = 0;

	// target node is not the root node
	// 01 - get the idx of target in targetParent
	// 02 - get the depth of the target in the root(tree) -> currentDepth >= 1
	if (targetParent) {
		for (size_t i = 0; i < targetParent->children.size(); ++i) {
			if (targetParent->children[i] == target) {
				idxTargetInParent = static_cast<int>(i);
				break;
			}
		}
		if (idxTargetInParent == -1) return; // error situation

		std::function<int(shared_ptr<TreeNode>, int)> findDepth = [&](shared_ptr<TreeNode> node, int depth) -> int {
			if (node == target) return depth;
			for (auto& child : node->children) {
				int d = findDepth(child, depth + 1);
				if (d != -1) return d;
			}
			return -1;
			};
		currentDepth = findDepth(root, 0);
	}

	auto replaceInParent = [&](const shared_ptr<TreeNode>& repl) {
		if (!targetParent) {
			root = repl;
		}
		else {
			targetParent->children[static_cast<size_t>(idxTargetInParent)] = repl;
		}
		};

	int mutationType = rng() % 3;

	switch (mutationType) {
	case 0: { // type - modify
		FilterType newType = static_cast<FilterType>(rng() % (NUM_TYPE_FUNC + 1));
		target->type = newType;
		adjustChildrenForType(target, currentDepth, maxDepth);
		break;
	}
	case 1: { // type - insert
		int remainingDepth = maxDepth - currentDepth;
		if (remainingDepth <= 1) break;

		auto newNode = make_shared<TreeNode>();
		newNode->type = static_cast<FilterType>(1 + (rng() % NUM_TYPE_FUNC));

		if (isBinaryFilter(newNode->type)) {
			newNode->children.push_back(generateRandomTree(currentDepth + 1, maxDepth));
			newNode->children.push_back(target);
		}
		else {
			newNode->children.push_back(target);
		}
		replaceInParent(newNode);
		break;
	}
	case 2: { // type - delete
		if (!isTerminal(target->type) && !target->children.empty()) {
			replaceInParent(target->children[0]);
		}
		break;
	}
	}
	// make sure the depth is within [0, maxDepth]
	confirmDepth(root);
}

double calculateF1Score(double precision, double recall) {
	if (precision + recall == 0) return 0.0;
	return 2.0 * (precision * recall) / (precision + recall);
}

// for calculating the fValue of the ind and writting the organized info into group-arr and groupDvInfoArr
double calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], int numInd) {
	double f1_score[numSets];
	for (int idxSet = 0; idxSet < numSets; idxSet++) {
		int tp = 0, fp = 0, fn = 0;
		for (int i = 0; i < metaImg_g[idxSet].rows; i++) {
			for (int j = 0; j < metaImg_g[idxSet].cols; j++) {
				// if the metaImg processed without threshFunc, then return min-score directly.
				if (metaImg_g[idxSet].at<uchar>(i, j) != 0 && metaImg_g[idxSet].at<uchar>(i, j) != 255) {
					return 0.01;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 0) {
					tp += 1;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 255) {
					fp += 1;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 255 && tarImg_g[idxSet].at<uchar>(i, j) == 0) {
					fn += 1;
				}
			}
		}
		if (tp == 0) tp += 1;
		if (fp == 0) fp += 1;
		if (fn == 0) fn += 1;
		double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
		double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
		f1_score[idxSet] = calculateF1Score(precision, recall);
	}
	double sum_f1 = 0.0;

	for (int idxSet = 0; idxSet < numSets; idxSet++) {
		if (numInd != -1) { // in the last generation
			indFValInfo[numInd][idxSet] = f1_score[idxSet];
		}
		sum_f1 += f1_score[idxSet];
	}

	if (numInd != -1) {
		indFValInfo[numInd][numSets] = sum_f1;
	}

	return sum_f1;
}

double calScoreByInd(const shared_ptr<TreeNode>& node, Mat imgArr[][2], int numInd) {

	Mat tarImg[numSets];
	Mat resImg[numSets];

	for (int i = 0; i < numSets; i++) {
		tarImg[i] = imgArr[i][1];
	}

	for (int i = 0; i < numSets; i++) {
		resImg[i] = executeTree(node, imgArr[i][0]);
	}

	return calculateMetrics(resImg, tarImg, numInd);
}

genType getCurGenInfo(vector<shared_ptr<TreeNode>>& population, Mat imgArr[][2]) {

	int i = 0, j = 0;
	double firstScore = calScoreByInd(population[0], imgArr, -1);
	double minFValue = firstScore;
	double maxFValue = firstScore;
	double aveFValue = 0.0;
	double deviation = 0.0;
	double variance = 0.0;
	double sumFValue = 0.0;

	double scoreArr[POP_SIZE];
	double tempFValue = 0.0;

	genType curGenInfo;

	for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
		scoreArr[idxInd] = calScoreByInd(population[idxInd], imgArr, -1);
	}

	// for getting maxFValue, curMaxFvalIdx, minFValue, sumFValue in cur generation
	for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
		tempFValue = scoreArr[idxInd];
		sumFValue += tempFValue;
		if (tempFValue > maxFValue) {
			maxFValue = tempFValue;
			curMaxFvalIdx = idxInd;
		}
		if (tempFValue < minFValue) {
			minFValue = tempFValue;
		}
	}

	curGenInfo.eliteTree = cloneTree(population[curMaxFvalIdx]);
	curGenInfo.eliteFValue = maxFValue;
	aveFValue = sumFValue / POP_SIZE;
	curGenInfo.genMinFValue = minFValue;
	curGenInfo.genAveFValue = aveFValue;
	for (int idxInd = 0; idxInd < POP_SIZE; idxInd++)
	{
		double diff = scoreArr[idxInd] - aveFValue;
		variance += diff * diff;
	}
	deviation = sqrt(variance / POP_SIZE);
	curGenInfo.genDevFValue = deviation;

	return curGenInfo;
}

string filterTypeToString(FilterType type) {
	switch (type) {
	case TERMINAL_INPUT:     return "TERMINAL_INPUT";
	case GAUSSIAN_BLUR:      return "GAUSSIAN_BLUR";
	case MED_BLUR:           return "MED_BLUR";
	case BLUR:               return "BLUR";
	case BILATERAL_FILTER:   return "BILATERAL_FILTER";
	case SOBEL_X:            return "SOBEL_X";
	case SOBEL_Y:            return "SOBEL_Y";
	case CANNY:              return "CANNY";
	case DIFF_PROCESS:       return "DIFF_PROCESS";
	case THRESHOLD_9:        return "THRESHOLD_9";
	case THRESHOLD_31:       return "THRESHOLD_31";
	case THRESHOLD_63:       return "THRESHOLD_63";
	case THRESHOLD_127:      return "THRESHOLD_127";
	case ERODE:              return "ERODE";
	case DILATE:             return "DILATE";
	case CON_PRO_SINGLE_TIME:return "CON_PRO_SINGLE_TIME";
	case BITWISE_AND:        return "BITWISE_AND";
	case BITWISE_OR:         return "BITWISE_OR";
	case BITWISE_NOT:        return "BITWISE_NOT";
	case BITWISE_XOR:        return "BITWISE_XOR";
	default:                 return "UNKNOWN";
	}
}

void printTree(const shared_ptr<TreeNode>& node, int depth = 0, FILE* fl_printTree = stdout) {
	if (!node) return;
	for (int i = 0; i < depth; ++i) {
		fprintf(fl_printTree, "    ");
	}
	fprintf(fl_printTree, "%s\n", filterTypeToString(node->type).c_str());
	for (const auto& child : node->children) {
		printTree(child, depth + 1, fl_printTree);
	}
}

void multiProcess(Mat imgArr[][2]) {
	Mat resImg[numSets];
	Mat tarImg[numSets];

	char imgName_pro[numSets][256];
	char folderPath[numSets][256];
	char imgName_final[numSets][256];

	// for recording the f_value of every generation (max, min, ave, dev)
	FILE* fl_fValue = nullptr;
	errno_t err = fopen_s(&fl_fValue, "./imgs_0820_2025_v1/output/f_value.txt", "w");
	if (err != 0 || fl_fValue == nullptr) {
		perror("Cannot open the file");
		return;
	}

	// for recording the f_value of elite-ind in last gen (setX1, setX2, ..., Max)
	FILE* fl_maxFval = nullptr;
	errno_t err2 = fopen_s(&fl_maxFval, "./imgs_0820_2025_v1/output/maxFvalInfo_final.txt", "w");
	if (err2 != 0 || fl_maxFval == nullptr) {
		perror("Cannot open the file");
		return;
	}

	FILE* fl_printTree = nullptr;
	errno_t err3 = fopen_s(&fl_printTree, "./imgs_0820_2025_v1/output/printed_tree.txt", "w");
	if (err3 != 0 || fl_printTree == nullptr) {
		perror("Cannot open the file");
		return;
	}

	for (int idxProTimes = 0; idxProTimes < sysRunTimes; idxProTimes++) {
		vector<genType> genInfo;

		vector<shared_ptr<TreeNode>> population;
		for (int i = 0; i < POP_SIZE; ++i) {
			population.push_back(generateRandomTree());
		}

		for (int numGen = 0; numGen < GENERATIONS; numGen++) {
			cout << "---------idxProTimes: " << idxProTimes + 1 << ", generation: " << numGen + 1 << "---------" << endl;
			int idx1 = rng() % POP_SIZE;
			int idx2 = rng() % POP_SIZE;
			while (idx2 == idx1) idx2 = rng() % POP_SIZE;

			auto parent1 = cloneTree(population[idx1]);
			auto parent2 = cloneTree(population[idx2]);

			vector<pair<double, shared_ptr<TreeNode>>> family;

			double score1 = calScoreByInd(parent1, imgArr, -1);
			double score2 = calScoreByInd(parent2, imgArr, -1);

			family.push_back({ score1, parent1 });
			family.push_back({ score2, parent2 });

			for (int k = 0; k < OFFSPRING_COUNT; ++k) {
				auto childA = cloneTree(parent1);
				auto childB = cloneTree(parent2);
				crossover(childA, childB);
				auto chosen = (prob(rng) < 0.5) ? childA : childB;
				double fit = calScoreByInd(chosen, imgArr, -1);
				family.push_back({ fit, chosen });
			}

			for (int idxInd = 0; idxInd < (OFFSPRING_COUNT + 2); idxInd++) {
				if (prob(rng) < MUTATION_RATE) {
					mutate(family[idxInd].second);
					family[idxInd].first = calScoreByInd(family[idxInd].second, imgArr, -1);
				}
			}

			sort(family.rbegin(), family.rend()); // descending sort by f1_score(ind.first)
			auto elite = family[0];
			double total = 0;
			for (const auto& f : family) total += f.first;
			double r = prob(rng) * total, accum = 0;
			shared_ptr<TreeNode> rouletteSelected = family[1].second; // fallback
			double scoreRouletteSelected = 0.01;
			for (const auto& f : family) {
				accum += f.first;
				if (accum >= r) {
					rouletteSelected = f.second;
					scoreRouletteSelected = f.first;
					break;
				}
			}

			// when elite.first <= score1 && scoreRouletteSecelected <= score2
			// make sure the ind with max-score in last generation would not been replaced by elite and roulette.
			if (elite.first > score1) {
				population[idx1] = cloneTree(elite.second);
			}

			if (scoreRouletteSelected > score2) {
				population[idx2] = cloneTree(rouletteSelected);
			}

			genInfo.push_back(getCurGenInfo(population, imgArr));
			printf("the score of elite(%d gen): %.4f\n", numGen + 1, genInfo[numGen].eliteFValue);

			if (numGen == GENERATIONS - 1) { // if reached the last gen, then write the info-score of the population into the indFValInfo
				for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
					calScoreByInd(population[idxInd], imgArr, idxInd);
				}
			}
		}
		printf("---------------- GEN-END --------------\n");

		if (indFValInfo[curMaxFvalIdx][numSets] > curThreshFVal) {
			curThreshFVal = indFValInfo[curMaxFvalIdx][numSets];
			Mat resImg_02;
			Mat res;

			for (int idxGen = 0; idxGen < GENERATIONS; idxGen++) {
				if ((idxGen + 1) % 1000 == 0) {
					for (int idxSet = 0; idxSet < numSets; idxSet++) {
						resImg_02 = executeTree(genInfo[idxGen].eliteTree, imgArr[idxSet][0]);
						sprintf_s(imgName_pro[idxSet], "./imgs_0820_2025_v1/output/img_0%d/Gen-%d-t%d.png", idxSet + 1, idxGen + 1, idxProTimes + 1);
						imwrite(imgName_pro[idxSet], resImg_02);
						if (idxGen == GENERATIONS - 1) {
							vector<Mat> images = { resImg_02, imgArr[idxSet][1] };
							hconcat(images, res);
							sprintf_s(imgName_final[idxSet], "./imgs_0820_2025_v1/output/img_0%d/imgs_final-t%d.png", idxSet + 1, idxProTimes + 1);
							imwrite(imgName_final[idxSet], res);
						}
					}
				}
			}
			printTree(genInfo[GENERATIONS - 1].eliteTree, 0, fl_printTree);

			for (int i = 0; i < GENERATIONS; i++) {
				fprintf(fl_fValue, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
			}

			for (int i = 0; i <= numSets; i++) {
				fprintf(fl_maxFval, "%.4f ", indFValInfo[curMaxFvalIdx][i]);
			}
			fprintf(fl_maxFval, "\n");
		}

		// to reset the val of curMaxFvalIdx
		curMaxFvalIdx = 0;
	}

	fclose(fl_fValue);
	fclose(fl_maxFval);
	fclose(fl_printTree);
}