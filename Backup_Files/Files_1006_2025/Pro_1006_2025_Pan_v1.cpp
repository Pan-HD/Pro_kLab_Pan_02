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
#define POP_SIZE 10 // Pop_Size of GP
#define GENERATIONS 10 // Generation of GP
#define OFFSPRING_COUNT 16 // OFFSPRING_COUNT of GP
#define MUTATION_RATE 0.9 // GP
#define NUM_TYPE_FUNC 16 // GP
#define MAX_DEPTH 3 // { 0, 1, 2, ... } GP
#define ENABLE_GA true
#define GA_POP 10
#define GA_GENERATIONS 10
#define INITIAL_BIAS_THRESHOLD 0.05
#define BIAS_DECAY 0.9
#define BIAS_WINDOW 5

// =====================================================
// Random utilities
// =====================================================
random_device rd;
mt19937 rng(rd());
uniform_real_distribution<> prob(0.0, 1.0);
uniform_int_distribution<> randint(0, 999999);

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
    THRESHOLD,
    ERODE,
    DILATE,
    CONTOUR_PROCESS,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_NOT,
    BITWISE_XOR,
};

struct ParamDesc { // describle of params
    int n;
    double minv;
    double maxv;
};
unordered_map<FilterType, ParamDesc> g_paramDesc; // <type, desc of params>

void initParamDesc() {
    g_paramDesc[GAUSSIAN_BLUR] = { 2, 1.0, 31.0 };
    g_paramDesc[MED_BLUR] = { 1, 1.0, 31.0 };
    g_paramDesc[BLUR] = { 1, 1.0, 31.0 };
    g_paramDesc[BILATERAL_FILTER] = { 3, 1.0, 150.0 };
    g_paramDesc[SOBEL_X] = { 1, 1.0, 7.0 };
    g_paramDesc[SOBEL_Y] = { 1, 1.0, 7.0 };
    g_paramDesc[CANNY] = { 2, 1.0, 255.0 };
    g_paramDesc[THRESHOLD] = { 1, 0.0, 255.0 };
    g_paramDesc[ERODE] = { 1, 0.0, 5.0 };
    g_paramDesc[DILATE] = { 1, 0.0, 5.0 };
    g_paramDesc[CONTOUR_PROCESS] = { 5, 0.0, 15.0 };
}

struct TreeNode {
    FilterType type;
    vector<shared_ptr<TreeNode>> children;
    vector<double> params;
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

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
}

shared_ptr<TreeNode> cloneTree(const shared_ptr<TreeNode>& node) {
    if (!node) return nullptr;
    auto newNode = make_shared<TreeNode>();
    newNode->type = node->type;
    newNode->params = node->params;
    for (auto& c : node->children) newNode->children.push_back(cloneTree(c));
    return newNode;
}

// =====================================================
// Gray code utilities
// =====================================================

/*
  Turn bin to gray -> Convert a regular integer into an integer that can be interpreted as a Gray code.
  Eg: 3 -> 0011(bin) -> 0010(gray) -> 2
*/
inline int binaryToGray(int num) { return num ^ (num >> 1); }

/*
  Turn gray to bin -> Convert an integer that can be interpreted as a Gray code into a regular integer.
  Eg: 2 -> 0010(gray) -> 0011(bin) -> 3
*/
inline int grayToBinary(int num) {
    for (int mask = num >> 1; mask != 0; mask >>= 1) num ^= mask;
    return num;
}

/*
  Convert the integer n into a binary string of length bits.
  intToBits(5, 8) → "00000101"
*/
inline string intToBits(int n, int bits = 8) {
    string s(bits, '0');
    for (int i = bits - 1; i >= 0; --i) s[i] = (n & 1) + '0', n >>= 1;
    return s;
}

/*
  Convert a binary string of length bits into the integer n
  "00000101" → intToBits(5, 8) → 5
*/
inline int bitsToInt(const string& s) {
    int val = 0;
    for (char c : s) val = (val << 1) | (c - '0');
    return val;
}

/*
  Map a real-valued parameter (e.g., σ = 1.2) to a Gray-coded binary string.
  -> (1.2 - 0.0)/(5.0 - 0.0) = 0.24
  -> 0.24 * 255(2^8 - 1) ≈ 61
  -> binaryToGray(61) → 111101 → 100011(gray) → 35(gray)
  -> intToBits(35, 8) → "00100011"
*/
inline string grayEncode(double val, double minv, double maxv, int bits = 8) {
    double norm = (val - minv) / (maxv - minv);
    norm = max(0.0, min(1.0, norm));
    int bin = int(norm * ((1 << bits) - 1));
    return intToBits(binaryToGray(bin), bits);
}

/*
  Decode the Gray code back into the corresponding real-valued parameter.
  -> grayDecode("00111101", 0.0, 5.0)　→　61(bin)　
  　 →　norm = 61 / 255 ≈ 0.239　→　val = 0 + 0.239*5 = 1.195
*/
inline double grayDecode(const string& gray, double minv, double maxv, int bits = 8) {
    int bin = grayToBinary(bitsToInt(gray));
    double norm = double(bin) / ((1 << bits) - 1);
    return minv + norm * (maxv - minv);
}

/*
  Randomly flip certain bits in the Gray-coded string with a probability of rate (genetic mutation).
  "00110110(gray)" -> "00111110(gray)"
*/
inline string mutateGrayBits(string s, double rate = 0.01) {
    for (auto& c : s)
        if (prob(rng) < rate) c = (c == '0' ? '1' : '0');
    return s;
}

shared_ptr<TreeNode> generateRandomTree(int depth = 0, int maxDepth = MAX_DEPTH) {
    if (depth >= maxDepth || prob(rng) < 0.1) {
        auto t = make_shared<TreeNode>();
        t->type = TERMINAL_INPUT;
        return t;
    }
    FilterType t = static_cast<FilterType>(1 + (rng() % NUM_TYPE_FUNC));
    auto node = make_shared<TreeNode>();
    node->type = t;

    if (g_paramDesc.count(t)) {
        // pd -> { numParam, minVal, maxVal }
        auto pd = g_paramDesc[t];
        node->params.resize(pd.n);
        for (int i = 0; i < pd.n; i++) {
            // Create a random number generator that returns values uniformly distributed over the interval [minv, maxv].
            uniform_real_distribution<> ud(pd.minv, pd.maxv);
            node->params[i] = ud(rng);
        }
    }
    int numChildren = (t == BITWISE_AND || t == BITWISE_OR || t == BITWISE_XOR || t == DIFF_PROCESS) ? 2 : 1;
    for (int i = 0; i < numChildren; i++) node->children.push_back(generateRandomTree(depth + 1, maxDepth));
    return node;
}

int getMapVal(int iptNum) {
    return iptNum;
}

Mat executeTree(shared_ptr<TreeNode> node, const Mat& input) {
    if (!node) return input;
    switch (node->type) {
    case TERMINAL_INPUT:
        return input.clone();
    case GAUSSIAN_BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        GaussianBlur(executeTree(node->children[0], input), dst, Size(k, k), node->params[1]);
        return dst;
    }
    case MED_BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        medianBlur(executeTree(node->children[0], input), dst, k);
        return dst;
    }
    case BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        blur(executeTree(node->children[0], input), dst, Size(k, k));
        return dst;
    }
    case BILATERAL_FILTER: {
        Mat dst;
        bilateralFilter(executeTree(node->children[0], input), dst,
            int(node->params[0]), node->params[1], node->params[2]);
        return dst;
    }
    case SOBEL_X: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        Sobel(executeTree(node->children[0], input), dst, CV_8U, 1, 0, k);
        return dst;
    }
    case SOBEL_Y: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        Sobel(executeTree(node->children[0], input), dst, CV_8U, 0, 1, k);
        return dst;
    }
    case CANNY: {
        Mat dst;
        Canny(executeTree(node->children[0], input), dst, node->params[0], node->params[1]);
        return dst;
    }
    case DIFF_PROCESS: {
        Mat dst = Mat::zeros(Size(input.cols, input.rows), CV_8UC1);
        Mat postImg = executeTree(node->children[0], input);
        Mat preImg = executeTree(node->children[1], input);
        for (int j = 0; j < input.rows; j++)
        {
            for (int i = 0; i < input.cols; i++) {
                // int diffVal = executeTree(node->children[0], input).at<uchar>(j, i) - executeTree(node->children[1], input).at<uchar>(j, i);

                int diffVal = postImg.at<uchar>(j, i) - preImg.at<uchar>(j, i);
                if (diffVal < 0) {
                    diffVal = 0;
                }
                dst.at<uchar>(j, i) = diffVal;
            }
        }
        return dst;
    }
    case THRESHOLD: {
        Mat dst;
        threshold(executeTree(node->children[0], input), dst, node->params[0], 255, THRESH_BINARY);
        return dst;
    }
    case ERODE: {
        Mat dst;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(1 + 2 * int(node->params[0]), 1 + 2 * int(node->params[0])));
        erode(executeTree(node->children[0], input), dst, kernel);
        return dst;
    }
    case DILATE: {
        Mat dst;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(1 + 2 * int(node->params[0]), 1 + 2 * int(node->params[0])));
        dilate(executeTree(node->children[0], input), dst, kernel);
        return dst;
    }

    case CONTOUR_PROCESS: {
        Mat maskImg = executeTree(node->children[0], input).clone();
        // node->params[0]: Kernel-Size of ERODE -> { 1, 3, 5, 7 }
        int k = ((int(node->params[0])) / 2) | 1;
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
        // node->params[1]: Execute-Nums of ERODE FUNC -> { 0, 1, 2, ..., 7 }
        for (int idxET = 0; idxET < (int(node->params[1]) / 2); idxET++) {
            erode(maskImg, maskImg, kernel);
        }
        vector<vector<Point>> contours;
        findContours(maskImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        Mat mask(maskImg.rows, maskImg.cols, CV_8UC1, cv::Scalar(255));

        int selType = 0;
        for (const auto& contour : contours) {
            Rect bounding_box = boundingRect(contour);
            double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
            // node->params[2]: Status of exeType -> { 0, 1, 2 }
            if ((int(node->params[2]) / 5) < 3) {
                selType = int(node->params[2]) / 5;
            }
            else {
                selType = 2;
            }
            if (selType == 0) {
                // node->params[3]: Range of delta-aspectRatio -> { 0, 1, 2, ..., 7 }
                if (aspect_ratio >= (1 - (int(node->params[3]) / 2) * 0.1) && aspect_ratio <= (1 + (int(node->params[3]) / 2) * 0.1)) {
                    drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(0), -1);
                }
            }
            else if (selType == 1) {
                // node->params[4]: Range of delta-square -> { 0, 1, ..., 15 } 
                if (contourArea(contour) >= 100 * int(node->params[4])) {
                    drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(0), -1);
                }
            }
            else if (selType == 2) {
                if ((aspect_ratio >= (1 - int(node->params[3]) * 0.1) && aspect_ratio <= (1 + int(node->params[3]) * 0.1)) && (contourArea(contour) >= 100 * int(node->params[4]))) {
                    drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(0), -1);
                }
            }
        }
        return mask;
    }

    case BITWISE_AND: {
        Mat dst;
        bitwise_and(executeTree(node->children[0], input), executeTree(node->children[1], input), dst);
        return dst;
    }
    case BITWISE_OR: {
        Mat dst;
        bitwise_or(executeTree(node->children[0], input), executeTree(node->children[1], input), dst);
        return dst;
    }
    case BITWISE_NOT: {
        Mat dst;
        bitwise_not(executeTree(node->children[0], input), dst);
        return dst;
    }
    case BITWISE_XOR: {
        Mat dst;
        bitwise_xor(executeTree(node->children[0], input), executeTree(node->children[1], input), dst);
        return dst;
    }
    default:
        return input.clone();
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
            // root = repl;
            root = cloneTree(repl);
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

double calcBias(const vector<genType>& genInfo, int window = BIAS_WINDOW) {
    if (genInfo.size() < window + 1) return 1.0;
    double recentAvg = 0.0;
    for (int i = genInfo.size() - window; i < (int)genInfo.size(); ++i) recentAvg += genInfo[i].eliteFValue;
    recentAvg /= window;
    double totalAvg = 0.0;
    for (int idxGen = 0; idxGen < genInfo.size(); idxGen++) {
        totalAvg += genInfo[idxGen].eliteFValue;
    }
    totalAvg /= genInfo.size();
    double bias = fabs(recentAvg - totalAvg) / (totalAvg + 1e-9);
    return bias;
}

/*
    Function: Collect all the nodes with Params in a note-tree.
*/
void collectParams(shared_ptr<TreeNode> root, vector<shared_ptr<TreeNode>>& out) {
    if (!root) return;
    if (!root->params.empty())
        out.push_back(root);
    for (auto& c : root->children)
        collectParams(c, out);
}

vector<double> runGrayGA_forTree(shared_ptr<TreeNode> rootInGP, Mat imgArr[][2]) {
    Mat oriImg[numSets];
    Mat tarImg[numSets];

    for (int i = 0; i < numSets; i++) {
        oriImg[i] = imgArr[i][0];
    }

    for (int i = 0; i < numSets; i++) {
        tarImg[i] = imgArr[i][1];
    }

    // =====================================================
    // Pre Processing of GA
    // =====================================================
    vector<shared_ptr<TreeNode>> paramNodes;
    collectParams(rootInGP, paramNodes);
    if (paramNodes.empty()) return {};

    // bounds: 
    //    [ <minV, maxV>,  -> paramNodes[0] (with all params of the node)
    //      <minV, maxV>,  -> paramNodes[1] (with all params of the node)
    //      ......
    //      <minV, maxV>,  -> paramNodes[n] (with all params of the node)]
    vector<pair<double, double>> bounds;

    // baseGenes: 
    //    [ strVal,  -> paramNodes[0] (with all params of the node)
    //      strVal,  -> paramNodes[1] (with all params of the node)
    //      ......
    //      strVal,  -> paramNodes[n] (with all params of the node)]
    vector<string> baseGenes;
    for (auto& pNode : paramNodes) {

        // pd: { n, minV, maxV }
        //    pd.n -> nums of params in cur node
        auto pd = g_paramDesc[pNode->type];
        for (int i = 0; i < pd.n; i++) {
            bounds.push_back({ pd.minv, pd.maxv });
            baseGenes.push_back(grayEncode(pNode->params[i], pd.minv, pd.maxv, 8));
        }
    }
    // numsParams: the nums of all params waited to be optimize
    int numsParams = baseGenes.size();

    struct GrayInd {
        vector<string> arrStrVal;
        double fit;
    };
    vector<GrayInd> pop(GA_POP);

    // Initial all inds in GA-POP
    //   Assign for inds.arrStrVal and ind.fit
    for (auto& ind : pop) {
        ind.arrStrVal = baseGenes;
        for (auto& g : ind.arrStrVal) g = mutateGrayBits(g, 0.05);

        // arrDecoded: 
        //    [ realVal,  -> paramNodes[0] (with all params of the node)
        //      realVal,  -> paramNodes[1] (with all params of the node)
        //      ......
        //      realVal,  -> paramNodes[n] (with all params of the node)]
        vector<double> arrDecoded;
        for (int i = 0; i < numsParams; i++) arrDecoded.push_back(grayDecode(ind.arrStrVal[i], bounds[i].first, bounds[i].second));
        auto rootTreeCloned = cloneTree(rootInGP);

        vector<shared_ptr<TreeNode>> arrParamNode;
        collectParams(rootTreeCloned, arrParamNode);

        int pos = 0;
        for (auto& pNode : arrParamNode)
            // pNode->params: the params-arr(double) of the tree-node
            for (auto& p : pNode->params) p = arrDecoded[pos++];
        ind.fit = calScoreByInd(rootTreeCloned, imgArr, -1);

    }

    // =====================================================
    // Officially Processing of GA
    // =====================================================
    for (int gen = 0; gen < GA_GENERATIONS; gen++) {
        sort(pop.begin(), pop.end(), [](auto& a, auto& b) { return a.fit > b.fit; });

        // Preparing Space(Array) for the optimized inds 
        vector<GrayInd> newpop;
        // Elite back
        newpop.push_back(pop[0]);

        while ((int)newpop.size() < GA_POP) {
            int a = rng() % GA_POP, b = rng() % GA_POP;
            GrayInd child = pop[a];
            for (int i = 0; i < numsParams; i++) {
                // Crossover in GA
                if (prob(rng) < 0.5) child.arrStrVal[i] = pop[b].arrStrVal[i];
                // Mutate in GA
                child.arrStrVal[i] = mutateGrayBits(child.arrStrVal[i], 0.01);
            }
            vector<double> arrDecodedChild;
            for (int i = 0; i < numsParams; i++) arrDecodedChild.push_back(grayDecode(child.arrStrVal[i], bounds[i].first, bounds[i].second));

            auto rootCloned = cloneTree(rootInGP);
            vector<shared_ptr<TreeNode>> arrParamNode;
            collectParams(rootCloned, arrParamNode);

            int pos = 0;
            for (auto& pNode : arrParamNode)
                // pNode->params: the params-arr(double) of the tree-node
                for (auto& p : pNode->params) p = arrDecodedChild[pos++];
            child.fit = calScoreByInd(rootCloned, imgArr, -1);
            newpop.push_back(child);
        }
        pop.swap(newpop);
    }

    sort(pop.begin(), pop.end(), [](auto& a, auto& b) { return a.fit > b.fit; });
    vector<double> decoded;
    for (int i = 0; i < numsParams; i++) decoded.push_back(grayDecode(pop.front().arrStrVal[i], bounds[i].first, bounds[i].second));
    return decoded;
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
    case THRESHOLD:          return "THRESHOLD";
    case ERODE:              return "ERODE";
    case DILATE:             return "DILATE";
    case CONTOUR_PROCESS:    return "CONTOUR_PROCESS";
    case BITWISE_AND:        return "BITWISE_AND";
    case BITWISE_OR:         return "BITWISE_OR";
    case BITWISE_NOT:        return "BITWISE_NOT";
    case BITWISE_XOR:        return "BITWISE_XOR";
    default:                 return "UNKNOWN";
    }
}

void printTree(const shared_ptr<TreeNode>& node,
    int depth = 0,
    FILE* fp = stdout) {
    if (!node) return;
    for (int i = 0; i < depth; ++i)
        fprintf(fp, "    ");
    fprintf(fp, "%s", filterTypeToString(node->type).c_str());
    if (!node->params.empty()) {
        fprintf(fp, " (");
        for (size_t i = 0; i < node->params.size(); i++) {
            fprintf(fp, "%.2f", node->params[i]);
            if (i + 1 < node->params.size())
                fprintf(fp, ", ");
        }
        fprintf(fp, ")");
    }
    fprintf(fp, "\n");
    for (const auto& c : node->children)
        printTree(c, depth + 1, fp);
}

void multiProcess(Mat imgArr[][2]) {
    Mat resImg[numSets];
    Mat tarImg[numSets];

    char imgName_pro[numSets][256];
    char folderPath[numSets][256];
    char imgName_final[numSets][256];

    // for recording the f_value of every generation (max, min, ave, dev)
    FILE* fl_fValue = nullptr;
    errno_t err = fopen_s(&fl_fValue, "./imgs_1006_2025_v1/output/f_value.txt", "w");
    if (err != 0 || fl_fValue == nullptr) {
        perror("Cannot open the file");
        return;
    }

    // for recording the f_value of elite-ind in last gen (setX1, setX2, ..., Max)
    FILE* fl_maxFval = nullptr;
    errno_t err2 = fopen_s(&fl_maxFval, "./imgs_1006_2025_v1/output/maxFvalInfo_final.txt", "w");
    if (err2 != 0 || fl_maxFval == nullptr) {
        perror("Cannot open the file");
        return;
    }

    FILE* fl_printTree = nullptr;
    errno_t err3 = fopen_s(&fl_printTree, "./imgs_1006_2025_v1/output/printed_tree.txt", "w");
    if (err3 != 0 || fl_printTree == nullptr) {
        perror("Cannot open the file");
        return;
    }

    initParamDesc();

    for (int idxProTimes = 0; idxProTimes < sysRunTimes; idxProTimes++) {
        vector<genType> genInfo;

        vector<shared_ptr<TreeNode>> population;
        for (int i = 0; i < POP_SIZE; ++i) {
            population.push_back(generateRandomTree());
        }

        // replaced by genInfo[numGen].eliteFValue
        vector<double> eliteHist;
        double biasThreshold = INITIAL_BIAS_THRESHOLD;

        for (int numGen = 0; numGen < GENERATIONS; numGen++) {

            cout << "---------idxProTimes: " << idxProTimes + 1 << ", generation: " << numGen + 1 << "---------" << endl;



            // replaced by indFValInfo[numInd][numSets];
            vector<double> scores(POP_SIZE);

            int idx1 = rng() % POP_SIZE;
            int idx2 = rng() % POP_SIZE;
            while (idx2 == idx1) idx2 = rng() % POP_SIZE;



            auto parent1 = cloneTree(population[idx1]);
            auto parent2 = cloneTree(population[idx2]);

            vector<pair<double, shared_ptr<TreeNode>>> family;


            cout << "ENTER-01" << endl;
            double score1 = calScoreByInd(parent1, imgArr, -1);
            cout << "ENTER-02" << endl;
            double score2 = calScoreByInd(parent2, imgArr, -1);
            cout << "ENTER-03" << endl;


            family.push_back({ score1, parent1 });
            family.push_back({ score2, parent2 });

            cout << "ENTER-04" << endl;

            for (int k = 0; k < OFFSPRING_COUNT; ++k) {
                auto childA = cloneTree(parent1);
                auto childB = cloneTree(parent2);
                crossover(childA, childB);
                auto chosen = (prob(rng) < 0.5) ? childA : childB;
                double fit = calScoreByInd(chosen, imgArr, -1);
                family.push_back({ fit, chosen });
            }

            cout << "ENTER-05" << endl;

            for (int idxInd = 0; idxInd < (OFFSPRING_COUNT + 2); idxInd++) {
                if (prob(rng) < MUTATION_RATE) {
                    mutate(family[idxInd].second);
                    family[idxInd].first = calScoreByInd(family[idxInd].second, imgArr, -1);
                }
            }

            cout << "ENTER-06" << endl;

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
            double bias = calcBias(genInfo);
            printf("the score of elite(%d gen): %.4f, bias: %.2f\n", numGen + 1, genInfo[numGen].eliteFValue, bias);

            if (numGen != GENERATIONS - 1) {
                // ---- Trigger GA when Bias low ----
                if (ENABLE_GA && bias < biasThreshold) {
                    cout << "[PT-ACTIT] Trigger GA Phase (Bias=" << bias << ")" << endl;
                    auto best = runGrayGA_forTree(population[idx1], imgArr);
                    if (!best.empty()) {
                        vector<shared_ptr<TreeNode>> pn;
                        collectParams(population[idx1], pn);
                        int pos = 0;
                        for (auto& n : pn)
                            for (auto& p : n->params) p = best[pos++];
                    }
                    biasThreshold *= BIAS_DECAY;
                }
            }
            else {
                // if reached the last gen, then write the info-score of the population into the indFValInfo
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
                        sprintf_s(imgName_pro[idxSet], "./imgs_1006_2025_v1/output/img_0%d/Gen-%d-t%d.png", idxSet + 1, idxGen + 1, idxProTimes + 1);
                        imwrite(imgName_pro[idxSet], resImg_02);
                        if (idxGen == GENERATIONS - 1) {
                            vector<Mat> images = { resImg_02, imgArr[idxSet][1] };
                            hconcat(images, res);
                            sprintf_s(imgName_final[idxSet], "./imgs_1006_2025_v1/output/img_0%d/imgs_final-t%d.png", idxSet + 1, idxProTimes + 1);
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

int main(void) {
    Mat imgArr[numSets][2]; // imgArr -> storing all images numSets(numSets pairs) * 2(ori, tar)
    char inputPathName_ori[256];
    char inputPathName_tar[256];

    if (numSets == 1) {
        sprintf_s(inputPathName_ori, "./imgs_1006_2025_v1/input/oriImg_0%d.png", idSet);
        sprintf_s(inputPathName_tar, "./imgs_1006_2025_v1/input/tarImg_0%d.png", idSet);
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
            sprintf_s(inputPathName_ori, "./imgs_1006_2025_v1/input/oriImg_0%d.png", i + 1);
            sprintf_s(inputPathName_tar, "./imgs_1006_2025_v1/input/tarImg_0%d.png", i + 1);
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
