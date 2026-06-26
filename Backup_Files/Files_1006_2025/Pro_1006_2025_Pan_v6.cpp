#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define sysRunTimes 1
#define numSets 8 // the num of sets(pairs)
#define idSet 1 // for mark the selected set if the numSets been set of 1

// GP parameters
#define POP_SIZE 150 // Pop_Size of GP
#define GENERATIONS 15000 // Generation of GP
#define OFFSPRING_COUNT 20 // OFFSPRING_COUNT of GP
#define MUTATION_RATE 0.9 // GP
#define NUM_TYPE_FUNC 16 // GP
#define MAX_DEPTH 12 // { 0, 1, 2, ... } GP

// GA parameters
// #define ENABLE_GA true
#define GA_TRIGGER_THRESH 2.2
#define GA_POP 30
#define GA_GENERATIONS 100
#define INITIAL_BIAS_THRESHOLD 0.23
#define BIAS_DECAY 0.99
#define BIAS_WINDOW 5

// =====================================================
// Random utilities (unified)
// =====================================================
static std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
static std::uniform_real_distribution<double> uni_real(0.0, 1.0);

inline int rand_int(int a, int b) { // inclusive
    std::uniform_int_distribution<int> d(a, b);
    return d(rng);
}
inline double rand_real() {
    return uni_real(rng);
}

// convenience small wrappers used in old code
#define prob_rand() (rand_real())

// =====================================================
// Types and param descriptions
// =====================================================
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
static unordered_map<FilterType, ParamDesc> g_paramDesc; // <type, desc of params>

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

static unordered_map<FilterType, vector<double>> g_paramDesc_safeVal; // <type, desc of params(array)>
void initParamDesc_safeVal() {
    g_paramDesc_safeVal[GAUSSIAN_BLUR].push_back(5.0); // kernel size
    g_paramDesc_safeVal[GAUSSIAN_BLUR].push_back(1.5); // sigma

    g_paramDesc_safeVal[MED_BLUR].push_back(19.0); // kernel size

    g_paramDesc_safeVal[BLUR].push_back(19.0); // kernel size

    g_paramDesc_safeVal[BILATERAL_FILTER].push_back(9.0); // d 
    g_paramDesc_safeVal[BILATERAL_FILTER].push_back(75.0); // sigmaColor
    g_paramDesc_safeVal[BILATERAL_FILTER].push_back(75.0); // sigmaSpace

    g_paramDesc_safeVal[SOBEL_X].push_back(3.0); // kernel size
    g_paramDesc_safeVal[SOBEL_Y].push_back(3.0); // kernel size

    g_paramDesc_safeVal[CANNY].push_back(100.0); // t1
    g_paramDesc_safeVal[CANNY].push_back(200.0); // t2

    g_paramDesc_safeVal[THRESHOLD].push_back(9.0); // threshVal

    g_paramDesc_safeVal[ERODE].push_back(1.0); // kernel size ( val = param * 2 + 1 )
    g_paramDesc_safeVal[DILATE].push_back(1.0); // kernel size ( val = param * 2 + 1 )

    g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(6.0); // kernel size ( val = param / 2 )
    g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(6.0); // erode・dilate times ( val = param / 2 )
    g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(12.0); // selType ( val = param / 5 )
    g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(2.0); // range ( val = param / 2 )
    g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(2.0); // areaTh
}

// =====================================================
// Tree node
// =====================================================
struct TreeNode {
    FilterType type = TERMINAL_INPUT;
    vector<shared_ptr<TreeNode>> children;
    vector<double> params;
};

// =====================================================
// getCurGenInfo (unchanged logic but uses calScoreByInd)
// =====================================================
struct genType {
    shared_ptr<TreeNode> eliteTree;
    double eliteFValue;
    double genMinFValue;
    double genAveFValue;
    double genDevFValue;
};

double indFValInfo[POP_SIZE][numSets + 1];
int curMaxFvalIdx = 0;
double curThreshFVal = 3.00;
// bool assignTriggerFlag = false;

// clone
shared_ptr<TreeNode> cloneTree(const shared_ptr<TreeNode>& node) {
    if (!node) return nullptr;
    auto newNode = make_shared<TreeNode>();
    newNode->type = node->type;
    newNode->params = node->params;
    for (auto& c : node->children) newNode->children.push_back(cloneTree(c));
    return newNode;
}

// =====================================================
// Gray code utilities (unchanged logic)
// =====================================================
inline int binaryToGray(int num) { return num ^ (num >> 1); }
inline int grayToBinary(int num) { for (int mask = num >> 1; mask != 0; mask >>= 1) num ^= mask; return num; }

inline string intToBits(int n, int bits = 8) {
    string s(bits, '0');
    for (int i = bits - 1; i >= 0; --i) { s[i] = (char)('0' + (n & 1)); n >>= 1; }
    return s;
}

inline int bitsToInt(const string& s) {
    int val = 0;
    for (char c : s) val = (val << 1) | (c - '0');
    return val;
}

inline string grayEncode(double val, double minv, double maxv, int bits = 8) {
    double norm = (val - minv) / (maxv - minv);
    norm = max(0.0, min(1.0, norm));
    int bin = int(norm * ((1 << bits) - 1));
    return intToBits(binaryToGray(bin), bits);
}

inline double grayDecode(const string& gray, double minv, double maxv, int bits = 8) {
    int bin = grayToBinary(bitsToInt(gray));
    double norm = double(bin) / ((1 << bits) - 1);
    return minv + norm * (maxv - minv);
}

inline string mutateGrayBits(string s, double rate = 0.01) {
    for (auto& c : s)
        if (rand_real() < rate) c = (c == '0' ? '1' : '0');
    return s;
}

bool isSafeValType(FilterType t) {
    if (t == THRESHOLD || t == ERODE || t == DILATE || t == SOBEL_X || t == SOBEL_Y) {
        return false;
    }
    else {
        return true;
    }
}

// =====================================================
// Random tree generation (use rand_int / rand_real)
// =====================================================
shared_ptr<TreeNode> generateRandomTree(int depth = 0, int maxDepth = MAX_DEPTH) {
    if (depth >= maxDepth || rand_real() < 0.1) {
        auto t = make_shared<TreeNode>();
        t->type = TERMINAL_INPUT;
        return t;
    }
    // choose type from 1..NUM_TYPE_FUNC inclusive
    int t_idx = rand_int(1, NUM_TYPE_FUNC);
    FilterType t = static_cast<FilterType>(t_idx);
    auto node = make_shared<TreeNode>();
    node->type = t;

    if (g_paramDesc.count(t)) {
        int numParams = g_paramDesc[t].n;
        node->params.resize(numParams);

        if (isSafeValType(t)) {
            for (int i = 0; i < numParams; i++) {
                node->params[i] = g_paramDesc_safeVal[t][i];
            }
        }
        else {
            for (int i = 0; i < numParams; i++) {
                std::uniform_real_distribution<double> ud(g_paramDesc[t].minv, g_paramDesc[t].maxv);
                node->params[i] = ud(rng);
            }
        }
    }

    int numChildren = (t == BITWISE_AND || t == BITWISE_OR || t == BITWISE_XOR || t == DIFF_PROCESS) ? 2 : 1;
    for (int i = 0; i < numChildren; i++) node->children.push_back(generateRandomTree(depth + 1, maxDepth));
    return node;
}

// =====================================================
// Utilities to inspect/adjust tree arity and depth
// =====================================================
bool isTerminal(FilterType type) {
    return (type == TERMINAL_INPUT);
}
bool isBinaryFilter(FilterType type) {
    return (type == DIFF_PROCESS || type == BITWISE_AND || type == BITWISE_OR || type == BITWISE_XOR);
}
int getTreeMaxDepth(const shared_ptr<TreeNode>& node, int depth = 0) {
    if (!node) return depth;
    if (node->children.empty()) return depth;
    int maxChildDepth = depth;
    for (auto& child : node->children) maxChildDepth = max(maxChildDepth, getTreeMaxDepth(child, depth + 1));
    return maxChildDepth;
}

void ensureArity(shared_ptr<TreeNode>& node) {
    if (!node) return;
    int required = isBinaryFilter(node->type) ? 2 : (isTerminal(node->type) ? 0 : 1);
    // if terminal -> must have 0 children
    if (isTerminal(node->type)) {
        node->children.clear();
        return;
    }
    while ((int)node->children.size() < required) {
        node->children.push_back(make_shared<TreeNode>(TreeNode{ TERMINAL_INPUT, {}, {} }));
    }
    while ((int)node->children.size() > required) node->children.pop_back();
}

void adjustChildrenForType(shared_ptr<TreeNode>& node, int currentDepth, int maxDepth, int overFlag = 0) {
    if (!node) return;
    int remaining = maxDepth - currentDepth;
    if (isTerminal(node->type)) {
        node->children.clear();
        return;
    }
    if (remaining <= 1) {
        node->children.clear();
        int cnt = isBinaryFilter(node->type) ? 2 : 1;
        for (int i = 0; i < cnt; ++i) node->children.push_back(make_shared<TreeNode>(TreeNode{ TERMINAL_INPUT, {}, {} }));
    }
    else {
        int required = isBinaryFilter(node->type) ? 2 : 1;
        if (overFlag) node->children.clear();
        while ((int)node->children.size() < required) node->children.push_back(generateRandomTree(currentDepth + 1, maxDepth));
        while ((int)node->children.size() > required) node->children.pop_back();
    }
    // Recursively ensure arity for children
    for (size_t i = 0; i < node->children.size(); ++i) {
        adjustChildrenForType(node->children[i], currentDepth + 1, maxDepth, overFlag);
    }
}
void confirmDepth(shared_ptr<TreeNode>& root, int maxDepth = MAX_DEPTH) {
    if (!root) return;
    int finalDepth = getTreeMaxDepth(root);
    if (finalDepth > maxDepth) adjustChildrenForType(root, 0, maxDepth, 1);
    // Also ensure nodes have correct arity
    // (walk all nodes and correct)
    std::function<void(shared_ptr<TreeNode>)> walk = [&](shared_ptr<TreeNode> n) {
        if (!n) return;
        ensureArity(n);
        for (auto& c : n->children) walk(c);
        };
    walk(root);
}

double getFinalParamVal(const shared_ptr<TreeNode>& node, int idxParam) {
    switch (node->type) {
    case TERMINAL_INPUT:
        return -1.0;
    case GAUSSIAN_BLUR: {
        switch (idxParam) {
        case 0: {
            int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
            if ((k % 2) == 0) k |= 1;
            return (double)k;
        }
        case 1: {
            double sigma = node->params.size() > 1 ? node->params[1] : 1.5;
            return sigma;
        }
        default:
            return -1.0;
        }
    }
    case MED_BLUR: {
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        return (double)k;
    }
    case BLUR: {
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        return (double)k;
    }
    case BILATERAL_FILTER: {
        switch (idxParam) {
        case 0: {
            int d = node->params.size() > 0 ? int(node->params[0]) : 9;
            return (double)d;
        }
        case 1: {
            double sigmaColor = node->params.size() > 1 ? node->params[1] : 75;
            return sigmaColor;
        }
        case 2: {
            double sigmaSpace = node->params.size() > 2 ? node->params[2] : 75;
            return sigmaSpace;
        }
        default:
            return -1.0;
        }
    }
    case SOBEL_X: {
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        return (double)k;
    }
    case SOBEL_Y: {
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        return (double)k;
    }
    case CANNY: {
        switch (idxParam) {
        case 0: {
            double t1 = node->params.size() > 0 ? node->params[0] : 100;
            return t1;
        }
        case 1: {
            double t2 = node->params.size() > 1 ? node->params[1] : 200;
            return t2;
        }
        default:
            return -1.0;
        }
    }
    case DIFF_PROCESS:
        return -1.0;
    case THRESHOLD: {
        double th = node->params.size() > 0 ? node->params[0] : 127.0;
        return th;
    }
    case ERODE: {
        int r = node->params.size() > 0 ? int(node->params[0]) : 1;
        int k = 1 + 2 * max(0, r);
        return (double)k;
    }
    case DILATE: {
        int r = node->params.size() > 0 ? int(node->params[0]) : 1;
        int k = 1 + 2 * max(0, r);
        return (double)k;
    }
    case CONTOUR_PROCESS: {
        switch (idxParam) {
        case 0: {
            int kk = node->params.size() > 0 ? int(node->params[0]) : 1;
            int k = ((kk) / 2) | 1;
            return (double)k;
        }
        case 1: {
            int times = node->params.size() > 1 ? int(node->params[1]) / 2 : 0;
            return (double)times;
        }
        case 2: {
            int selType = 0;
            if (node->params.size() > 2) selType = min(2, int(node->params[2] / 5));
            return (double)selType;
        }
        case 3: {
            int range = node->params.size() > 3 ? int(node->params[3]) / 2 : 0;
            return (double)range;
        }
        case 4: {
            int areaTh = node->params.size() > 4 ? int(node->params[4]) : 1;
            return (double)areaTh;
        }
        default:
            return -1.0;
        }
    }
    case BITWISE_AND:
        return -1.0;
    case BITWISE_OR:
        return -1.0;
    case BITWISE_NOT:
        return -1.0;
    case BITWISE_XOR:
        return -1.0;
    default:
        return -1.0;
    }
}

// =====================================================
// executeTree with caching for child results (avoid repeated recursion per pixel)
// =====================================================
Mat executeTree(const shared_ptr<TreeNode>& node, const Mat& input) {
    if (!node) return input.clone();
    // For unary, compute child result once; for binary, compute both once
    switch (node->type) {
    case TERMINAL_INPUT:
        return input.clone();
    case GAUSSIAN_BLUR: {
        Mat child = executeTree(node->children[0], input);
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        Mat dst;
        double sigma = node->params.size() > 1 ? node->params[1] : 1.5;
        GaussianBlur(child, dst, Size(k, k), sigma);
        return dst;
    }
    case MED_BLUR: {
        Mat child = executeTree(node->children[0], input);
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        Mat dst;
        medianBlur(child, dst, k);
        return dst;
    }
    case BLUR: {
        Mat child = executeTree(node->children[0], input);
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        Mat dst;
        blur(child, dst, Size(k, k));
        return dst;
    }
    case BILATERAL_FILTER: {
        Mat child = executeTree(node->children[0], input);
        int d = node->params.size() > 0 ? int(node->params[0]) : 9;
        double sigmaColor = node->params.size() > 1 ? node->params[1] : 75;
        double sigmaSpace = node->params.size() > 2 ? node->params[2] : 75;
        Mat dst;
        bilateralFilter(child, dst, d, sigmaColor, sigmaSpace);
        return dst;
    }
    case SOBEL_X: {
        Mat child = executeTree(node->children[0], input);
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        Mat dst;
        Sobel(child, dst, CV_16S, 1, 0, k);
        // convert back to 8-bit absolute
        Mat dst8;
        convertScaleAbs(dst, dst8);
        return dst8;
    }
    case SOBEL_Y: {
        Mat child = executeTree(node->children[0], input);
        int k = max(1, int(node->params.size() > 0 ? int(node->params[0]) : 3));
        if ((k % 2) == 0) k |= 1;
        Mat dst;
        Sobel(child, dst, CV_16S, 0, 1, k);
        Mat dst8;
        convertScaleAbs(dst, dst8);
        return dst8;
    }
    case CANNY: {
        Mat child = executeTree(node->children[0], input);
        double t1 = node->params.size() > 0 ? node->params[0] : 100;
        double t2 = node->params.size() > 1 ? node->params[1] : 200;
        Mat dst;
        Canny(child, dst, t1, t2);
        return dst;
    }
    case DIFF_PROCESS: {
        // --- IMPORTANT: compute child outputs once (cache) ---
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst = Mat::zeros(input.size(), CV_8UC1);
        CV_Assert(a.size() == b.size());
        for (int y = 0; y < a.rows; ++y) {
            const uchar* pa = a.ptr<uchar>(y);
            const uchar* pb = b.ptr<uchar>(y);
            uchar* pd = dst.ptr<uchar>(y);
            for (int x = 0; x < a.cols; ++x) {
                int diffVal = int(pa[x]) - int(pb[x]);
                if (diffVal < 0) diffVal = 0;
                pd[x] = static_cast<uchar>(std::min(255, diffVal));
            }
        }
        return dst;
    }
    case THRESHOLD: {
        Mat child = executeTree(node->children[0], input);
        double th = node->params.size() > 0 ? node->params[0] : 127.0;
        Mat dst;
        threshold(child, dst, th, 255, THRESH_BINARY);
        return dst;
    }
    case ERODE: {
        Mat child = executeTree(node->children[0], input);
        int r = node->params.size() > 0 ? int(node->params[0]) : 1;
        int k = 1 + 2 * max(0, r);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
        Mat dst;
        erode(child, dst, kernel);
        return dst;
    }
    case DILATE: {
        Mat child = executeTree(node->children[0], input);
        int r = node->params.size() > 0 ? int(node->params[0]) : 1;
        int k = 1 + 2 * max(0, r);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
        Mat dst;
        dilate(child, dst, kernel);
        return dst;
    }
    case CONTOUR_PROCESS: {
        Mat maskImg = executeTree(node->children[0], input).clone();
        int kk = node->params.size() > 0 ? int(node->params[0]) : 1;
        int k = ((kk) / 2) | 1;
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
        int times = node->params.size() > 1 ? int(node->params[1]) / 2 : 0;
        for (int t = 0; t < times; ++t) erode(maskImg, maskImg, kernel);
        vector<vector<Point>> contours;
        findContours(maskImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        Mat mask(maskImg.rows, maskImg.cols, CV_8UC1, Scalar(255));
        int selType = 0;
        if (node->params.size() > 2) selType = min(2, int(node->params[2] / 5));
        for (const auto& contour : contours) {
            Rect bb = boundingRect(contour);
            double aspect_ratio = double(bb.width) / double(bb.height + 1e-9);
            if (selType == 0) {
                int range = node->params.size() > 3 ? int(node->params[3]) / 2 : 0;
                if (aspect_ratio >= (1 - range * 0.1) && aspect_ratio <= (1 + range * 0.1)) {
                    drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(0), -1);
                }
            }
            else if (selType == 1) {
                int areaTh = node->params.size() > 4 ? int(node->params[4]) : 1;
                if (contourArea(contour) >= 100 * areaTh) {
                    drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(0), -1);
                }
            }
            else {
                int range = node->params.size() > 3 ? int(node->params[3]) : 0;
                int areaTh = node->params.size() > 4 ? int(node->params[4]) : 1;
                if ((aspect_ratio >= (1 - range * 0.1) && aspect_ratio <= (1 + range * 0.1)) && (contourArea(contour) >= 100 * areaTh)) {
                    drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(0), -1);
                }
            }
        }
        return mask;
    }
    case BITWISE_AND: {
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst;
        bitwise_and(a, b, dst);
        return dst;
    }
    case BITWISE_OR: {
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst;
        bitwise_or(a, b, dst);
        return dst;
    }
    case BITWISE_NOT: {
        Mat a = executeTree(node->children[0], input);
        Mat dst;
        bitwise_not(a, dst);
        return dst;
    }
    case BITWISE_XOR: {
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst;
        bitwise_xor(a, b, dst);
        return dst;
    }
    default:
        return input.clone();
    }
}

// =====================================================
// Node collection & utilities
// =====================================================
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

void collectParams(shared_ptr<TreeNode> root, vector<shared_ptr<TreeNode>>& out) {
    if (!root) return;
    if (!root->params.empty()) out.push_back(root);
    for (auto& c : root->children) collectParams(c, out);
}

// =====================================================
// F1 metric and scoring (force binary outputs before evaluation)
// =====================================================
double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

double calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], int numInd) {
    double f1_score[numSets];
    for (int idxSet = 0; idxSet < numSets; idxSet++) {
        int tp = 0, fp = 0, fn = 0;
        CV_Assert(metaImg_g[idxSet].size() == tarImg_g[idxSet].size());
        for (int i = 0; i < metaImg_g[idxSet].rows; i++) {
            for (int j = 0; j < metaImg_g[idxSet].cols; j++) {
                if (metaImg_g[idxSet].at<uchar>(i, j) != 0 && metaImg_g[idxSet].at<uchar>(i, j) != 255) {
                    // non-binary reached (shouldn't after thresholding) => penalize
                    return 0.01;
                }
                if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 0) tp += 1;
                if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 255) fp += 1;
                if (metaImg_g[idxSet].at<uchar>(i, j) == 255 && tarImg_g[idxSet].at<uchar>(i, j) == 0) fn += 1;
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

    //for (int idxSet = 0; idxSet < numSets; idxSet++) {
    //    sum_f1 += f1_score[idxSet];
    //}

    for (int idxSet = 0; idxSet < numSets; idxSet++) {
        if (numInd != -1) { // in the last generation
            indFValInfo[numInd][idxSet] = f1_score[idxSet];
        }
        sum_f1 += f1_score[idxSet];
    }

    if (numInd != -1) {
        indFValInfo[numInd][numSets] = sum_f1;
    }

    // store in indFValInfo externally if needed by caller
    return sum_f1;
}

// calScoreByInd: execute tree for each set, force binary by thresholding if needed
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
    double firstScore = calScoreByInd(population[0], imgArr, -1);
    double minFValue = firstScore;
    double maxFValue = firstScore;
    double aveFValue = 0.0;
    double deviation = 0.0;
    double variance = 0.0;
    double sumFValue = 0.0;
    double scoreArr[POP_SIZE];

    genType curGenInfo;
    for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
        scoreArr[idxInd] = calScoreByInd(population[idxInd], imgArr, -1);
    }

    for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
        double tmp = scoreArr[idxInd];
        sumFValue += tmp;
        if (tmp > maxFValue) { maxFValue = tmp; curMaxFvalIdx = idxInd; }
        if (tmp < minFValue) { minFValue = tmp; }
    }
    curGenInfo.eliteTree = cloneTree(population[curMaxFvalIdx]);
    curGenInfo.eliteFValue = maxFValue;
    aveFValue = sumFValue / POP_SIZE;
    curGenInfo.genMinFValue = minFValue;
    curGenInfo.genAveFValue = aveFValue;
    for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
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
    for (int i = 0; i < depth; ++i) fprintf(fp, "    ");
    fprintf(fp, "%s", filterTypeToString(node->type).c_str());
    if (!node->params.empty()) {
        fprintf(fp, " (");
        for (size_t i = 0; i < node->params.size(); i++) {
            fprintf(fp, "%.2f", getFinalParamVal(node, i));
            if (i + 1 < node->params.size()) fprintf(fp, ", ");
        }
        fprintf(fp, ")");
    }
    fprintf(fp, "\n");
    for (const auto& c : node->children) printTree(c, depth + 1, fp);
}

// =====================================================
// genetic operators: crossover & mutate (robustified)
// =====================================================
void crossover(shared_ptr<TreeNode>& a, shared_ptr<TreeNode>& b) {
    vector<NodeWithParent> nodesA, nodesB;
    collectNodesWithParents(a, nullptr, nodesA);
    collectNodesWithParents(b, nullptr, nodesB);
    vector<NodeWithParent> validA, validB;
    for (const auto& np : nodesA) if (np.second) validA.push_back(np);
    for (const auto& np : nodesB) if (np.second) validB.push_back(np);
    if (validA.empty() || validB.empty()) return;
    int idxA = rand_int(0, (int)validA.size() - 1);
    int idxB = rand_int(0, (int)validB.size() - 1);

    auto nodeA = validA[idxA].first;
    auto parentA = validA[idxA].second;
    auto nodeB = validB[idxB].first;
    auto parentB = validB[idxB].second;

    auto itA = find(parentA->children.begin(), parentA->children.end(), nodeA);
    auto itB = find(parentB->children.begin(), parentB->children.end(), nodeB);
    if (itA != parentA->children.end() && itB != parentB->children.end()) {
        // swap
        swap(*itA, *itB);
        // ensure arity & depth constraints after swap
        ensureArity(*itA);
        ensureArity(*itB);
        confirmDepth(a);
        confirmDepth(b);
    }
}

void mutate(shared_ptr<TreeNode>& root, int maxDepth = MAX_DEPTH) {
    vector<NodeWithParent> nodesRoot;
    collectNodesWithParents(root, nullptr, nodesRoot);
    if (nodesRoot.empty()) return;
    size_t pick = rand_int(0, (int)nodesRoot.size() - 1);
    auto target = nodesRoot[pick].first;
    auto targetParent = nodesRoot[pick].second;

    int idxTargetInParent = -1;
    int currentDepth = 0;
    if (targetParent) {
        for (size_t i = 0; i < targetParent->children.size(); ++i)
            if (targetParent->children[i] == target) { idxTargetInParent = (int)i; break; }
        if (idxTargetInParent == -1) return;
        function<int(shared_ptr<TreeNode>, int)> findDepth = [&](shared_ptr<TreeNode> node, int depth) -> int {
            if (node == target) return depth;
            for (auto& c : node->children) {
                int d = findDepth(c, depth + 1);
                if (d != -1) return d;
            }
            return -1;
            };
        currentDepth = findDepth(root, 0);
    }
    auto replaceInParent = [&](const shared_ptr<TreeNode>& repl) {
        if (!targetParent) root = repl;
        else targetParent->children[static_cast<size_t>(idxTargetInParent)] = repl;
        };

    int mutationType = rand_int(0, 2);
    switch (mutationType) {
    case 0: { // modify type (and params)
        int newTypeIdx = rand_int(0, NUM_TYPE_FUNC); // allow terminal sometimes
        target->type = static_cast<FilterType>(newTypeIdx);
        // if new type has params, initialize them
        if (g_paramDesc.count(target->type)) {
            // auto pd = g_paramDesc[target->type];
            int numParams = g_paramDesc[target->type].n;
            target->params.resize(numParams);
            for (int i = 0; i < numParams; ++i) {
                //std::uniform_real_distribution<double> ud(pd.minv, pd.maxv);
                //target->params[i] = ud(rng);
                target->params[i] = g_paramDesc_safeVal[target->type][i];
            }
        }
        else {
            target->params.clear();
        }
        adjustChildrenForType(target, currentDepth, maxDepth);
        break;
    }
    case 1: { // insert above target
        int remainingDepth = maxDepth - currentDepth;
        if (remainingDepth <= 1) break;
        auto newNode = make_shared<TreeNode>();
        int newTypeIdx = rand_int(1, NUM_TYPE_FUNC);
        newNode->type = static_cast<FilterType>(newTypeIdx);
        if (g_paramDesc.count(newNode->type)) {
            int numParams = g_paramDesc[newNode->type].n;
            newNode->params.resize(numParams);
            if (isSafeValType(newNode->type)) {
                for (int i = 0; i < numParams; ++i) {
                    newNode->params[i] = g_paramDesc_safeVal[newNode->type][i];
                }
            }
            else {
                for (int i = 0; i < numParams; ++i) {
                    std::uniform_real_distribution<double> ud(g_paramDesc[newNode->type].minv, g_paramDesc[newNode->type].maxv);
                    newNode->params[i] = ud(rng);
                }
            }
        }
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
    case 2: { // delete target (replace with its first child)
        if (!isTerminal(target->type) && !target->children.empty()) {
            replaceInParent(target->children[0]);
        }
        break;
    }
    }
    confirmDepth(root);
}

// =====================================================
// GA: runGrayGA_forTree (optimize parameters of one tree)
// returns decoded parameter vector in DFS order (same ordering as collectParams)
// =====================================================
vector<double> runGrayGA_forTree(shared_ptr<TreeNode> rootInGP, Mat imgArr[][2]) {
    // collect parameter nodes
    vector<shared_ptr<TreeNode>> paramNodes;
    collectParams(rootInGP, paramNodes);
    if (paramNodes.empty()) return {}; // nothing to optimize

    // build bounds and baseGenes
    vector<pair<double, double>> bounds;
    vector<string> baseGenes;
    for (auto& pNode : paramNodes) {
        auto pd = g_paramDesc[pNode->type];
        for (int i = 0; i < pd.n; ++i) {
            bounds.emplace_back(pd.minv, pd.maxv);
            baseGenes.push_back(grayEncode(pNode->params[i], pd.minv, pd.maxv, 8));
        }
    }
    int numsParams = (int)baseGenes.size();

    struct GrayInd { vector<string> arrStrVal; double fit = 0.01; };
    vector<GrayInd> pop(GA_POP);

    // initialize GA population
    for (int idxInd = 0; idxInd < GA_POP; idxInd++) {
        pop[idxInd].arrStrVal = baseGenes;
        if (idxInd >= 1) {
            for (auto& g : pop[idxInd].arrStrVal) g = mutateGrayBits(g, 0.05);
        }
        // decode and evaluate
        vector<double> arrDecoded;
        arrDecoded.reserve(numsParams);
        for (int i = 0; i < numsParams; ++i) arrDecoded.push_back(grayDecode(pop[idxInd].arrStrVal[i], bounds[i].first, bounds[i].second));
        auto rootCloned = cloneTree(rootInGP);
        vector<shared_ptr<TreeNode>> nodesToSet;
        collectParams(rootCloned, nodesToSet);
        int pos = 0;
        for (auto& pn : nodesToSet)
            for (auto& p : pn->params) p = arrDecoded[pos++];
        pop[idxInd].fit = calScoreByInd(rootCloned, imgArr, -1);
    }

    // GA loop (elitism + one-point-ish crossover per gene)
    for (int gen = 0; gen < GA_GENERATIONS; ++gen) {
        sort(pop.begin(), pop.end(), [](const GrayInd& a, const GrayInd& b) { return a.fit > b.fit; });
        vector<GrayInd> newpop;
        newpop.push_back(pop[0]); // elite
        while ((int)newpop.size() < GA_POP) {
            int a = rand_int(0, GA_POP - 1), b = rand_int(0, GA_POP - 1);
            GrayInd child = pop[a];
            for (int i = 0; i < numsParams; ++i) {
                if (rand_real() < 0.5) child.arrStrVal[i] = pop[b].arrStrVal[i];
                // mutation
                child.arrStrVal[i] = mutateGrayBits(child.arrStrVal[i], 0.01);
            }
            // decode and evaluate
            vector<double> arrDecoded;
            arrDecoded.reserve(numsParams);
            for (int i = 0; i < numsParams; ++i) arrDecoded.push_back(grayDecode(child.arrStrVal[i], bounds[i].first, bounds[i].second));
            auto rootCloned = cloneTree(rootInGP);
            vector<shared_ptr<TreeNode>> nodesToSet;
            collectParams(rootCloned, nodesToSet);
            int pos = 0;
            for (auto& pn : nodesToSet)
                for (auto& p : pn->params) p = arrDecoded[pos++];
            child.fit = calScoreByInd(rootCloned, imgArr, -1);
            newpop.push_back(child);
        }
        pop.swap(newpop);
    }

    sort(pop.begin(), pop.end(), [](const GrayInd& a, const GrayInd& b) { return a.fit > b.fit; });
    vector<double> decoded;
    for (int i = 0; i < (int)baseGenes.size(); ++i)
        decoded.push_back(grayDecode(pop.front().arrStrVal[i], bounds[i].first, bounds[i].second));
    return decoded;
}

// =====================================================
// bias calculation
// =====================================================
double calcBias(const vector<genType>& genInfo, int window = BIAS_WINDOW) {
    if (genInfo.size() < (size_t)(window + 1)) return 1.0;
    double recentAvg = 0.0;
    for (int i = (int)genInfo.size() - window; i < (int)genInfo.size(); ++i) recentAvg += genInfo[i].eliteFValue;
    recentAvg /= window;
    double totalAvg = 0.0;
    for (auto& g : genInfo) totalAvg += g.eliteFValue;
    totalAvg /= genInfo.size();
    double bias = fabs(recentAvg - totalAvg) / (totalAvg + 1e-9);
    return bias;
}

// =====================================================
// Top-level GP+GA loop
// =====================================================
void multiProcess(Mat imgArr[][2]) {
    Mat resImg[numSets];
    Mat tarImg[numSets];

    char imgName_pro[numSets][256];
    char imgName_final[numSets][256];

    FILE* fl_fValue = nullptr;
    errno_t err = fopen_s(&fl_fValue, "./imgs_1006_2025_v5/output/f_value.txt", "w");
    if (err != 0 || fl_fValue == nullptr) {
        perror("Cannot open the file");
        // continue without file
    }

    FILE* fl_maxFval = nullptr;
    errno_t err2 = fopen_s(&fl_maxFval, "./imgs_1006_2025_v5/output/maxFvalInfo_final.txt", "w");
    if (err2 != 0 || fl_maxFval == nullptr) {
        perror("Cannot open the file");
    }

    FILE* fl_printTree = nullptr;
    errno_t err3 = fopen_s(&fl_printTree, "./imgs_1006_2025_v5/output/printed_tree.txt", "w");
    if (err3 != 0 || fl_printTree == nullptr) {
        perror("Cannot open the file");
    }

    initParamDesc();
    initParamDesc_safeVal();

    // protection lifetime: if GA optimized an index at gen G, set protectedUntil[idx]=G+1
    vector<int> protectedUntil(POP_SIZE, -1);

    /*
      Break-Point-01
    */
    bool flag_tri_GA = false;
    int idx_opt_GA = -1;
    double fitness_opt_GA = 0.0;

    for (int idxProTimes = 0; idxProTimes < sysRunTimes; idxProTimes++) {
        vector<genType> genInfo;
        vector<shared_ptr<TreeNode>> population;
        population.reserve(POP_SIZE);
        for (int i = 0; i < POP_SIZE; ++i) population.push_back(generateRandomTree());

        double biasThreshold = INITIAL_BIAS_THRESHOLD;

        for (int numGen = 0; numGen < GENERATIONS; numGen++) {
            // cout << "---------idxProTimes: " << idxProTimes + 1 << ", generation: " << numGen + 1 << "---------" << endl;
            printf("---------idxProTimes: %d, generation: %d---------\n", idxProTimes + 1, numGen + 1);

            /*
              Break-Point-02
            */
            if (flag_tri_GA) {
                printf("(02)(res of last gen) the idx_opt_GA: %d, the fitness_opt_GA(cur-gen): %.4f\n", idx_opt_GA, calScoreByInd(population[idx_opt_GA], imgArr, -1));
            }

            // select two parents that are NOT protected for this generation
            int idx1 = rand_int(0, POP_SIZE - 1);
            int idx2 = rand_int(0, POP_SIZE - 1);
            auto choose_nonprotected = [&](int avoidGen)->int {
                int tries = 0;
                while (tries < 50) {
                    int c = rand_int(0, POP_SIZE - 1);
                    // modified point, " 「<=」→「<」 "
                    if (protectedUntil[c] < avoidGen) return c;
                    tries++;
                }
                // fallback
                return rand_int(0, POP_SIZE - 1);
                };
            idx1 = choose_nonprotected(numGen);
            idx2 = choose_nonprotected(numGen);
            while (idx2 == idx1) idx2 = choose_nonprotected(numGen);

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
                auto chosen = (rand_real() < 0.5) ? childA : childB;
                double fit = calScoreByInd(chosen, imgArr, -1);
                family.push_back({ fit, chosen });
            }

            for (int idxInd = 0; idxInd < (OFFSPRING_COUNT + 2); idxInd++) {
                if (rand_real() < MUTATION_RATE) {
                    mutate(family[idxInd].second);
                    family[idxInd].first = calScoreByInd(family[idxInd].second, imgArr, -1);
                }
            }

            sort(family.rbegin(), family.rend()); // descending by fitness
            auto elite = family[0];
            double total = 0;
            for (const auto& f : family) total += f.first;
            double r = rand_real() * total, accum = 0;
            shared_ptr<TreeNode> rouletteSelected = family[1].second;
            double scoreRouletteSelected = 0.01;
            for (const auto& f : family) {
                accum += f.first;
                if (accum >= r) { rouletteSelected = f.second; scoreRouletteSelected = f.first; break; }
            }

            if (elite.first > score1) {
                population[idx1] = cloneTree(elite.second);
            }
            if (scoreRouletteSelected > score2) {
                population[idx2] = cloneTree(rouletteSelected);
            }

            genInfo.push_back(getCurGenInfo(population, imgArr));
            double bias = calcBias(genInfo);
            // printf("the score of elite(%d gen): %.4f, bias: %.4f\n", numGen + 1, genInfo[numGen].eliteFValue, bias);

            /*
              Break-Point-03
            */
            printf("(03)the idx of idx_ind_GP: %d, the fitness of GP: %.4f, the bias: %.4f", curMaxFvalIdx, genInfo[numGen].eliteFValue, bias);

            // ---- Trigger GA when Bias low ----
            if (numGen != GENERATIONS - 1) {
                // if (ENABLE_GA && bias < biasThreshold) {
                if (genInfo[numGen].eliteFValue >= GA_TRIGGER_THRESH && bias < biasThreshold) {
                    // cout << "[PT-ACTIT] Trigger GA Phase (Bias=" << bias << ")" << endl;
                    printf("[PT-ACTIT] Trigger GA Phase (Bias: %.2f)\n", bias);

                    // GA on current elite (curMaxFvalIdx computed in getCurGenInfo)
                    auto eliteIndex = curMaxFvalIdx;
                    auto eliteTreeClone = cloneTree(population[eliteIndex]);
                    auto bestParams = runGrayGA_forTree(eliteTreeClone, imgArr);
                    if (!bestParams.empty()) {
                        // write back decoded params into eliteTreeClone then into population[eliteIndex]
                        vector<shared_ptr<TreeNode>> paramNodes;
                        collectParams(eliteTreeClone, paramNodes);
                        int pos = 0;
                        for (auto& n : paramNodes) {
                            for (auto& p : n->params) {
                                if (pos < (int)bestParams.size()) p = bestParams[pos++];
                            }
                        }

                        /*
                          Break-Point-04
                        */
                        if (calScoreByInd(eliteTreeClone, imgArr, -1) <= genInfo[numGen].eliteFValue) {
                            printf("(04)Although GA optimization was triggered, it failed to improve the fitness value.\n");
                            if (flag_tri_GA) flag_tri_GA = false;
                        }
                        else {
                            // replace population's elite with optimized tree
                            population[eliteIndex] = cloneTree(eliteTreeClone);
                            /*
                              Break-Point-05
                            */
                            flag_tri_GA = true;
                            idx_opt_GA = eliteIndex;
                            fitness_opt_GA = calScoreByInd(population[eliteIndex], imgArr, -1);
                            printf("(05)(cur-GA) the idx_opt_GA: %d, the fitness_opt_GA: %.4f\n", idx_opt_GA, fitness_opt_GA);
                            // protect this index in the next generation (avoid being selected & immediately broken)
                            protectedUntil[eliteIndex] = numGen + 1;
                            // cout << "[PT-ACTIT] GA wrote optimized params to elite index " << eliteIndex << " and protected until gen " << numGen + 2 << endl;
                            printf("[PT-ACTIT] GA wrote optimized params to elite index %d and protected until gen %d\n", eliteIndex, numGen + 2);
                            biasThreshold *= BIAS_DECAY;
                        }
                    }
                }
                else {
                    /*
                      Break-Point-06
                    */
                    if (flag_tri_GA) flag_tri_GA = false;
                }
            }
            else {
                // final generation: record indFValInfo
                for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
                    calScoreByInd(population[idxInd], imgArr, idxInd);
                    // optionally also store per-set results if needed
                }
            }
        } // end generation

        printf("---------------- GEN-END --------------\n");

        // If best improved beyond threshold, write outputs and save tree
        if (indFValInfo[curMaxFvalIdx][numSets] > curThreshFVal) {
            curThreshFVal = indFValInfo[curMaxFvalIdx][numSets];
            printTree(genInfo.back().eliteTree, 0, fl_printTree);
            // write f-values
            for (size_t i = 0; i < genInfo.size(); ++i) {
                if (fl_fValue) fprintf(fl_fValue, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
            }
            if (fl_maxFval) {
                for (int i = 0; i <= numSets; i++) fprintf(fl_maxFval, "%.4f ", indFValInfo[curMaxFvalIdx][i]);
                fprintf(fl_maxFval, "\n");
            }
            // save final images for elite of last generation
            for (int idxSet = 0; idxSet < numSets; idxSet++) {
                Mat res = executeTree(genInfo.back().eliteTree, imgArr[idxSet][0]);
                sprintf_s(imgName_pro[idxSet], "./imgs_1006_2025_v5/output/img_0%d/Gen-Final-t%d.png", idxSet + 1, idxProTimes + 1);
                imwrite(imgName_pro[idxSet], res);
                Mat concat;
                vector<Mat> vec = { res, imgArr[idxSet][1] };
                hconcat(vec, concat);
                sprintf_s(imgName_final[idxSet], "./imgs_1006_2025_v5/output/img_0%d/imgs_final-t%d.png", idxSet + 1, idxProTimes + 1);
                imwrite(imgName_final[idxSet], concat);
            }
        }

        // reset curMaxFvalIdx
        curMaxFvalIdx = 0;
    } // end sysRunTimes

    if (fl_fValue) fclose(fl_fValue);
    if (fl_maxFval) fclose(fl_maxFval);
    if (fl_printTree) fclose(fl_printTree);
}

// =====================================================
// main
// =====================================================
int main(void) {
    Mat imgArr[numSets][2];
    char inputPathName_ori[256];
    char inputPathName_tar[256];

    if (numSets == 1) {
        sprintf_s(inputPathName_ori, "./imgs_1006_2025_v5/input/oriImg_0%d.png", idSet);
        sprintf_s(inputPathName_tar, "./imgs_1006_2025_v5/input/tarImg_0%d.png", idSet);
        imgArr[0][0] = imread(inputPathName_ori, 0);
        imgArr[0][1] = imread(inputPathName_tar, 0);
    }
    else {
        for (int i = 0; i < numSets; i++) {
            sprintf_s(inputPathName_ori, "./imgs_1006_2025_v5/input/oriImg_0%d.png", i + 1);
            sprintf_s(inputPathName_tar, "./imgs_1006_2025_v5/input/tarImg_0%d.png", i + 1);
            imgArr[i][0] = imread(inputPathName_ori, 0);
            imgArr[i][1] = imread(inputPathName_tar, 0);
        }
    }

    // basic checks
    for (int i = 0; i < numSets; ++i) {
        if (imgArr[i][0].empty() || imgArr[i][1].empty()) {
            cerr << "Warning: image pair " << i << " could not be loaded. Check paths." << endl;
        }
    }

    multiProcess(imgArr);
    return 0;
}
