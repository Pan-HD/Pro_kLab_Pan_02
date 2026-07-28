// Week-01 GP/CUDA calibration version
// Based on the stable deterministic GP/CUDA training version.
// Purpose:
//   1) calibrate runtime for TRAIN_SIZE = 5 / 10 / 20 / 50 / 100,
//   2) estimate GA overhead at TRAIN_SIZE = 20,
//   3) collect sec/generation, CAL_SCORE count, GA trigger/success/failure statistics.
// Configuration defaults can be overridden by compiler macros or edited below.
//
// Configuration: USE_CUDA=1, USE_FUSED_METRICS=1, ENABLE_CC_FILTER=1.
// CC_FILTER is expected to be linked with the deterministic GPU CCL fallback implementation.
// Validation completed: random-tree TwoExec, repeat-score, old/fused same-output metrics,
// and 2000-generation training with EliteDropCount=0.

#include <fstream>
#include <sstream>
#include <stack>
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
#include <omp.h>
#include <mutex>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <climits>
// (!important) The foreground mode (pills: 0, crack: 1) must be confirmed before the program runs.
#include "./CommonComponents/SegmentationConfig.h" 
#include "./CUDA_Components/cuda_metrics_fused.cuh"

using namespace std;
using namespace cv;

#ifndef IMG_START_IDX
#define IMG_START_IDX 111
#endif

#ifndef TRAIN_SIZE
#define TRAIN_SIZE 50
#endif

// #define idSet 1 // for mark the selected set if the TRAIN_SIZE been set of 1

// GP parameters
#ifndef POP_SIZE
#define POP_SIZE 100 // Pop_Size of GP
#endif

// Week-01 default is 1000 generations for calibration.
// For the final ViEW-scale runs, override this to 10000.
#ifndef GENERATIONS
#define GENERATIONS 10000 // oriVal: 1000, 10000
#endif
#ifndef OFFSPRING_COUNT
#define OFFSPRING_COUNT 20 // OFFSPRING_COUNT of GP
#endif

#ifndef MUTATION_RATE
#define MUTATION_RATE 0.9 // GP
#endif

#define NUM_TYPE_FUNC 16 // GP

// depth is counted from 0, so MAX_DEPTH=12 means 13 tree levels.
#ifndef MAX_DEPTH
#define MAX_DEPTH 12 // { 0, 1, 2, ... } GP
#endif

// GA parameters
// For Week-01 and multi-TRAIN_SIZE experiments, trigger thresholds are normalized
// by TRAIN_SIZE: eliteMeanF = eliteFValue / TRAIN_SIZE.
#ifndef GA_TRIGGER_MEAN_THRESH
#define GA_TRIGGER_MEAN_THRESH 0.31 // oriVal: 0.77
#endif

#ifndef PRUNE_TRIGGER_MEAN_THRESH
#define PRUNE_TRIGGER_MEAN_THRESH 0.31 // oriVal: 0.77
#endif

#ifndef GA_POP
#define GA_POP 30
#endif

#ifndef GA_GENERATIONS
#define GA_GENERATIONS 100
#endif

#ifndef INITIAL_BIAS_THRESHOLD
#define INITIAL_BIAS_THRESHOLD 0.23
#endif

#ifndef BIAS_DECAY
#define BIAS_DECAY 0.99
#endif

#ifndef BIAS_WINDOW
#define BIAS_WINDOW 5
#endif

#ifndef USE_FIXED_SEED
#define USE_FIXED_SEED 1
#endif

#ifndef RANDOM_SEED
#define RANDOM_SEED 44
#endif

// A delay threshold for re-invoking the GA module—defined as the number of generations that must
// pass before the GA module can be called again—set to reduce computational time when
// optimization fails after invoking the GA module.
#ifndef NUM_DELAY_GA
#define NUM_DELAY_GA 200
#endif

#ifndef USE_CUDA
#define USE_CUDA 1
#endif

//#define CUDA_EQ_TEST_BILATERAL 0
//#define CUDA_EQ_TEST_MED       0 
//#define CUDA_EQ_TEST_BLUR      0 
//#define CUDA_EQ_TEST_ERODE     0 
//#define CUDA_EQ_TEST_DILATE    0 
//#define CUDA_EQ_TEST_DIFF      0

#define USE_FUSED_METRICS 1
// TEST-B
#define ENABLE_CC_FILTER 1

// =====================================================
// Week-01 experiment switches
// =====================================================
// Use this file for two first-week calibration modes:
//
// Mode A: GP-only time calibration
//   WEEK01_ENABLE_GA = 0
//   TRAIN_SIZE = 5 / 10 / 20 / 50 / 100
//   GENERATIONS = 500 or 1000
//
// Mode B: GA overhead calibration
//   WEEK01_ENABLE_GA = 1
//   TRAIN_SIZE = 20
//   GENERATIONS = 1000
//
// Example compiler overrides:
//   /DTRAIN_SIZE=20 /DGENERATIONS=1000 /DWEEK01_ENABLE_GA=1 /DRANDOM_SEED=44
#ifndef WEEK01_ENABLE_GA
#define WEEK01_ENABLE_GA 0
#endif

#ifndef WEEK01_ENABLE_PRUNING
#define WEEK01_ENABLE_PRUNING 0
#endif

#ifndef WEEK01_LOG_INTERVAL
#define WEEK01_LOG_INTERVAL 50
#endif

#ifndef WEEK01_RUN_LABEL
#define WEEK01_RUN_LABEL "week02_TS50_GP_G10000_seed44"
#endif

#ifndef WEEK01_SUMMARY_CSV
#define WEEK01_SUMMARY_CSV "./imgs_0710_2026_v1/output/train_output/TS50_GP_G10000_seed44_summary.csv"
#endif


#if USE_CUDA
#define CAL_SCORE calScoreByIndCUDA
#else
#define CAL_SCORE calScoreByInd
#endif

// =====================================================
// Random utilities (unified)
// =====================================================
// static std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

static thread_local std::mt19937 rng(
#if USE_FIXED_SEED
    RANDOM_SEED + omp_get_thread_num()
#else
    (unsigned)(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + omp_get_thread_num()
#endif
);

/*
    CUDA Runtime
*/
// thread_local cv::cuda::Stream gCudaStream;
inline cv::cuda::Stream& getCudaStream()
{
    thread_local cv::cuda::Stream s;
    return s;
}

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
    // CONTOUR_PROCESS,
    CC_FILTER,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_NOT,
    BITWISE_XOR,
};

/*
    Filter Cache Modules
*/
struct GaussianKey
{
    int srcType;
    int dstType;
    int ksize;
    int sigma100;

    bool operator==(const GaussianKey& o) const {
        return srcType == o.srcType &&
            dstType == o.dstType &&
            ksize == o.ksize &&
            sigma100 == o.sigma100;
    }
};

struct GaussianKeyHash
{
    size_t operator()(
        const GaussianKey& k) const {
        return
            ((size_t)k.srcType) ^
            ((size_t)k.dstType << 8) ^
            ((size_t)k.ksize << 16) ^
            ((size_t)k.sigma100 << 24);
    }
};

thread_local unordered_map<GaussianKey, cv::Ptr<cv::cuda::Filter>, GaussianKeyHash> gGaussianCache;

// static std::mutex gGaussianMutex;

// MedianKey
struct MedianKey
{
    int srcType;
    int ksize;

    bool operator==(const MedianKey& o) const {
        return srcType == o.srcType &&
            ksize == o.ksize;
    }
};

struct MedianKeyHash
{
    size_t operator()(
        const MedianKey& k) const {
        return
            ((size_t)k.srcType) ^
            ((size_t)k.ksize << 8);
    }
};
thread_local unordered_map<MedianKey, cv::Ptr<cv::cuda::Filter>, MedianKeyHash> gMedianCache;
// static std::mutex gMedianMutex;

// BoxKey
struct BoxKey
{
    int srcType;
    int dstType;
    int ksize;

    bool operator==(const BoxKey& o) const {
        return srcType == o.srcType && dstType == o.dstType && ksize == o.ksize;
    }
};

struct BoxKeyHash
{
    size_t operator()(const BoxKey& k) const {
        return
            ((size_t)k.srcType) ^
            ((size_t)k.dstType << 8) ^
            ((size_t)k.ksize << 16);
    }
};
thread_local unordered_map<BoxKey, cv::Ptr<cv::cuda::Filter>, BoxKeyHash> gBoxCache;
// static std::mutex gBoxMutex;

// SobelKey
struct SobelKey
{
    int srcType;
    int dstType;
    int dx;
    int dy;
    int ksize;

    bool operator==(const SobelKey& o) const {
        return srcType == o.srcType &&
            dstType == o.dstType &&
            dx == o.dx &&
            dy == o.dy &&
            ksize == o.ksize;
    }
};

struct SobelKeyHash
{
    size_t operator()(const SobelKey& k) const {
        return
            ((size_t)k.srcType) ^
            ((size_t)k.dstType << 8) ^
            ((size_t)k.dx << 16) ^
            ((size_t)k.dy << 24) ^
            ((size_t)k.ksize << 32);
    }
};

thread_local unordered_map<SobelKey, cv::Ptr<cv::cuda::Filter>, SobelKeyHash> gSobelCache;
// static std::mutex gSobelMutex;

// CannyKey
struct CannyKey
{
    int t1;
    int t2;
    bool operator==(const CannyKey& o) const {
        return  t1 == o.t1 &&
            t2 == o.t2;
    }
};
struct CannyKeyHash
{
    size_t operator()(const CannyKey& k) const {
        return
            ((size_t)k.t1) ^
            ((size_t)k.t2 << 8);
    }
};
thread_local unordered_map<CannyKey, cv::Ptr<cv::cuda::CannyEdgeDetector>, CannyKeyHash> gCannyCache;
// static std::mutex gCannyMutex;

// MorphologyKey
struct MorphologyKey
{
    int op; // 0 for erode, 1 for dilate
    int srcType;
    int ksize;
    bool operator==(const MorphologyKey& o) const {
        return op == o.op &&
            srcType == o.srcType &&
            ksize == o.ksize;
    }
};

struct MorphologyKeyHash
{
    size_t operator()(const MorphologyKey& k) const {
        return
            ((size_t)k.op) ^
            ((size_t)k.srcType << 8) ^
            ((size_t)k.ksize << 16);
    }
};

thread_local unordered_map<MorphologyKey, cv::Ptr<cv::cuda::Filter>, MorphologyKeyHash> gMorphologyCache;
// static std::mutex gMorphologyMutex;

struct ParamDesc { // describle of params
    int n;
    double minv;
    double maxv;
};
static unordered_map<FilterType, ParamDesc> g_paramDesc; // <type, desc of params>

void initParamDesc() {
    g_paramDesc.clear();
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
    // g_paramDesc[CONTOUR_PROCESS] = { 5, 0.0, 15.0 };
    g_paramDesc[CC_FILTER] = { 2, 0.0, 100.0 };
}

static unordered_map<FilterType, vector<double>> g_paramDesc_safeVal; // <type, desc of params(array)>
void initParamDesc_safeVal() {
    g_paramDesc_safeVal.clear();
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

    //g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(6.0); // kernel size ( val = param / 2 )
    //g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(6.0); // erode・dilate times ( val = param / 2 )
    //g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(12.0); // selType ( val = param / 5 )
    //g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(2.0); // range ( val = param / 2 )
    //g_paramDesc_safeVal[CONTOUR_PROCESS].push_back(2.0); // areaTh

    g_paramDesc_safeVal[CC_FILTER].push_back(5.0); // areaTh
    g_paramDesc_safeVal[CC_FILTER].push_back(3.0); // aspectRange
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

    int eliteIndex;
};

struct PruneResult {
    bool success = false;
    double scoreAfter = 0.0;
};

struct MetricsGPU
{
    int tp;
    int fp;
    int fn;
};

#if USE_CUDA
cv::cuda::GpuMat gImgArr[TRAIN_SIZE][2];
cv::cuda::GpuMat gTarImgArr[TRAIN_SIZE];
#endif

double indFValInfo[POP_SIZE][TRAIN_SIZE + 1];
int lastGenFailGA = -1;
int countPruneSum = 0;
int countPruneSuccess = 0;
int countPruneFail = 0;
int gEliteDropCount = 0;

// =====================================================
// Week-01 runtime and module instrumentation
// =====================================================
long long gCalScoreCallCount = 0;
double gCalScoreTotalSec = 0.0;

int gGetCurGenInfoCalls = 0;
double gGetCurGenInfoTotalSec = 0.0;

int gGATriggerCount = 0;
int gGASuccessCount = 0;
int gGAFailCount = 0;
int gGAEmptyCount = 0;
int gGABlockedCount = 0;
double gGATotalSec = 0.0;
double gGABestSingleSec = 0.0;

int gWeek01PruneAttemptCount = 0;
int gWeek01PruneSuccessCount = 0;
int gWeek01PruneFailCount = 0;
double gWeek01PruneTotalSec = 0.0;


#if USE_CUDA
void uploadTrainingImagesToGPU(Mat imgArr[][2])
{
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        gImgArr[i][0].upload(imgArr[i][0]);
        gImgArr[i][1].upload(imgArr[i][1]);
        gTarImgArr[i].upload(imgArr[i][1]);
    }
}
#endif

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
#if WEEK01_ENABLE_GA
    if (t == THRESHOLD || t == ERODE || t == DILATE || t == SOBEL_X || t == SOBEL_Y) {
        return false;
    }
    else {
        return true;
    }
#else
    return true;
#endif
}

FilterType randomFunctionType()
{
    vector<FilterType> funcs = {
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
#if ENABLE_CC_FILTER
        CC_FILTER,
#endif
        BITWISE_AND,
        BITWISE_OR,
        BITWISE_NOT,
        BITWISE_XOR
    };

    return funcs[rand_int(0, (int)funcs.size() - 1)];
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

    /*
    int t_idx = rand_int(1, NUM_TYPE_FUNC);
    FilterType t = static_cast<FilterType>(t_idx);
    */
    FilterType t = randomFunctionType();

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

//Mat executeContourProcessCPU(const Mat& srcMask, const vector<double>& params) { // Waiting for Module-Testing
//    Mat maskImg = srcMask.clone();
//    Mat contourInput;
//#if FOREGROUND_WHITE
//    bitwise_not(maskImg, contourInput);
//#else
//    contourInput = maskImg;
//#endif
//
//    int kk = params.size() > 0 ? int(params[0]) : 1;
//    int k = ((kk) / 2) | 1;
//    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
//    int times = params.size() > 1 ? int(params[1]) / 2 : 0;
//    for (int t = 0; t < times; ++t) {
//        erode(contourInput, contourInput, kernel);
//    }
//    vector<vector<Point>> contours;
//    findContours(contourInput, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
//    Mat mask(contourInput.rows, contourInput.cols, CV_8UC1, Scalar(BG_PIXEL));
//    int selType = 0;
//    if (params.size() > 2) {
//        selType = min(2, int(params[2] / 5));
//    }
//    for (const auto& contour : contours) {
//        Rect bb = boundingRect(contour);
//        double aspect_ratio = double(bb.width) / double(bb.height + 1e-9);
//        if (selType == 0) {
//            int range = params.size() > 3 ? int(params[3]) / 2 : 0;
//            if (aspect_ratio >= (1 - range * 0.1) && aspect_ratio <= (1 + range * 0.1)) {
//                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(FG_PIXEL), -1);
//            }
//        }
//        else if (selType == 1) {
//            int areaTh = params.size() > 4 ? int(params[4]) : 1;
//            if (contourArea(contour) >= 100 * areaTh) {
//                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(FG_PIXEL), -1);
//            }
//        }
//        else {
//            int range = params.size() > 3 ? int(params[3]) : 0;
//            int areaTh = params.size() > 4 ? int(params[4]) : 1;
//            if (aspect_ratio >= (1 - range * 0.1) && aspect_ratio <= (1 + range * 0.1) && contourArea(contour) >= 100 * areaTh) {
//                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(FG_PIXEL), -1);
//            }
//        }
//    }
//#if FOREGROUND_WHITE
//    bitwise_not(mask, mask);
//#endif
//    return mask;
//}

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
        // case CONTOUR_PROCESS:    return "CONTOUR_PROCESS";
    case CC_FILTER:          return "CC_FILTER";
    case BITWISE_AND:        return "BITWISE_AND";
    case BITWISE_OR:         return "BITWISE_OR";
    case BITWISE_NOT:        return "BITWISE_NOT";
    case BITWISE_XOR:        return "BITWISE_XOR";
    default:                 return "UNKNOWN";
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
               //case CONTOUR_PROCESS: {
               //    Mat child = executeTree(node->children[0], input);
               //    return executeContourProcessCPU(child, node->params);
               //}
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

cv::Ptr<cv::cuda::Filter> getGaussianFilter(int srcType, int dstType, int ksize, double sigma) {
    GaussianKey key;
    key.srcType = srcType;
    key.dstType = dstType;
    key.ksize = ksize;
    key.sigma100 = (int)(sigma * 100);
    {
        // std::lock_guard<std::mutex> lock(gGaussianMutex);
        auto it = gGaussianCache.find(key);
        if (it != gGaussianCache.end()) {
            return it->second;
        }
        auto filter = cv::cuda::createGaussianFilter(srcType, dstType, cv::Size(ksize, ksize), sigma);
        gGaussianCache[key] = filter;
        return filter;
    }
}

cv::Ptr<cv::cuda::Filter> getMedianFilter(int srcType, int ksize) {
    MedianKey key;
    key.srcType = srcType;
    key.ksize = ksize;
    {
        // std::lock_guard<std::mutex> lock(gMedianMutex);
        auto it = gMedianCache.find(key);
        if (it != gMedianCache.end())
        {
            return it->second;
        }
        auto filter = cv::cuda::createMedianFilter(srcType, ksize);
        gMedianCache[key] = filter;
        return filter;
    }
}

cv::Ptr<cv::cuda::Filter> getBoxFilter(int srcType, int dstType, int ksize) {
    BoxKey key;
    key.srcType = srcType;
    key.dstType = dstType;
    key.ksize = ksize;

    {
        // std::lock_guard<std::mutex> lock(gBoxMutex);

        auto it = gBoxCache.find(key);
        if (it != gBoxCache.end())
        {
            return it->second;
        }

        auto filter = cv::cuda::createBoxFilter(srcType, dstType, cv::Size(ksize, ksize));
        gBoxCache[key] = filter;
        return filter;
    }
}

cv::Ptr<cv::cuda::Filter> getSobelFilter(int srcType, int dstType, int dx, int dy, int ksize) {
    SobelKey key;
    key.srcType = srcType;
    key.dstType = dstType;
    key.dx = dx;
    key.dy = dy;
    key.ksize = ksize;

    {
        // std::lock_guard<std::mutex> lock(gSobelMutex);

        auto it = gSobelCache.find(key);
        if (it != gSobelCache.end())
        {
            return it->second;
        }

        auto filter = cv::cuda::createSobelFilter(srcType, dstType, dx, dy, ksize);
        gSobelCache[key] = filter;
        return filter;
    }
}

cv::Ptr<cv::cuda::CannyEdgeDetector> getCannyEdgeDetector(double t1, double t2) {
    CannyKey key;
    key.t1 = (int)t1;
    key.t2 = (int)t2;
    {
        // std::lock_guard<std::mutex> lock(gCannyMutex);
        auto it = gCannyCache.find(key);
        if (it != gCannyCache.end())
        {
            return it->second;
        }
        auto filter = cv::cuda::createCannyEdgeDetector(t1, t2);
        gCannyCache[key] = filter;
        return filter;
    }
}

cv::Ptr<cv::cuda::Filter> getMorphologyFilter(int op, int srcType, int ksize) {
    MorphologyKey key;
    key.op = op;
    key.srcType = srcType;
    key.ksize = ksize;
    {
        // std::lock_guard<std::mutex> lock(gMorphologyMutex);
        auto it = gMorphologyCache.find(key);
        if (it != gMorphologyCache.end())
        {
            return it->second;
        }
        Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
        auto filter = cv::cuda::createMorphologyFilter(op, srcType, kernel);
        gMorphologyCache[key] = filter;
        return filter;
    }
}

cv::cuda::GpuMat fallbackCPU(
    const shared_ptr<TreeNode>& node,
    const cv::cuda::GpuMat& input)
{
    Mat cpuInput;
    input.download(cpuInput, getCudaStream());
    getCudaStream().waitForCompletion();

    Mat cpuRes = executeTree(node, cpuInput);

    cv::cuda::GpuMat gpuRes;
    gpuRes.upload(cpuRes, getCudaStream());
    getCudaStream().waitForCompletion();

    return gpuRes;
}

cv::cuda::GpuMat executeCCFilterCUDA(
    const cv::cuda::GpuMat& src,
    const std::vector<double>& params
);

#if USE_CUDA
struct CudaEvalContext
{
    std::vector<cv::cuda::GpuMat> keepAlive;

    cv::cuda::GpuMat hold(const cv::cuda::GpuMat& m)
    {
        if (!m.empty()) {
            keepAlive.push_back(m);
        }
        return m;
    }
};
#endif

cv::cuda::GpuMat executeTreeCUDA(const shared_ptr<TreeNode>& node, const cv::cuda::GpuMat& input, CudaEvalContext& ctx)
{
    if (!node)
    {
        cv::cuda::GpuMat dst;
        input.copyTo(dst, getCudaStream());
        return ctx.hold(dst);
    }
    try
    {
        switch (node->type)
        {
        case TERMINAL_INPUT:
        {
            // Keep terminal copy on the same CUDA stream used by the rest of executeTreeCUDA.
            cv::cuda::GpuMat dst;
            input.copyTo(dst, getCudaStream());
            return ctx.hold(dst);
        }
        case GAUSSIAN_BLUR:
        {
            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            int k =
                max(1,
                    int(node->params.size() > 0 ?
                        int(node->params[0]) : 3));
            if ((k % 2) == 0)
                k |= 1;
            double sigma =
                node->params.size() > 1 ?
                node->params[1] : 1.5;
            cv::cuda::GpuMat dst;
            auto filter =
                getGaussianFilter(
                    child.type(),
                    child.type(),
                    k,
                    sigma);

            // filter->apply(child, dst, gCudaStream);
            filter->apply(child, dst, getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case MED_BLUR:
        {
            //#if CUDA_EQ_TEST_MED
            //            return fallbackCPU(node, input);
            //#endif

            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            int k =
                max(1,
                    int(node->params.size() > 0 ?
                        int(node->params[0]) : 3));
            if ((k % 2) == 0)
                k |= 1;
            cv::cuda::GpuMat dst;

            /*
            auto filter =
                cv::cuda::createMedianFilter(
                    child.type(),
                    k);
            */
            auto filter = getMedianFilter(child.type(), k);

            filter->apply(child, dst, getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case BLUR:
        {
            //#if CUDA_EQ_TEST_BLUR
            //            return fallbackCPU(node, input);
            //#endif

            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            int k =
                max(1,
                    int(node->params.size() > 0 ?
                        int(node->params[0]) : 3));
            if ((k % 2) == 0)
                k |= 1;
            cv::cuda::GpuMat dst;
            /*
            auto filter =
                cv::cuda::createBoxFilter(
                    child.type(),
                    child.type(),
                    Size(k, k));
            */
            auto filter = getBoxFilter(child.type(), child.type(), k);
            filter->apply(child, dst, getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case BILATERAL_FILTER:
        {
            //#if CUDA_EQ_TEST_BILATERAL
            //            return fallbackCPU(node, input);
            //#endif

            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);

            int d = node->params.size() > 0 ? int(node->params[0]) : 9;
            double sigmaColor = node->params.size() > 1 ? node->params[1] : 75.0;
            double sigmaSpace = node->params.size() > 2 ? node->params[2] : 75.0;

            // d must be positive and preferably odd
            d = std::max(1, d);
            if ((d % 2) == 0)
                d |= 1;

            // Avoid extremely large GP-generated parameters
            d = std::min(d, 31);
            sigmaColor = std::max(1.0, sigmaColor);
            sigmaSpace = std::max(1.0, sigmaSpace);

            cv::cuda::GpuMat dst;

            cv::cuda::bilateralFilter(
                child,
                dst,
                d,
                sigmaColor,
                sigmaSpace,
                cv::BORDER_DEFAULT,
                getCudaStream()
            );

            // return dst;
            return ctx.hold(dst);
        }
        //case BILATERAL_FILTER: {
        //    auto child = executeTreeCUDA(node->children[0], input);
        //    Mat cpuMask;
        //    child.download(cpuMask, getCudaStream());
        //    getCudaStream().waitForCompletion();
        //    Mat cpuRes = executeTree(node, cpuMask);
        //    cv::cuda::GpuMat gpuRes;
        //    gpuRes.upload(cpuRes, getCudaStream());
        //    return gpuRes;
        //}
        case SOBEL_X:
        {
            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            int k =
                max(1,
                    int(node->params.size() > 0 ?
                        int(node->params[0]) : 3));
            if ((k % 2) == 0)
                k |= 1;
            cv::cuda::GpuMat grad16;
            cv::cuda::GpuMat abs16;
            cv::cuda::GpuMat grad8;
            /*
            auto sobel =
                cv::cuda::createSobelFilter(
                    CV_8UC1,
                    CV_16S,
                    1,
                    0,
                    k);
            */
            auto sobel = getSobelFilter(CV_8UC1, CV_16S, 1, 0, k);
            sobel->apply(child, grad16, getCudaStream());
            cv::cuda::absdiff(
                grad16,
                cv::Scalar::all(0),
                abs16,
                getCudaStream());
            abs16.convertTo(grad8, CV_8U, getCudaStream());
            ctx.hold(grad16);
            ctx.hold(abs16);
            // return grad8;
            return ctx.hold(grad8);
        }
        case SOBEL_Y:
        {
            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            int k =
                max(1,
                    int(node->params.size() > 0 ?
                        int(node->params[0]) : 3));
            if ((k % 2) == 0)
                k |= 1;
            cv::cuda::GpuMat grad16;
            cv::cuda::GpuMat abs16;
            cv::cuda::GpuMat grad8;
            /*
            auto sobel =
                cv::cuda::createSobelFilter(
                    CV_8UC1,
                    CV_16S,
                    0,
                    1,
                    k);
            */
            auto sobel = getSobelFilter(CV_8UC1, CV_16S, 0, 1, k);
            sobel->apply(child, grad16, getCudaStream());
            cv::cuda::absdiff(
                grad16,
                cv::Scalar::all(0),
                abs16,
                getCudaStream());
            abs16.convertTo(grad8, CV_8U, getCudaStream());
            ctx.hold(grad16);
            ctx.hold(abs16);
            // return grad8;
            return ctx.hold(grad8);
        }
        case CANNY:
        {
            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            double t1 =
                node->params.size() > 0 ?
                node->params[0] : 100.0;
            double t2 =
                node->params.size() > 1 ?
                node->params[1] : 200.0;
            cv::cuda::GpuMat dst;
            /*
            auto canny =
                cv::cuda::createCannyEdgeDetector(
                    t1,
                    t2);
            */
            auto canny = getCannyEdgeDetector(t1, t2);
            canny->detect(child, dst, getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case DIFF_PROCESS:
        {
            //#if CUDA_EQ_TEST_DIFF
            //            return fallbackCPU(node, input);
            //#endif

            auto a = executeTreeCUDA(node->children[0], input, ctx);
            auto b = executeTreeCUDA(node->children[1], input, ctx);

            cv::cuda::GpuMat dst;

            cv::cuda::subtract(
                a,
                b,
                dst,
                cv::noArray(),
                CV_8U,
                getCudaStream());

            // return dst;
            return ctx.hold(dst);
        }
        case THRESHOLD:
        {
            auto child =
                executeTreeCUDA(
                    node->children[0],
                    input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            double th =
                node->params.size() > 0 ?
                node->params[0] : 127.0;
            cv::cuda::GpuMat dst;
            cv::cuda::threshold(child, dst, th, 255, THRESH_BINARY, getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case ERODE:
        {
            //#if CUDA_EQ_TEST_ERODE
            //            return fallbackCPU(node, input);
            //#endif

            auto child =
                executeTreeCUDA(
                    node->children[0],
                    input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            int r =
                node->params.size() > 0 ?
                int(node->params[0]) : 1;
            int k = 1 + 2 * max(0, r);
            /*
            Mat kernel =
                getStructuringElement(
                    MORPH_RECT,
                    Size(k, k));
            cv::cuda::GpuMat dst;
            auto filter =
                cv::cuda::createMorphologyFilter(
                    MORPH_ERODE,
                    child.type(),
                    kernel);
            */
            cv::cuda::GpuMat dst;
            auto filter = getMorphologyFilter(MORPH_ERODE, child.type(), k);
            filter->apply(child, dst, getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case DILATE:
        {
            //#if CUDA_EQ_TEST_DILATE
            //            return fallbackCPU(node, input);
            //#endif

            auto child =
                executeTreeCUDA(
                    node->children[0],
                    input, ctx);
            CV_Assert(child.type() == CV_8UC1);
            int r =
                node->params.size() > 0 ?
                int(node->params[0]) : 1;
            int k = 1 + 2 * max(0, r);
            /*
            Mat kernel =
                getStructuringElement(
                    MORPH_RECT,
                    Size(k, k));
            cv::cuda::GpuMat dst;
            auto filter =
                cv::cuda::createMorphologyFilter(
                    MORPH_DILATE,
                    child.type(),
                    kernel);
            */
            cv::cuda::GpuMat dst;
            auto filter = getMorphologyFilter(MORPH_DILATE, child.type(), k);
            filter->apply(child, dst, getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case CC_FILTER:
        {
            auto child = executeTreeCUDA(node->children[0], input, ctx);
            CV_Assert(child.type() == CV_8UC1);

            // executeCCFilterCUDA is implemented with raw CUDA kernels.
            // Wait for the OpenCV CUDA stream before handing its child output to that path.
            getCudaStream().waitForCompletion();

            cv::cuda::GpuMat dst = executeCCFilterCUDA(child, node->params);

            // Keep this synchronization for the stable training baseline:
            // it guarantees that the returned GpuMat is ready before downstream OpenCV CUDA operations.
            cudaDeviceSynchronize();

            return ctx.hold(dst);
        }
        case BITWISE_AND:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input, ctx);
            auto b =
                executeTreeCUDA(
                    node->children[1],
                    input, ctx);
            CV_Assert(a.type() == b.type());
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_and(a, b, dst, cv::noArray(), getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case BITWISE_OR:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input, ctx);
            auto b =
                executeTreeCUDA(
                    node->children[1],
                    input, ctx);
            CV_Assert(a.type() == b.type());
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_or(a, b, dst, cv::noArray(), getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case BITWISE_XOR:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input, ctx);
            auto b =
                executeTreeCUDA(
                    node->children[1],
                    input, ctx);
            CV_Assert(a.type() == b.type());
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_xor(a, b, dst, cv::noArray(), getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        case BITWISE_NOT:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input, ctx);
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_not(a, dst, cv::noArray(), getCudaStream());
            // return dst;
            return ctx.hold(dst);
        }
        /*
            CONTOUR_PROCESS
            Remain Calculating in CPU
        */
        //case CONTOUR_PROCESS: {
        //    auto child = executeTreeCUDA(node->children[0], input);
        //    Mat cpuMask;
        //    child.download(cpuMask, getCudaStream());
        //    getCudaStream().waitForCompletion();
        //    Mat cpuRes = executeContourProcessCPU(cpuMask, node->params);
        //    cv::cuda::GpuMat gpuRes;
        //    gpuRes.upload(cpuRes, getCudaStream());
        //    return gpuRes;
        //}
        default:
        {
            CV_Error(
                cv::Error::StsNotImplemented,
                ("CUDA path not implemented: " +
                    filterTypeToString(node->type)).c_str());
        }
        }
    }
    catch (const cv::Exception& e)
    {
        cerr << "[CUDA ERROR] " << e.what() << endl;
        CV_Error(cv::Error::StsError, e.what());
    }
}

// =====================================================
// Node collection & utilities
// =====================================================
using NodeWithParent = pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>>;

void collectNodesWithParents(const shared_ptr<TreeNode>& node, const shared_ptr<TreeNode>& parent, vector<NodeWithParent>& result) {
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

MetricsGPU calcMetricsOneGPU(
    const cv::cuda::GpuMat& pred,
    const cv::cuda::GpuMat& gt)
{
    MetricsGPU m;

    cv::cuda::GpuMat pred0;
    cv::cuda::GpuMat pred255;

    cv::cuda::GpuMat gt0;
    cv::cuda::GpuMat gt255;

    cv::cuda::GpuMat tpMask;
    cv::cuda::GpuMat fpMask;
    cv::cuda::GpuMat fnMask;

    //----------------------------------
    // compare
    //----------------------------------

    cv::cuda::compare(
        pred,
        cv::Scalar(0),
        pred0,
        cv::CMP_EQ,
        getCudaStream());

    cv::cuda::compare(
        pred,
        cv::Scalar(255),
        pred255,
        cv::CMP_EQ,
        getCudaStream());

    cv::cuda::compare(
        gt,
        cv::Scalar(0),
        gt0,
        cv::CMP_EQ,
        getCudaStream());

    cv::cuda::compare(
        gt,
        cv::Scalar(255),
        gt255,
        cv::CMP_EQ,
        getCudaStream());

    cv::cuda::GpuMat& predFG =
        (FG_PIXEL == 255) ? pred255 : pred0;

    cv::cuda::GpuMat& predBG =
        (FG_PIXEL == 255) ? pred0 : pred255;

    cv::cuda::GpuMat& gtFG =
        (FG_PIXEL == 255) ? gt255 : gt0;

    cv::cuda::GpuMat& gtBG =
        (FG_PIXEL == 255) ? gt0 : gt255;

    //----------------------------------
    // TP
    //----------------------------------

    cv::cuda::bitwise_and(
        predFG,
        gtFG,
        tpMask,
        cv::noArray(),
        getCudaStream());

    //----------------------------------
    // FP
    //----------------------------------

    cv::cuda::bitwise_and(
        predFG,
        gtBG,
        fpMask,
        cv::noArray(),
        getCudaStream());

    //----------------------------------
    // FN
    //----------------------------------

    cv::cuda::bitwise_and(
        predBG,
        gtFG,
        fnMask,
        cv::noArray(),
        getCudaStream());

    //----------------------------------
    // sync
    //----------------------------------

    getCudaStream().waitForCompletion();

    //----------------------------------
    // count
    //----------------------------------

    m.tp =
        cv::cuda::countNonZero(tpMask);

    m.fp =
        cv::cuda::countNonZero(fpMask);

    m.fn =
        cv::cuda::countNonZero(fnMask);

    return m;
}

double calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], int numInd) {
    double f1_score[TRAIN_SIZE];
    for (int idxSet = 0; idxSet < TRAIN_SIZE; idxSet++) {
        int tp = 0, fp = 0, fn = 0;
        CV_Assert(metaImg_g[idxSet].size() == tarImg_g[idxSet].size());
        for (int i = 0; i < metaImg_g[idxSet].rows; i++) {
            for (int j = 0; j < metaImg_g[idxSet].cols; j++) {
                if (metaImg_g[idxSet].at<uchar>(i, j) != 0 && metaImg_g[idxSet].at<uchar>(i, j) != 255) {
                    // non-binary reached (shouldn't after thresholding) => penalize
                    return 0.01;
                }
                if (metaImg_g[idxSet].at<uchar>(i, j) == FG_PIXEL && tarImg_g[idxSet].at<uchar>(i, j) == FG_PIXEL) tp += 1;
                if (metaImg_g[idxSet].at<uchar>(i, j) == FG_PIXEL && tarImg_g[idxSet].at<uchar>(i, j) == BG_PIXEL) fp += 1;
                if (metaImg_g[idxSet].at<uchar>(i, j) == BG_PIXEL && tarImg_g[idxSet].at<uchar>(i, j) == FG_PIXEL) fn += 1;
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

    for (int idxSet = 0; idxSet < TRAIN_SIZE; idxSet++) {
        if (numInd != -1) { // in the last generation
            indFValInfo[numInd][idxSet] = f1_score[idxSet];
        }
        sum_f1 += f1_score[idxSet];
    }

    if (numInd != -1) {
        indFValInfo[numInd][TRAIN_SIZE] = sum_f1;
    }

    // store in indFValInfo externally if needed by caller
    return sum_f1;
}

bool isBinaryImageGPU(const cv::cuda::GpuMat& img)
{
    cv::cuda::GpuMat eq0;
    cv::cuda::GpuMat eq255;
    cv::cuda::GpuMat validMask;

    cv::cuda::compare(
        img,
        cv::Scalar(0),
        eq0,
        cv::CMP_EQ,
        getCudaStream());

    cv::cuda::compare(
        img,
        cv::Scalar(255),
        eq255,
        cv::CMP_EQ,
        getCudaStream());

    cv::cuda::bitwise_or(
        eq0,
        eq255,
        validMask,
        cv::noArray(),
        getCudaStream());

    getCudaStream().waitForCompletion();

    int validCount =
        cv::cuda::countNonZero(
            validMask);

    int totalPixels =
        img.rows *
        img.cols;

    return
        validCount ==
        totalPixels;
}

// calScoreByInd: execute tree for each set, force binary by thresholding if needed
double calScoreByInd(const shared_ptr<TreeNode>& node, Mat imgArr[][2], int numInd) {
    Mat tarImg[TRAIN_SIZE];
    Mat resImg[TRAIN_SIZE];
    for (int i = 0; i < TRAIN_SIZE; i++) {
        tarImg[i] = imgArr[i][1];
    }
    for (int i = 0; i < TRAIN_SIZE; i++) {
        resImg[i] = executeTree(node, imgArr[i][0]);
    }
    return calculateMetrics(resImg, tarImg, numInd);
}

#if USE_CUDA
double calScoreByIndCUDA_old(
    const shared_ptr<TreeNode>& node,
    Mat imgArr[][2],
    int numInd)
{
    double sum_f1 = 0.0;

    for (int idxSet = 0; idxSet < TRAIN_SIZE; idxSet++) {
        CudaEvalContext ctx;

        cv::cuda::GpuMat resImg =
            executeTreeCUDA(node, gImgArr[idxSet][0], ctx);

        ctx.hold(resImg);

        if (!isBinaryImageGPU(resImg)) {
            return 0.01;
        }

        MetricsGPU m =
            calcMetricsOneGPU(resImg, gTarImgArr[idxSet]);

        long long tp = m.tp;
        long long fp = m.fp;
        long long fn = m.fn;

        if (tp == 0) tp++;
        if (fp == 0) fp++;
        if (fn == 0) fn++;

        double precision = double(tp) / double(tp + fp);
        double recall = double(tp) / double(tp + fn);
        double f1 = calculateF1Score(precision, recall);

        if (numInd != -1) {
            indFValInfo[numInd][idxSet] = f1;
        }

        sum_f1 += f1;
    }

    if (numInd != -1) {
        indFValInfo[numInd][TRAIN_SIZE] = sum_f1;
    }

    return sum_f1;
}
#endif

#if USE_CUDA
double calScoreByIndCUDA_fused(
    const shared_ptr<TreeNode>& node,
    Mat imgArr[][2],
    int numInd)
{
    double sum_f1 = 0.0;

    for (int idxSet = 0; idxSet < TRAIN_SIZE; idxSet++) {
        CudaEvalContext ctx;

        cv::cuda::GpuMat resImg =
            executeTreeCUDA(node, gImgArr[idxSet][0], ctx);

        ctx.hold(resImg);

        MetricsGPUFused m =
            calcMetricsOneGPUFused(
                resImg,
                gTarImgArr[idxSet],
                FG_PIXEL,
                getCudaStream());

        if (m.invalid > 0) {
            return 0.01;
        }

        long long tp = m.tp;
        long long fp = m.fp;
        long long fn = m.fn;

        if (tp == 0) tp++;
        if (fp == 0) fp++;
        if (fn == 0) fn++;

        double precision = double(tp) / double(tp + fp);
        double recall = double(tp) / double(tp + fn);
        double f1 = calculateF1Score(precision, recall);

        if (numInd != -1) {
            indFValInfo[numInd][idxSet] = f1;
        }

        sum_f1 += f1;
    }

    if (numInd != -1) {
        indFValInfo[numInd][TRAIN_SIZE] = sum_f1;
    }

    return sum_f1;
}
#endif

#if USE_CUDA
double calScoreByIndCUDA(
    const shared_ptr<TreeNode>& node,
    Mat imgArr[][2],
    int numInd)
{
    auto _week01_score_t0 = std::chrono::high_resolution_clock::now();

#if USE_FUSED_METRICS
    double _week01_score = calScoreByIndCUDA_fused(node, imgArr, numInd);
#else
    double _week01_score = calScoreByIndCUDA_old(node, imgArr, numInd);
#endif

    auto _week01_score_t1 = std::chrono::high_resolution_clock::now();
    gCalScoreCallCount++;
    gCalScoreTotalSec +=
        std::chrono::duration<double>(_week01_score_t1 - _week01_score_t0).count();

    return _week01_score;
}
#endif

genType getCurGenInfo(vector<shared_ptr<TreeNode>>& population, Mat imgArr[][2]) {
    auto _week01_getcur_t0 = std::chrono::high_resolution_clock::now();
    double scoreArr[POP_SIZE];

#if USE_CUDA
    for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
        scoreArr[idxInd] = CAL_SCORE(population[idxInd], imgArr, -1);
    }
#else
#pragma omp parallel for
    for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
        scoreArr[idxInd] = CAL_SCORE(population[idxInd], imgArr, -1);
    }
#endif

    double minFValue = scoreArr[0];
    double maxFValue = scoreArr[0];
    double sumFValue = 0.0;
    double variance = 0.0;
    int localEliteIdx = 0;

    for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
        double tmp = scoreArr[idxInd];
        sumFValue += tmp;

        if (tmp > maxFValue) {
            maxFValue = tmp;
            localEliteIdx = idxInd;
        }

        if (tmp < minFValue) {
            minFValue = tmp;
        }
    }

    double aveFValue = sumFValue / POP_SIZE;

    for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
        double diff = scoreArr[idxInd] - aveFValue;
        variance += diff * diff;
    }

    genType curGenInfo;
    curGenInfo.eliteIndex = localEliteIdx;
    curGenInfo.eliteTree = cloneTree(population[localEliteIdx]);
    curGenInfo.eliteFValue = maxFValue;
    curGenInfo.genMinFValue = minFValue;
    curGenInfo.genAveFValue = aveFValue;
    curGenInfo.genDevFValue = sqrt(variance / POP_SIZE);

    auto _week01_getcur_t1 = std::chrono::high_resolution_clock::now();
    gGetCurGenInfoCalls++;
    gGetCurGenInfoTotalSec +=
        std::chrono::duration<double>(_week01_getcur_t1 - _week01_getcur_t0).count();

    return curGenInfo;
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
               //case CONTOUR_PROCESS: {
               //    switch (idxParam) {
               //    case 0: {
               //        int kk = node->params.size() > 0 ? int(node->params[0]) : 1;
               //        int k = ((kk) / 2) | 1;
               //        return (double)k;
               //    }
               //    case 1: {
               //        int times = node->params.size() > 1 ? int(node->params[1]) / 2 : 0;
               //        return (double)times;
               //    }
               //    case 2: {
               //        int selType = 0;
               //        if (node->params.size() > 2) selType = min(2, int(node->params[2] / 5));
               //        return (double)selType;
               //    }
               //    case 3: {
               //        int range = node->params.size() > 3 ? int(node->params[3]) / 2 : 0;
               //        return (double)range;
               //    }
               //    case 4: {
               //        int areaTh = node->params.size() > 4 ? int(node->params[4]) : 1;
               //        return (double)areaTh;
               //    }
               //    default:
               //        return -1.0;
               //    }
               //}

    case CC_FILTER: {
        switch (idxParam) {
        case 0: {
            int minArea = node->params.size() > 0 ? max(1, int(node->params[0])) : 1;
            return (double)minArea;
        }
        case 1: {
            float aspectRange = node->params.size() > 1 ? max(0.0f, (float)node->params[1]) : 0.0f;
            return (double)aspectRange;
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

void printTree(const shared_ptr<TreeNode>& node, int depth = 0, FILE* fpSys = nullptr, FILE* fpRead = nullptr) {
    if (!node) return;
    for (int i = 0; i < depth; ++i) {
        fprintf(fpSys, "    ");
        fprintf(fpRead, "    ");
    }
    fprintf(fpSys, "%s", filterTypeToString(node->type).c_str());
    fprintf(fpRead, "%s", filterTypeToString(node->type).c_str());
    if (!node->params.empty()) {
        fprintf(fpSys, " (");
        fprintf(fpRead, " (");
        for (size_t i = 0; i < node->params.size(); i++) {
            fprintf(fpSys, "%.2f", node->params[i]);
            fprintf(fpRead, "%.2f", getFinalParamVal(node, i));
            if (i + 1 < node->params.size()) {
                fprintf(fpSys, ", ");
                fprintf(fpRead, ", ");
            }
        }
        fprintf(fpSys, ")");
        fprintf(fpRead, ")");
    }
    fprintf(fpSys, "\n");
    fprintf(fpRead, "\n");
    for (const auto& c : node->children) printTree(c, depth + 1, fpSys, fpRead);
}

FilterType stringToFilterType(const string& s)
{
    static unordered_map<string, FilterType> mp = {
        {"TERMINAL_INPUT", TERMINAL_INPUT},
        {"GAUSSIAN_BLUR", GAUSSIAN_BLUR},
        {"MED_BLUR", MED_BLUR},
        {"BLUR", BLUR},
        {"BILATERAL_FILTER", BILATERAL_FILTER},
        {"SOBEL_X", SOBEL_X},
        {"SOBEL_Y", SOBEL_Y},
        {"CANNY", CANNY},
        {"DIFF_PROCESS", DIFF_PROCESS},
        {"THRESHOLD", THRESHOLD},
        {"ERODE", ERODE},
        {"DILATE", DILATE},
        // {"CONTOUR_PROCESS", CONTOUR_PROCESS},
        {"CC_FILTER", CC_FILTER},
        {"BITWISE_AND", BITWISE_AND},
        {"BITWISE_OR", BITWISE_OR},
        {"BITWISE_NOT", BITWISE_NOT},
        {"BITWISE_XOR", BITWISE_XOR}
    };

    if (mp.count(s))
        return mp[s];

    return TERMINAL_INPUT;
}

string trim(const string& str)
{
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

shared_ptr<TreeNode> parseTreeLine(const string& line) {
    auto node = make_shared<TreeNode>();
    string content = trim(line);
    if (content.empty())
    {
        throw runtime_error("Empty line.");
    }
    size_t p1 = content.find('(');
    // None params
    if (p1 == string::npos)
    {
        node->type = stringToFilterType(content);
        return node;
    }
    size_t p2 = content.find(')', p1);
    if (p2 == string::npos)
    {
        throw runtime_error("Missing ')' in line: " + content);
    }
    string typeStr = trim(content.substr(0, p1));
    if (typeStr.empty())
    {
        throw runtime_error("Empty filter type.");
    }
    // Transfer type string to enum
    node->type = stringToFilterType(typeStr);
    string paramStr = content.substr(p1 + 1, p2 - p1 - 1);
    paramStr = trim(paramStr);
    stringstream ss(paramStr);
    string token;
    while (getline(ss, token, ','))
    {
        token = trim(token);
        if (token.empty())
        {
            throw runtime_error("Empty parameter.");
        }
        try
        {
            node->params.push_back(stod(token));
        }
        catch (const exception&)
        {
            throw runtime_error("Invalid parameter: " + token);
        }
    }
    return node;
}

shared_ptr<TreeNode> loadTreeFromFile(const string& filename)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        cerr << "Cannot open tree file: " << filename << endl;
        return nullptr;
    }
    // For storing the depth of nodes and the pointer of the node.
    vector<pair<int, shared_ptr<TreeNode>>> depthStack;
    // line: the content of current line.
    string line;
    shared_ptr<TreeNode> root = nullptr;
    while (getline(fin, line))
    {
        if (line.empty())
            continue;
        int leadingSpaces = 0;
        while (leadingSpaces < (int)line.size() && line[leadingSpaces] == ' ')
        {
            leadingSpaces++;
        }
        if (leadingSpaces % 4 != 0)
        {
            cerr << "Invalid indentation\n";
        }
        int depth = leadingSpaces / 4;
        auto node = parseTreeLine(line);
        if (!node)
        {
            cerr << "Parse error\n";
            continue;
        }
        // root
        if (depth == 0)
        {
            root = node;
            depthStack.clear();
            depthStack.push_back({ depth, node });
            continue;
        }
        // find parentNode and add current node as child
        while (!depthStack.empty() && depthStack.back().first >= depth)
        {
            depthStack.pop_back();
        }
        if (!depthStack.empty())
        {
            auto parent = depthStack.back().second;
            parent->children.push_back(node);
        }
        depthStack.push_back({ depth, node });
    }
    fin.close();
    confirmDepth(root);
    return root;
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
            int numParams = g_paramDesc[target->type].n;
            target->params.resize(numParams);
            for (int i = 0; i < numParams; ++i) {
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
// Intron Pruning Utilities
// =====================================================

// collect subtree nodes
void collectSubtreeNodes(const shared_ptr<TreeNode>& node, vector<shared_ptr<TreeNode>>& out) {
    if (!node) return;
    out.push_back(node);
    for (auto& c : node->children) {
        collectSubtreeNodes(c, out);
    }
}
// PruneResult
// void pruneIntrons(shared_ptr<TreeNode>& root, Mat imgArr[][2])
PruneResult pruneIntrons(shared_ptr<TreeNode>& root, Mat imgArr[][2])
{
    vector<NodeWithParent> allNodes;
    collectNodesWithParents(root, nullptr, allNodes);
    if (allNodes.size() <= 2)
        return { false, 0.0 };
    int idxN1 = rand_real() < 0.5 ? rand_int(0, (int)allNodes.size() - 1) : rand_int(1, (int)allNodes.size() - 1);
    auto N1 = allNodes[idxN1].first;
    auto parentN1 = allNodes[idxN1].second;
    vector<shared_ptr<TreeNode>> subtreeNodes;
    collectSubtreeNodes(N1, subtreeNodes);
    if (subtreeNodes.size() <= 1)
        return { false, 0.0 };
    int idxN2 = rand_int(1, (int)subtreeNodes.size() - 1);
    auto N2 = subtreeNodes[idxN2];
    double scoreBefore = CAL_SCORE(root, imgArr, -1);
    auto rootPruned = cloneTree(root);
    vector<NodeWithParent> clonedNodes;
    collectNodesWithParents(rootPruned, nullptr, clonedNodes);
    auto clonedN1 = clonedNodes[idxN1].first;
    auto clonedParentN1 = clonedNodes[idxN1].second;
    vector<shared_ptr<TreeNode>> clonedSubtreeNodes;
    collectSubtreeNodes(clonedN1, clonedSubtreeNodes);
    auto clonedN2 = clonedSubtreeNodes[idxN2];
    if (!clonedParentN1)
    {
        rootPruned = cloneTree(clonedN2);
    }
    else
    {
        for (auto& c : clonedParentN1->children)
        {
            if (c == clonedN1)
            {
                c = cloneTree(clonedN2);
                break;
            }
        }
    }
    confirmDepth(rootPruned);
    double scoreAfter = CAL_SCORE(rootPruned, imgArr, -1);
    if (scoreAfter >= scoreBefore)
    {
        root = rootPruned;
        return { true, scoreAfter };
    }
    return { false, 0.0 };
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
        pop[idxInd].fit = CAL_SCORE(rootCloned, imgArr, -1);
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
            child.fit = CAL_SCORE(rootCloned, imgArr, -1);
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

void debugRescoreDrop(
    const vector<genType>& genInfo,
    Mat imgArr[][2],
    int numGen,
    FILE* fpDebug)
{
    if (genInfo.size() < 2) return;

    const genType& prevGen = genInfo[genInfo.size() - 2];
    const genType& curGen = genInfo[genInfo.size() - 1];

    const double eps = 1e-9;

    if (curGen.eliteFValue + eps >= prevGen.eliteFValue) {
        return;
    }

    gEliteDropCount++;

    printf("\n[DROP-DETECTED] gen=%d, prevElite=%.10f, curElite=%.10f\n",
        numGen + 1,
        prevGen.eliteFValue,
        curGen.eliteFValue);

    if (fpDebug) {
        fprintf(fpDebug,
            "\n[DROP-DETECTED] gen=%d, prevElite=%.10f, curElite=%.10f\n",
            numGen + 1,
            prevGen.eliteFValue,
            curGen.eliteFValue);
    }

    printf("[RE-SCORE] previous elite tree:\n");
    if (fpDebug) fprintf(fpDebug, "[RE-SCORE] previous elite tree:\n");

    for (int r = 0; r < 10; r++) {
        double s = CAL_SCORE(prevGen.eliteTree, imgArr, -1);
        printf("  prevElite run %02d: %.10f\n", r, s);
        if (fpDebug) fprintf(fpDebug, "  prevElite run %02d: %.10f\n", r, s);
    }

    printf("[RE-SCORE] current elite tree:\n");
    if (fpDebug) fprintf(fpDebug, "[RE-SCORE] current elite tree:\n");

    for (int r = 0; r < 10; r++) {
        double s = CAL_SCORE(curGen.eliteTree, imgArr, -1);
        printf("  curElite  run %02d: %.10f\n", r, s);
        if (fpDebug) fprintf(fpDebug, "  curElite  run %02d: %.10f\n", r, s);
    }

    if (fpDebug) fflush(fpDebug);
}


// =====================================================
// Top-level GP+GA loop
// =====================================================
void multiProcess(Mat imgArr[][2]) {
#if USE_CUDA
    omp_set_num_threads(1);
#endif

    // Initialize parameter ranges and safe default values.
    initParamDesc();
    initParamDesc_safeVal();

    printf("\n[WEEK01-CONFIG] label=%s TRAIN_SIZE=%d GENERATIONS=%d POP_SIZE=%d MAX_DEPTH=%d USE_CUDA=%d USE_FUSED_METRICS=%d ENABLE_CC_FILTER=%d WEEK01_ENABLE_GA=%d WEEK01_ENABLE_PRUNING=%d RANDOM_SEED=%d\n",
        WEEK01_RUN_LABEL,
        TRAIN_SIZE,
        GENERATIONS,
        POP_SIZE,
        MAX_DEPTH + 1,
        USE_CUDA,
        USE_FUSED_METRICS,
        ENABLE_CC_FILTER,
        WEEK01_ENABLE_GA,
        WEEK01_ENABLE_PRUNING,
        RANDOM_SEED);
    printf("[WEEK01-CONFIG] GA_POP=%d GA_GENERATIONS=%d GA_TRIGGER_MEAN_THRESH=%.6f INITIAL_BIAS_THRESHOLD=%.6f NUM_DELAY_GA=%d\n\n",
        GA_POP,
        GA_GENERATIONS,
        GA_TRIGGER_MEAN_THRESH,
        INITIAL_BIAS_THRESHOLD,
        NUM_DELAY_GA);

    auto week01TrainStart = std::chrono::high_resolution_clock::now();

    Mat resImg[TRAIN_SIZE];
    Mat tarImg[TRAIN_SIZE];

    char imgName_pro[TRAIN_SIZE][256];
    char imgName_final[TRAIN_SIZE][256];

    FILE* fl_fValue = nullptr;
    errno_t err = fopen_s(&fl_fValue, "./imgs_0710_2026_v1/output/train_output/f_value.txt", "w");
    if (err != 0 || fl_fValue == nullptr) {
        perror("Cannot open the file");
        // continue without file
    }

    FILE* fl_maxFval = nullptr;
    errno_t err2 = fopen_s(&fl_maxFval, "./imgs_0710_2026_v1/output/train_output/maxFvalInfo_final.txt", "w");
    if (err2 != 0 || fl_maxFval == nullptr) {
        perror("Cannot open the file");
    }

    FILE* fl_printTree_sys = nullptr;
    errno_t err3 = fopen_s(&fl_printTree_sys, "./imgs_0710_2026_v1/output/train_output/printed_tree_sys.txt", "w");
    if (err3 != 0 || fl_printTree_sys == nullptr) {
        perror("Cannot open the file");
    }

    FILE* fl_printTree_read = nullptr;
    errno_t err4 = fopen_s(&fl_printTree_read, "./imgs_0710_2026_v1/output/train_output/printed_tree_read.txt", "w");
    if (err4 != 0 || fl_printTree_read == nullptr) {
        perror("Cannot open the file");
    }

    FILE* fl_logOptiGA = nullptr;
    errno_t err5 = fopen_s(&fl_logOptiGA, "./imgs_0710_2026_v1/output/train_output/log_opti_ga.txt", "a");
    if (err5 != 0 || fl_logOptiGA == nullptr) {
        perror("Cannot open the file");
    }

    FILE* fl_logPrune = nullptr;
    errno_t err6 = fopen_s(&fl_logPrune, "./imgs_0710_2026_v1/output/train_output/log_prune.txt", "a");
    if (err6 != 0 || fl_logPrune == nullptr) {
        perror("Cannot open the file");
    }

    FILE* fl_debugScore = nullptr;
    errno_t err7 = fopen_s(&fl_debugScore, "./imgs_0710_2026_v1/output/train_output/debug_rescore.txt", "w");
    if (err7 != 0 || fl_debugScore == nullptr) {
        perror("Cannot open debug_rescore.txt");
    }

    //initParamDesc();
    //initParamDesc_safeVal();

    // protection lifetime: if GA optimized an index at gen G, set protectedUntil[idx]=G+1
    vector<int> protectedUntil(POP_SIZE, -1);

    bool flag_tri_GA = false;
    int idx_opt_GA = -1;
    double fitness_opt_GA = 0.0;

    vector<genType> genInfo;
    vector<shared_ptr<TreeNode>> population;
    population.reserve(POP_SIZE);

    for (int i = 0; i < POP_SIZE; ++i) population.push_back(generateRandomTree());

    /*
    vector<string> eliteTreeFiles = {
        "./imgs_0710_2026_v1/input/elite_trees/best_01_7.17.txt",
        "./imgs_0710_2026_v1/input/elite_trees/best_02_6.95.txt",
        "./imgs_0710_2026_v1/input/elite_trees/best_03_6.82.txt"
    };
    for (const auto& file : eliteTreeFiles)
    {
        auto tree = loadTreeFromFile(file);
        printf("score: %.4f\n", CAL_SCORE(tree, imgArr, -1));
        if (tree)
        {
            mutate(tree);
            population.push_back(tree);
            cout << "[Seed Init] Loaded elite tree: " << file << endl;
        }
    }
    while ((int)population.size() < POP_SIZE)
    {
        population.push_back(generateRandomTree());
    }
    */

    double biasThreshold = INITIAL_BIAS_THRESHOLD;

    for (int numGen = 0; numGen < GENERATIONS; numGen++) {
        printf("---------generation: %d---------\n", numGen + 1);

        /*
          Break-Point-02
        */

        // select two parents that are NOT protected for this generation
        int idx1 = rand_int(0, POP_SIZE - 1);
        int idx2 = rand_int(0, POP_SIZE - 1);
        auto choose_nonprotected = [&](int avoidGen)->int {
            int tries = 0;
            while (tries < 50) {
                int c = rand_int(0, POP_SIZE - 1);
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
        double score1 = CAL_SCORE(parent1, imgArr, -1);
        double score2 = CAL_SCORE(parent2, imgArr, -1);

        family.push_back({ score1, parent1 });
        family.push_back({ score2, parent2 });

        for (int k = 0; k < OFFSPRING_COUNT; ++k) {
            auto childA = cloneTree(parent1);
            auto childB = cloneTree(parent2);
            crossover(childA, childB);
            auto chosen = (rand_real() < 0.5) ? childA : childB;
            double fit = CAL_SCORE(chosen, imgArr, -1);
            family.push_back({ fit, chosen });
        }

        for (int idxInd = 0; idxInd < (OFFSPRING_COUNT + 2); idxInd++)
        {
            if (rand_real() < MUTATION_RATE)
            {
                mutate(family[idxInd].second);
                family[idxInd].first = CAL_SCORE(family[idxInd].second, imgArr, -1);
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

        debugRescoreDrop(genInfo, imgArr, numGen, fl_debugScore);

        /*
            Pruning Module
            Week-01 default: disabled.
            Enable only when calibrating pruning overhead.
        */
#if WEEK01_ENABLE_PRUNING
        {
            double eliteMeanFForPrune =
                genInfo[numGen].eliteFValue / double(TRAIN_SIZE);

            if (eliteMeanFForPrune >= PRUNE_TRIGGER_MEAN_THRESH &&
                bias < biasThreshold &&
                rand_real() < 0.2)
            {
                auto _week01_prune_t0 =
                    std::chrono::high_resolution_clock::now();

                auto pruneRes =
                    pruneIntrons(population[genInfo[numGen].eliteIndex], imgArr);

                auto _week01_prune_t1 =
                    std::chrono::high_resolution_clock::now();

                double pruneSec =
                    std::chrono::duration<double>(
                        _week01_prune_t1 - _week01_prune_t0).count();

                gWeek01PruneAttemptCount++;
                gWeek01PruneTotalSec += pruneSec;
                countPruneSum += 1;

                if (pruneRes.success)
                {
                    printf("------[Pruning-Success] Intron subtree removed.------\n");
                    printf("scoreBeforePruning: %.10f, scoreAfterPruning: %.10f, sec=%.3f\n",
                        genInfo[numGen].eliteFValue,
                        pruneRes.scoreAfter,
                        pruneSec);
                    printf("-----------------------------------------------------\n");

                    countPruneSuccess += 1;
                    gWeek01PruneSuccessCount++;

                    if (fl_logPrune) {
                        fprintf(fl_logPrune, "%d: success sec=%.6f\n", numGen + 1, pruneSec);
                    }

                    genInfo.back() = getCurGenInfo(population, imgArr);
                    bias = calcBias(genInfo);
                }
                else {
                    printf("------[Pruning-Fail] No intron subtree was found or accepted. sec=%.3f------\n",
                        pruneSec);

                    countPruneFail += 1;
                    gWeek01PruneFailCount++;

                    if (fl_logPrune) {
                        fprintf(fl_logPrune, "%d: fail sec=%.6f\n", numGen + 1, pruneSec);
                    }
                }
            }
        }
#endif

        /*
          Break-Point-03
        */
        printf("(Res-GP)the idx of eliteInd: %d, the fitness of GP: %.4f, the bias: %.4f\n", genInfo[numGen].eliteIndex, genInfo[numGen].eliteFValue, bias);

        // ---- Trigger GA when bias is low ----
        if (numGen != GENERATIONS - 1) {
#if WEEK01_ENABLE_GA
            double eliteMeanFForGA =
                genInfo[numGen].eliteFValue / double(TRAIN_SIZE);

            if (eliteMeanFForGA >= GA_TRIGGER_MEAN_THRESH &&
                bias < biasThreshold)
            {
                // "first time of GA_Trigger" or "GA-Module succeeded in last Trigger"
                // or "the delay-gen after a GA failure is over".
                if (lastGenFailGA < 0 || (lastGenFailGA + NUM_DELAY_GA) < numGen) {
                    printf("[GA-TRIGGER] gen=%d eliteSum=%.10f eliteMean=%.10f bias=%.6f threshold=%.6f\n",
                        numGen + 1,
                        genInfo[numGen].eliteFValue,
                        eliteMeanFForGA,
                        bias,
                        biasThreshold);

                    gGATriggerCount++;

                    auto gaStart =
                        std::chrono::high_resolution_clock::now();

                    auto eliteIndex = genInfo[numGen].eliteIndex;
                    auto eliteTreeClone = cloneTree(population[eliteIndex]);

                    auto bestParams = runGrayGA_forTree(eliteTreeClone, imgArr);

                    auto gaEnd =
                        std::chrono::high_resolution_clock::now();

                    double gaSec =
                        std::chrono::duration<double>(gaEnd - gaStart).count();

                    gGATotalSec += gaSec;
                    if (gaSec > gGABestSingleSec) {
                        gGABestSingleSec = gaSec;
                    }

                    if (!bestParams.empty()) {
                        // write back decoded params into eliteTreeClone then into population[eliteIndex]
                        vector<shared_ptr<TreeNode>> paramNodes;
                        collectParams(eliteTreeClone, paramNodes);
                        int pos = 0;
                        for (auto& n : paramNodes) {
                            for (auto& p : n->params) {
                                if (pos < (int)bestParams.size()) {
                                    p = bestParams[pos++];
                                }
                            }
                        }

                        double scoreAfterGA =
                            CAL_SCORE(eliteTreeClone, imgArr, -1);

                        if (scoreAfterGA <= genInfo[numGen].eliteFValue) {
                            printf("[GA-FAIL] gen=%d before=%.10f after=%.10f gaSec=%.3f\n",
                                numGen + 1,
                                genInfo[numGen].eliteFValue,
                                scoreAfterGA,
                                gaSec);

                            if (fl_logOptiGA) {
                                fprintf(fl_logOptiGA,
                                    "[GA-FAIL] Gen=%d before=%.10f after=%.10f gaSec=%.6f\n",
                                    numGen + 1,
                                    genInfo[numGen].eliteFValue,
                                    scoreAfterGA,
                                    gaSec);
                            }

                            gGAFailCount++;
                            if (flag_tri_GA) {
                                flag_tri_GA = false;
                            }

                            // when GA module failed to improve fitness, GA-delay is triggered.
                            lastGenFailGA = numGen;
                        }
                        else {
                            // replace population's elite with optimized tree
                            population[eliteIndex] = cloneTree(eliteTreeClone);

                            flag_tri_GA = true;
                            idx_opt_GA = eliteIndex;
                            fitness_opt_GA = scoreAfterGA;

                            printf("[GA-SUCCESS] gen=%d idx=%d before=%.10f after=%.10f gaSec=%.3f\n",
                                numGen + 1,
                                idx_opt_GA,
                                genInfo[numGen].eliteFValue,
                                fitness_opt_GA,
                                gaSec);

                            if (fl_logOptiGA) {
                                fprintf(fl_logOptiGA,
                                    "[GA-SUCCESS] Gen=%d idx=%d before=%.10f after=%.10f gaSec=%.6f\n",
                                    numGen + 1,
                                    idx_opt_GA,
                                    genInfo[numGen].eliteFValue,
                                    fitness_opt_GA,
                                    gaSec);
                            }

                            gGASuccessCount++;

                            // protect this index in the next generation.
                            protectedUntil[eliteIndex] = numGen + 1;

                            printf("[GA-WRITEBACK] eliteIndex=%d protectedUntilGen=%d\n",
                                eliteIndex,
                                numGen + 2);

                            biasThreshold *= BIAS_DECAY;

                            if (lastGenFailGA >= 0) {
                                lastGenFailGA = -1;
                            }

                            // Refresh generation statistics after writeback.
                            genInfo.back() = getCurGenInfo(population, imgArr);
                            bias = calcBias(genInfo);
                        }
                    }
                    else {
                        gGAEmptyCount++;
                        printf("[GA-EMPTY] gen=%d No optimizable params. gaSec=%.3f\n",
                            numGen + 1,
                            gaSec);
                    }
                }
                else {
                    gGABlockedCount++;
                    printf("[GA-BLOCKED] gen=%d blocked by NUM_DELAY_GA. lastFailGen=%d\n",
                        numGen + 1,
                        lastGenFailGA + 1);

                    if (flag_tri_GA) {
                        flag_tri_GA = false;
                    }
                }
            }
            else {
                if (flag_tri_GA) {
                    flag_tri_GA = false;
                }
            }
#else
            if (flag_tri_GA) {
                flag_tri_GA = false;
            }
#endif
        }
        else {
            // final generation: record indFValInfo
#if USE_CUDA
            for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
                CAL_SCORE(population[idxInd], imgArr, idxInd);
            }
#else
#pragma omp parallel for
            for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
                CAL_SCORE(population[idxInd], imgArr, idxInd);
            }
#endif
        }

        if ((numGen == 0) ||
            ((numGen + 1) % WEEK01_LOG_INTERVAL == 0) ||
            (numGen == GENERATIONS - 1))
        {
            auto week01Now = std::chrono::high_resolution_clock::now();
            double elapsedSec =
                std::chrono::duration<double>(
                    week01Now - week01TrainStart).count();

            double secPerGen =
                elapsedSec / double(numGen + 1);

            double genPerMin =
                secPerGen > 0.0 ? 60.0 / secPerGen : 0.0;

            double curElite =
                genInfo.empty() ? 0.0 : genInfo.back().eliteFValue;

            printf("[WEEK01-PROGRESS] gen=%d/%d elapsedSec=%.3f secPerGen=%.6f genPerMin=%.6f eliteSum=%.10f eliteMean=%.10f calScoreCalls=%lld GA(T/S/F/B/E)=%d/%d/%d/%d/%d\n",
                numGen + 1,
                GENERATIONS,
                elapsedSec,
                secPerGen,
                genPerMin,
                curElite,
                curElite / double(TRAIN_SIZE),
                gCalScoreCallCount,
                gGATriggerCount,
                gGASuccessCount,
                gGAFailCount,
                gGABlockedCount,
                gGAEmptyCount);
        }
    } // end generation

    auto week01TrainEnd = std::chrono::high_resolution_clock::now();
    double week01TotalSec =
        std::chrono::duration<double>(
            week01TrainEnd - week01TrainStart).count();

    printf("---------------- GEN-END --------------\n");

    // curThreshFVal = indFValInfo[curMaxFvalIdx][TRAIN_SIZE];
    printTree(genInfo.back().eliteTree, 0, fl_printTree_sys, fl_printTree_read);
    fprintf(fl_logPrune, "sum-prune: %d, success: %d, fail: %d\n", countPruneSum, countPruneSuccess, countPruneFail);
    // write f-values
    for (size_t i = 0; i < genInfo.size(); ++i) {
        if (fl_fValue) fprintf(fl_fValue, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
    }
    if (fl_maxFval) {
        // for (int i = 0; i <= TRAIN_SIZE; i++) fprintf(fl_maxFval, "%.4f ", indFValInfo[curMaxFvalIdx][i]);
        for (int i = 0; i <= TRAIN_SIZE; i++) fprintf(fl_maxFval, "%.4f ", indFValInfo[genInfo.back().eliteIndex][i]);
        fprintf(fl_maxFval, "\n");
    }
    // save final images for elite of last generation
    for (int idxSet = 0; idxSet < TRAIN_SIZE; idxSet++) {

        Mat res;
#if USE_CUDA
        CudaEvalContext ctx;

        cv::cuda::GpuMat gres =
            executeTreeCUDA(genInfo.back().eliteTree, gImgArr[idxSet][0], ctx);
        ctx.hold(gres);

        getCudaStream().waitForCompletion();
        gres.download(res);
#else
        res = executeTree(genInfo.back().eliteTree, imgArr[idxSet][0]);
#endif

        sprintf_s(imgName_pro[idxSet],
            "./imgs_0710_2026_v1/output/train_output/resImgs/crack_%05d.png",
            IMG_START_IDX + idxSet);
        imwrite(imgName_pro[idxSet], res);
    }

    printf("\n[TRAIN-SUMMARY] EliteDropCount=%d\n", gEliteDropCount);
    if (fl_debugScore) {
        fprintf(fl_debugScore, "\n[TRAIN-SUMMARY] EliteDropCount=%d\n", gEliteDropCount);
        fflush(fl_debugScore);
    }

    double week01SecPerGen =
        week01TotalSec / double(GENERATIONS);
    double week01GenPerMin =
        week01SecPerGen > 0.0 ? 60.0 / week01SecPerGen : 0.0;
    double finalEliteSum =
        genInfo.empty() ? 0.0 : genInfo.back().eliteFValue;
    double finalEliteMean =
        finalEliteSum / double(TRAIN_SIZE);

    printf("[WEEK01-SUMMARY] label=%s TRAIN_SIZE=%d GENERATIONS=%d totalSec=%.3f secPerGen=%.6f genPerMin=%.6f finalEliteSum=%.10f finalEliteMean=%.10f EliteDropCount=%d\n",
        WEEK01_RUN_LABEL,
        TRAIN_SIZE,
        GENERATIONS,
        week01TotalSec,
        week01SecPerGen,
        week01GenPerMin,
        finalEliteSum,
        finalEliteMean,
        gEliteDropCount);

    printf("[CAL-SCORE-SUMMARY] calls=%lld totalSec=%.3f avgSec=%.9f\n",
        gCalScoreCallCount,
        gCalScoreTotalSec,
        gCalScoreCallCount > 0 ? gCalScoreTotalSec / double(gCalScoreCallCount) : 0.0);

    printf("[GETCURGEN-SUMMARY] calls=%d totalSec=%.3f avgSec=%.9f\n",
        gGetCurGenInfoCalls,
        gGetCurGenInfoTotalSec,
        gGetCurGenInfoCalls > 0 ? gGetCurGenInfoTotalSec / double(gGetCurGenInfoCalls) : 0.0);

    printf("[GA-SUMMARY] enabled=%d trigger=%d success=%d fail=%d blocked=%d empty=%d totalSec=%.3f avgTriggerSec=%.6f maxTriggerSec=%.6f\n",
        WEEK01_ENABLE_GA,
        gGATriggerCount,
        gGASuccessCount,
        gGAFailCount,
        gGABlockedCount,
        gGAEmptyCount,
        gGATotalSec,
        gGATriggerCount > 0 ? gGATotalSec / double(gGATriggerCount) : 0.0,
        gGABestSingleSec);

    printf("[PRUNE-SUMMARY] enabled=%d attempt=%d success=%d fail=%d totalSec=%.3f avgSec=%.6f\n",
        WEEK01_ENABLE_PRUNING,
        gWeek01PruneAttemptCount,
        gWeek01PruneSuccessCount,
        gWeek01PruneFailCount,
        gWeek01PruneTotalSec,
        gWeek01PruneAttemptCount > 0 ? gWeek01PruneTotalSec / double(gWeek01PruneAttemptCount) : 0.0);

    {
        bool needHeader = false;
        {
            std::ifstream checkCsv(WEEK01_SUMMARY_CSV);
            needHeader = !checkCsv.good() || checkCsv.peek() == std::ifstream::traits_type::eof();
        }

        std::ofstream csv(WEEK01_SUMMARY_CSV, std::ios::app);
        if (csv.is_open()) {
            if (needHeader) {
                csv << "label,train_size,generations,pop_size,max_depth,use_cuda,use_fused_metrics,enable_cc_filter,week01_enable_ga,week01_enable_pruning,random_seed,total_sec,sec_per_gen,gen_per_min,final_elite_sum,final_elite_mean,elite_drop_count,cal_score_calls,cal_score_total_sec,cal_score_avg_sec,getcur_calls,getcur_total_sec,getcur_avg_sec,ga_trigger,ga_success,ga_fail,ga_blocked,ga_empty,ga_total_sec,ga_avg_trigger_sec,ga_max_trigger_sec,prune_attempt,prune_success,prune_fail,prune_total_sec\n";
            }

            csv << WEEK01_RUN_LABEL << ","
                << TRAIN_SIZE << ","
                << GENERATIONS << ","
                << POP_SIZE << ","
                << MAX_DEPTH + 1 << ","
                << USE_CUDA << ","
                << USE_FUSED_METRICS << ","
                << ENABLE_CC_FILTER << ","
                << WEEK01_ENABLE_GA << ","
                << WEEK01_ENABLE_PRUNING << ","
                << RANDOM_SEED << ","
                << week01TotalSec << ","
                << week01SecPerGen << ","
                << week01GenPerMin << ","
                << finalEliteSum << ","
                << finalEliteMean << ","
                << gEliteDropCount << ","
                << gCalScoreCallCount << ","
                << gCalScoreTotalSec << ","
                << (gCalScoreCallCount > 0 ? gCalScoreTotalSec / double(gCalScoreCallCount) : 0.0) << ","
                << gGetCurGenInfoCalls << ","
                << gGetCurGenInfoTotalSec << ","
                << (gGetCurGenInfoCalls > 0 ? gGetCurGenInfoTotalSec / double(gGetCurGenInfoCalls) : 0.0) << ","
                << gGATriggerCount << ","
                << gGASuccessCount << ","
                << gGAFailCount << ","
                << gGABlockedCount << ","
                << gGAEmptyCount << ","
                << gGATotalSec << ","
                << (gGATriggerCount > 0 ? gGATotalSec / double(gGATriggerCount) : 0.0) << ","
                << gGABestSingleSec << ","
                << gWeek01PruneAttemptCount << ","
                << gWeek01PruneSuccessCount << ","
                << gWeek01PruneFailCount << ","
                << gWeek01PruneTotalSec
                << "\n";
        }
        else {
            printf("[WEEK01-WARN] Cannot open summary CSV: %s\n", WEEK01_SUMMARY_CSV);
        }
    }

    if (fl_fValue) fclose(fl_fValue);
    if (fl_maxFval) fclose(fl_maxFval);
    if (fl_printTree_sys) fclose(fl_printTree_sys);
    if (fl_printTree_read) fclose(fl_printTree_read);
    if (fl_logOptiGA) fclose(fl_logOptiGA);
    if (fl_logPrune) fclose(fl_logPrune);
    if (fl_debugScore) fclose(fl_debugScore);
}

// =====================================================
// main
// =====================================================
int main(void) {
    // cout << "[DEBUG] Entered Main Function." << endl;
    Mat imgArr[TRAIN_SIZE][2];
    char inputPathName_ori[256];
    char inputPathName_tar[256];

    /*
    if (TRAIN_SIZE == 1) {
        sprintf_s(inputPathName_ori, "./imgs_0710_2026_v1/input/oriImg_0%d.png", idSet);
        sprintf_s(inputPathName_tar, "./imgs_0710_2026_v1/input/tarImg_0%d.png", idSet);
        imgArr[0][0] = imread(inputPathName_ori, 0);
        imgArr[0][1] = imread(inputPathName_tar, 0);
    }
    else {
        for (int i = 0; i < TRAIN_SIZE; i++) {
            sprintf_s(inputPathName_ori, "./imgs_0710_2026_v1/input/oriImg_0%d.png", i + 1);
            sprintf_s(inputPathName_tar, "./imgs_0710_2026_v1/input/tarImg_0%d.png", i + 1);
            imgArr[i][0] = imread(inputPathName_ori, 0);
            imgArr[i][1] = imread(inputPathName_tar, 0);
        }
    }
    */

    /*
    for (int i = 0; i < TRAIN_SIZE; i++) {
        sprintf_s(inputPathName_ori, "./imgs_0710_2026_v1/input/oriImg_0%d.png", i + 1);
        sprintf_s(inputPathName_tar, "./imgs_0710_2026_v1/input/tarImg_0%d.png", i + 1);
        imgArr[i][0] = imread(inputPathName_ori, 0);
        imgArr[i][1] = imread(inputPathName_tar, 0);
    }
    */


    for (int i = 0; i < TRAIN_SIZE; i++) {
        sprintf_s(inputPathName_ori, "./imgs_0710_2026_v1/input/train/positive/images/crack_%05d.png", IMG_START_IDX + i);
        sprintf_s(inputPathName_tar, "./imgs_0710_2026_v1/input/train/positive/masks/crack_%05d.png", IMG_START_IDX + i);
        imgArr[i][0] = imread(inputPathName_ori, 0);
        imgArr[i][1] = imread(inputPathName_tar, 0);
    }


    // basic checks
    for (int i = 0; i < TRAIN_SIZE; ++i) {
        if (imgArr[i][0].empty() || imgArr[i][1].empty()) {
            cerr << "Warning: image pair " << i << " could not be loaded. Check paths." << endl;
        }
    }

#if USE_CUDA
    uploadTrainingImagesToGPU(imgArr);
#endif

    multiProcess(imgArr);
    return 0;
}