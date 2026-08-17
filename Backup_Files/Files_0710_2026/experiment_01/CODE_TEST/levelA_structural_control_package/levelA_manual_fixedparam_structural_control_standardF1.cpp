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
#define TRAIN_SIZE 20
#endif

// =====================================================
// Week-03 mixed training set configuration
// =====================================================
// Domain code:
//   0 = thin / low-contrast cracks
//   1 = thick / high-contrast cracks
//
// The formal ViEW experiment no longer uses contiguous IMG_START_IDX+i
// for TS05/TS10/TS20.  Use explicit image IDs and robust domain-balanced fitness.
#ifndef USE_EXPLICIT_TRAIN_INDICES
#define USE_EXPLICIT_TRAIN_INDICES 1
#endif

#ifndef USE_DOMAIN_BALANCED_FITNESS
#define USE_DOMAIN_BALANCED_FITNESS 1
#endif

#ifndef USE_ROBUST_DOMAIN_FITNESS
#define USE_ROBUST_DOMAIN_FITNESS 1
#endif

#ifndef WORST_K_PER_DOMAIN
#define WORST_K_PER_DOMAIN 3
#endif

#ifndef DOMAIN_ROBUST_MEAN_WEIGHT
#define DOMAIN_ROBUST_MEAN_WEIGHT 0.70
#endif

#ifndef DOMAIN_ROBUST_WORSTK_WEIGHT
#define DOMAIN_ROBUST_WORSTK_WEIGHT 0.30
#endif

#ifndef FINAL_WORST_DOMAIN_WEIGHT
#define FINAL_WORST_DOMAIN_WEIGHT 0.60
#endif

#ifndef FINAL_DOMAIN_MEAN_WEIGHT
#define FINAL_DOMAIN_MEAN_WEIGHT 0.40
#endif

#define TRAIN_DOMAIN_THIN 0
#define TRAIN_DOMAIN_THICK 1

static_assert(TRAIN_SIZE == 20, "This mixed-set source file expects TRAIN_SIZE=20.");

static const int TRAIN_IMAGE_IDS[TRAIN_SIZE] = {
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130
};

static const int TRAIN_DOMAIN[TRAIN_SIZE] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

inline int getTrainImageId(int idxSet)
{
#if USE_EXPLICIT_TRAIN_INDICES
    return TRAIN_IMAGE_IDS[idxSet];
#else
    return IMG_START_IDX + idxSet;
#endif
}

inline int getTrainDomain(int idxSet)
{
    return TRAIN_DOMAIN[idxSet];
}

inline const char* getTrainDomainName(int idxSet)
{
    return (getTrainDomain(idxSet) == TRAIN_DOMAIN_THIN) ? "thin" : "thick";
}



// #define idSet 1 // for mark the selected set if the TRAIN_SIZE been set of 1

// GP parameters
#ifndef POP_SIZE
#define POP_SIZE 100 // Pop_Size of GP
#endif

// Week-01 default is 1000 generations for calibration.
// For the final ViEW-scale runs, override this to 10000.
#ifndef GENERATIONS
#define GENERATIONS 10000 // Week-03 TS20 mixed-robustdomain formal run
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
#define RANDOM_SEED 42
#endif

#ifndef STR_HELPER
#define STR_HELPER(x) #x
#endif
#ifndef STR
#define STR(x) STR_HELPER(x)
#endif

#ifndef WEEK03_OUTPUT_PREFIX
#define WEEK03_OUTPUT_PREFIX "week03_TS20_mixed_robustdomain_GP_G10000_seed" STR(RANDOM_SEED)
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
//   /DTRAIN_SIZE=20 /DGENERATIONS=1000 /DWEEK01_ENABLE_GA=1 /DRANDOM_SEED=42
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
#define WEEK01_RUN_LABEL WEEK03_OUTPUT_PREFIX
#endif

#ifndef WEEK01_SUMMARY_CSV
#define WEEK01_SUMMARY_CSV "./imgs_0710_2026_v1/output/train_output/" WEEK03_OUTPUT_PREFIX "_summary.csv"
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


// =====================================================
// Week-03 robust domain-balanced fitness diagnostics
// =====================================================
struct DomainFitnessStats {
    double simpleSum = 0.0;
    double sumThin = 0.0;
    double sumThick = 0.0;
    double meanAll = 0.0;
    double meanThin = 0.0;
    double meanThick = 0.0;

    double worstKMeanThin = 0.0;
    double worstKMeanThick = 0.0;
    double robustMeanThin = 0.0;
    double robustMeanThick = 0.0;

    double balancedMean = 0.0;
    double balancedScore = 0.0;

    double robustDomainMean = 0.0;
    double robustWorstDomain = 0.0;
    double robustFinalMean = 0.0;
    double robustScore = 0.0;

    int nThin = 0;
    int nThick = 0;
};

inline double meanOfWorstK(vector<double> values, int k)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    int kk = std::max(1, std::min(k, (int)values.size()));
    double s = 0.0;
    for (int i = 0; i < kk; i++) s += values[i];
    return s / double(kk);
}

DomainFitnessStats computeDomainFitnessStats(const double f1_score[])
{
    DomainFitnessStats st;
    vector<double> thinVals;
    vector<double> thickVals;

    for (int idxSet = 0; idxSet < TRAIN_SIZE; idxSet++) {
        st.simpleSum += f1_score[idxSet];
        if (getTrainDomain(idxSet) == TRAIN_DOMAIN_THIN) {
            st.sumThin += f1_score[idxSet];
            st.nThin++;
            thinVals.push_back(f1_score[idxSet]);
        }
        else {
            st.sumThick += f1_score[idxSet];
            st.nThick++;
            thickVals.push_back(f1_score[idxSet]);
        }
    }

    int nAll = st.nThin + st.nThick;
    st.meanAll = (nAll > 0) ? (st.simpleSum / double(nAll)) : 0.0;
    st.meanThin = (st.nThin > 0) ? (st.sumThin / double(st.nThin)) : st.meanAll;
    st.meanThick = (st.nThick > 0) ? (st.sumThick / double(st.nThick)) : st.meanAll;

    st.worstKMeanThin = (st.nThin > 0) ? meanOfWorstK(thinVals, WORST_K_PER_DOMAIN) : st.meanAll;
    st.worstKMeanThick = (st.nThick > 0) ? meanOfWorstK(thickVals, WORST_K_PER_DOMAIN) : st.meanAll;

    st.robustMeanThin =
        DOMAIN_ROBUST_MEAN_WEIGHT * st.meanThin +
        DOMAIN_ROBUST_WORSTK_WEIGHT * st.worstKMeanThin;

    st.robustMeanThick =
        DOMAIN_ROBUST_MEAN_WEIGHT * st.meanThick +
        DOMAIN_ROBUST_WORSTK_WEIGHT * st.worstKMeanThick;

    // Reference score: equal weighting of raw thin/thick means.
    st.balancedMean = 0.5 * st.meanThin + 0.5 * st.meanThick;
    st.balancedScore = st.balancedMean * double(TRAIN_SIZE);

    // Robust-domain score:
    //   1) Penalize weak samples inside each domain using worst-k mean.
    //   2) Penalize imbalance between domains using min(robustThin, robustThick).
    st.robustDomainMean = 0.5 * (st.robustMeanThin + st.robustMeanThick);
    st.robustWorstDomain = std::min(st.robustMeanThin, st.robustMeanThick);
    st.robustFinalMean =
        FINAL_WORST_DOMAIN_WEIGHT * st.robustWorstDomain +
        FINAL_DOMAIN_MEAN_WEIGHT * st.robustDomainMean;
    st.robustScore = st.robustFinalMean * double(TRAIN_SIZE);

    return st;
}

void printFitnessModeOnce()
{
    static bool printedFitnessMode = false;
    if (printedFitnessMode) return;
    printedFitnessMode = true;

    printf("[FITNESS-MODE] USE_EXPLICIT_TRAIN_INDICES=%d\n", USE_EXPLICIT_TRAIN_INDICES);
    printf("[FITNESS-MODE] USE_DOMAIN_BALANCED_FITNESS=%d\n", USE_DOMAIN_BALANCED_FITNESS);
    printf("[FITNESS-MODE] USE_ROBUST_DOMAIN_FITNESS=%d\n", USE_ROBUST_DOMAIN_FITNESS);
    printf("[FITNESS-MODE] WORST_K_PER_DOMAIN=%d\n", WORST_K_PER_DOMAIN);
    printf("[FITNESS-MODE] DOMAIN_ROBUST_MEAN_WEIGHT=%.3f DOMAIN_ROBUST_WORSTK_WEIGHT=%.3f\n",
        double(DOMAIN_ROBUST_MEAN_WEIGHT), double(DOMAIN_ROBUST_WORSTK_WEIGHT));
    printf("[FITNESS-MODE] FINAL_WORST_DOMAIN_WEIGHT=%.3f FINAL_DOMAIN_MEAN_WEIGHT=%.3f\n",
        double(FINAL_WORST_DOMAIN_WEIGHT), double(FINAL_DOMAIN_MEAN_WEIGHT));
#if USE_ROBUST_DOMAIN_FITNESS
    printf("[FITNESS-MODE] score = TRAIN_SIZE * { %.3f * min(robustThin, robustThick) + %.3f * 0.5 * (robustThin + robustThick) }\n",
        double(FINAL_WORST_DOMAIN_WEIGHT), double(FINAL_DOMAIN_MEAN_WEIGHT));
    printf("[FITNESS-MODE] robustDomain = %.3f * meanF1(domain) + %.3f * worst%dMeanF1(domain)\n",
        double(DOMAIN_ROBUST_MEAN_WEIGHT), double(DOMAIN_ROBUST_WORSTK_WEIGHT), WORST_K_PER_DOMAIN);
#elif USE_DOMAIN_BALANCED_FITNESS
    printf("[FITNESS-MODE] score = TRAIN_SIZE * (0.5 * meanF1(thin) + 0.5 * meanF1(thick))\n");
#else
    printf("[FITNESS-MODE] score = simple sum of per-image F1 over TRAIN_SIZE\n");
#endif
    printf("[FITNESS-MODE] thin domain: TRAIN_DOMAIN=0, thick domain: TRAIN_DOMAIN=1\n");
}

void printDomainFitnessCheck(const char* tag, const double f1_score[], double storedScore)
{
    DomainFitnessStats st = computeDomainFitnessStats(f1_score);

    printf("%s simpleSum=%.10f balancedScore=%.10f robustScore=%.10f storedScore=%.10f\n",
        tag, st.simpleSum, st.balancedScore, st.robustScore, storedScore);
    printf("%s meanAll=%.10f meanThin=%.10f meanThick=%.10f nThin=%d nThick=%d\n",
        tag, st.meanAll, st.meanThin, st.meanThick, st.nThin, st.nThick);
    printf("%s worst%dThin=%.10f worst%dThick=%.10f robustThin=%.10f robustThick=%.10f\n",
        tag, WORST_K_PER_DOMAIN, st.worstKMeanThin,
        WORST_K_PER_DOMAIN, st.worstKMeanThick,
        st.robustMeanThin, st.robustMeanThick);
    printf("%s robustWorstDomain=%.10f robustDomainMean=%.10f robustFinalMean=%.10f\n",
        tag, st.robustWorstDomain, st.robustDomainMean, st.robustFinalMean);
}

double finalizeFitnessFromPerImageF1(const double f1_score[])
{
    printFitnessModeOnce();
    DomainFitnessStats st = computeDomainFitnessStats(f1_score);

#if USE_ROBUST_DOMAIN_FITNESS
    // Stronger validation objective for mixed thin/thick TS20:
    // equalize domains and penalize low-F1 samples inside each domain.
    return st.robustScore;
#elif USE_DOMAIN_BALANCED_FITNESS
    // Keep the original score scale close to "sum of F1 over TRAIN_SIZE",
    // while giving the thin and thick domains equal weight.
    return st.balancedScore;
#else
    return st.simpleSum;
#endif
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

    double finalScore = finalizeFitnessFromPerImageF1(f1_score);

    if (numInd != -1) {
        for (int idxSet = 0; idxSet < TRAIN_SIZE; idxSet++) {
            indFValInfo[numInd][idxSet] = f1_score[idxSet];
        }
        indFValInfo[numInd][TRAIN_SIZE] = finalScore;
    }

    return finalScore;
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
    double f1_score[TRAIN_SIZE];

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

        f1_score[idxSet] = f1;

        if (numInd != -1) {
            indFValInfo[numInd][idxSet] = f1;
        }
    }

    double finalScore = finalizeFitnessFromPerImageF1(f1_score);

    if (numInd != -1) {
        indFValInfo[numInd][TRAIN_SIZE] = finalScore;
    }

    return finalScore;
}
#endif


#if USE_CUDA
double calScoreByIndCUDA_fused(
    const shared_ptr<TreeNode>& node,
    Mat imgArr[][2],
    int numInd)
{
    double f1_score[TRAIN_SIZE];

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

        f1_score[idxSet] = f1;

        if (numInd != -1) {
            indFValInfo[numInd][idxSet] = f1;
        }
    }

    double finalScore = finalizeFitnessFromPerImageF1(f1_score);

    if (numInd != -1) {
        indFValInfo[numInd][TRAIN_SIZE] = finalScore;
    }

    return finalScore;
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
// Level A — parameter-matched structural-control baseline evaluator
// =====================================================
//
// PURPOSE:
//   Compare manually specified image-processing topologies against the
//   current fixed-parameter GP configuration while matching the operator
//   parameter values exactly.
//
// LEVEL-A FIXED PARAMETERS (same GP safe/default presets):
//   GAUSSIAN_BLUR (5, 1.5)
//   MED_BLUR      (19)
//   SOBEL_X/Y     (3)
//   CANNY         (100, 200)
//   THRESHOLD     (9)
//   ERODE         (1)
//   DILATE        (1)
//   CC_FILTER     (5, 3)
//
// No grid search, parameter tuning, or candidate selection is performed.
//
// M0/M3 include BITWISE_NOT as part of the previously defined manual
// dark-intensity topology.  Numeric parameters remain exactly GP-matched.
//
// NOTE:
//   The wider primary-200 experiment has already been inspected before
//   this additional Level-A control is executed. Therefore report Level A
//   as a post-hoc/supplementary structural-control analysis using
//   pre-existing topology definitions and GP parameter presets.
// =====================================================
//
// Two modes are provided:
//
//   1) TRAINING SANITY CHECK (safe default)
//      IDs 111-130, 20 images.
//      Purpose is implementation validation only:
//        - tree loads correctly,
//        - CUDA pipeline runs,
//        - output is binary,
//        - foreground polarity / path mistakes can be detected.
//      DO NOT use this mode for repeated manual parameter optimization.
//
//   2) PRIMARY-200 EVALUATION
//      IDs 211-410, 200 unseen images.
//      Requires explicit command-line confirmation:
//          --mode primary200 --confirm-levelA-fixed
//
// The four handcrafted baselines use the SAME TreeNode parser,
// executeTreeCUDA(), deterministic CC_FILTER implementation,
// fused TP/FP/FN metric and STANDARD F1 as the GP evaluator.
//
// CSV schemas intentionally match the current GP 200-image evaluator:
//   run_id,train_size,seed,...
// Baselines use train_size=0 and seed=0 because they are deterministic,
// non-evolutionary methods.
//
// Tree files expected under:
//   ./imgs_0710_2026_v1/input/test/baseline_tree
//
// Baseline topology and parameters are predeclared outside the test set.
// Do not modify topology/parameters after inspecting primary-200 results.
// =====================================================

#include <iomanip>
#include <numeric>
#include <cerrno>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifndef BASELINE_TREE_DIR
#define BASELINE_TREE_DIR "./imgs_0710_2026_v1/input/test/levelA_fixedparam_tree"
#endif

#ifndef BASELINE_TRAIN_IMAGE_DIR
#define BASELINE_TRAIN_IMAGE_DIR "./imgs_0710_2026_v1/input/train/positive/images"
#endif

#ifndef BASELINE_TRAIN_MASK_DIR
#define BASELINE_TRAIN_MASK_DIR "./imgs_0710_2026_v1/input/train/positive/masks"
#endif

#ifndef BASELINE_TEST_IMAGE_DIR
#define BASELINE_TEST_IMAGE_DIR "./imgs_0710_2026_v1/input/test/images"
#endif

#ifndef BASELINE_TEST_MASK_DIR
#define BASELINE_TEST_MASK_DIR "./imgs_0710_2026_v1/input/test/masks"
#endif

#ifndef BASELINE_OUTPUT_ROOT
#define BASELINE_OUTPUT_ROOT "./imgs_0710_2026_v1/output/traditional_baseline/levelA_fixedparam"
#endif

// 1 = save prediction PNGs.
// Only 80 sanity predictions and 800 primary-test predictions are produced.
#ifndef SAVE_BASELINE_RESULT_IMAGES
#define SAVE_BASELINE_RESULT_IMAGES 1
#endif

// Safe default if no command-line mode is supplied:
// 0 = training sanity, 1 = primary200.
// Keeping this at 0 helps avoid accidental primary-test inspection.
#ifndef BASELINE_DEFAULT_PRIMARY200
#define BASELINE_DEFAULT_PRIMARY200 0
#endif

static const int BASELINE_TRAIN_START_ID = 111;
static const int BASELINE_TRAIN_END_ID = 130;
static const int BASELINE_TRAIN_SIZE = 20;

static const int BASELINE_TEST_START_ID = 211;
static const int BASELINE_TEST_END_ID = 410;
static const int BASELINE_TEST_SIZE = 200;

static_assert(
    BASELINE_TRAIN_END_ID - BASELINE_TRAIN_START_ID + 1 == BASELINE_TRAIN_SIZE,
    "Baseline training sanity range must be 111-130."
);

static_assert(
    BASELINE_TEST_END_ID - BASELINE_TEST_START_ID + 1 == BASELINE_TEST_SIZE,
    "Baseline primary test range must be 211-410."
);

struct BaselineSpec
{
    std::string runId;
    std::string pipelineName;
    std::string treeFile;
};

struct BaselineDatasetSpec
{
    std::string modeName;
    std::string imageDir;
    std::string maskDir;
    std::string outputDir;
    std::string resultImageDir;
    std::string runSummaryPath;
    std::string perImagePath;
    int startId = 0;
    int endId = -1;
    int size = 0;
    bool isPrimary200 = false;
};

struct BaselineF1Stats
{
    double simpleSum = 0.0;
    double meanAll = 0.0;
    double stdAll = 0.0;  // population SD across images, matching GP evaluator
    double minF1 = 0.0;
    double maxF1 = 0.0;
};

#ifdef _WIN32
inline int baseline_mkdir_one_level(const std::string& path)
{
    return _mkdir(path.c_str());
}
#else
inline int baseline_mkdir_one_level(const std::string& path)
{
    return mkdir(path.c_str(), 0777);
}
#endif

bool baselineCreateDirectoriesRecursive(const std::string& rawPath)
{
    if (rawPath.empty()) return false;

    std::string path = rawPath;
    std::replace(path.begin(), path.end(), '\\', '/');

    std::string cur;

    for (size_t i = 0; i < path.size(); ++i) {
        const char ch = path[i];
        cur.push_back(ch);

        if (ch == '/' || i + 1 == path.size()) {
            if (cur.empty() ||
                cur == "/" ||
                cur == "./" ||
                cur == "../")
            {
                continue;
            }

            std::string dir = cur;

            if (!dir.empty() && dir.back() == '/') {
                dir.pop_back();
            }

            if (dir.empty() || dir == "." || dir == "..") {
                continue;
            }

            const int ret = baseline_mkdir_one_level(dir);

            if (ret != 0 && errno != EEXIST) {
                std::cerr
                    << "[WARN] Cannot create directory: "
                    << dir
                    << " errno="
                    << errno
                    << std::endl;
                return false;
            }
        }
    }

    return true;
}

std::vector<BaselineSpec> buildBaselineSpecs()
{
    std::vector<BaselineSpec> v;

    v.push_back({
        "levelA_M0_fixedparam",
        "M0 GaussianBlur->Threshold->BITWISE_NOT",
        std::string(BASELINE_TREE_DIR) +
        "/levelA_M0_fixedparam_printed_tree_sys.txt"
    });

    v.push_back({
        "levelA_M1_fixedparam",
        "M1 GaussianBlur->Canny->Dilate->CC_FILTER",
        std::string(BASELINE_TREE_DIR) +
        "/levelA_M1_fixedparam_printed_tree_sys.txt"
    });

    v.push_back({
        "levelA_M2_fixedparam",
        "M2 GaussianBlur->SobelX/Y->OR->Threshold->Dilate->CC_FILTER",
        std::string(BASELINE_TREE_DIR) +
        "/levelA_M2_fixedparam_printed_tree_sys.txt"
    });

    v.push_back({
        "levelA_M3_fixedparam",
        "M3 MedianBlur->Threshold->BITWISE_NOT->Erode->Dilate->CC_FILTER",
        std::string(BASELINE_TREE_DIR) +
        "/levelA_M3_fixedparam_printed_tree_sys.txt"
    });

    return v;
}

std::string baselineCrackFilename(int imageId)
{
    std::ostringstream oss;
    oss
        << "crack_"
        << std::setw(5)
        << std::setfill('0')
        << imageId
        << ".png";
    return oss.str();
}

bool baselineFileExists(const std::string& path)
{
    std::ifstream fin(path);
    return fin.good();
}

bool baselinePreflightTrees(const std::vector<BaselineSpec>& baselines)
{
    bool ok = true;

    for (const auto& b : baselines) {
        if (!baselineFileExists(b.treeFile)) {
            std::cerr
                << "[ERROR] Missing baseline tree: "
                << b.treeFile
                << std::endl;
            ok = false;
        }
        else {
            printf(
                "[TREE-OK] %s -> %s\n",
                b.runId.c_str(),
                b.treeFile.c_str());
        }
    }

    return ok;
}

BaselineDatasetSpec buildBaselineDatasetSpec(bool primary200)
{
    BaselineDatasetSpec d;

    if (!primary200) {
        d.modeName = "train_sanity";
        d.imageDir = BASELINE_TRAIN_IMAGE_DIR;
        d.maskDir = BASELINE_TRAIN_MASK_DIR;
        d.outputDir =
            std::string(BASELINE_OUTPUT_ROOT) +
            "/train_sanity";
        d.resultImageDir =
            d.outputDir +
            "/resImgs";
        d.runSummaryPath =
            d.outputDir +
            "/levelA_fixedparam_train_sanity_run_summary_standardF1.csv";
        d.perImagePath =
            d.outputDir +
            "/levelA_fixedparam_train_sanity_per_image_standardF1.csv";
        d.startId = BASELINE_TRAIN_START_ID;
        d.endId = BASELINE_TRAIN_END_ID;
        d.size = BASELINE_TRAIN_SIZE;
        d.isPrimary200 = false;
    }
    else {
        d.modeName = "primary200";
        d.imageDir = BASELINE_TEST_IMAGE_DIR;
        d.maskDir = BASELINE_TEST_MASK_DIR;
        d.outputDir =
            std::string(BASELINE_OUTPUT_ROOT) +
            "/primary200";
        d.resultImageDir =
            d.outputDir +
            "/resImgs";
        d.runSummaryPath =
            d.outputDir +
            "/levelA_fixedparam_primary200_run_summary_standardF1.csv";
        d.perImagePath =
            d.outputDir +
            "/levelA_fixedparam_primary200_per_image_standardF1.csv";
        d.startId = BASELINE_TEST_START_ID;
        d.endId = BASELINE_TEST_END_ID;
        d.size = BASELINE_TEST_SIZE;
        d.isPrimary200 = true;
    }

    return d;
}

void printBaselineUsage(const char* exe)
{
    printf("\nUsage:\n");
    printf("  %s --mode train-sanity\n", exe);
    printf("  %s --mode primary200 --confirm-levelA-fixed\n", exe);
    printf("\n");
    printf("If no arguments are supplied, train-sanity is used by default.\n");
    printf("Primary-200 intentionally requires --confirm-levelA-fixed.\n\n");
}

bool parseBaselineMode(
    int argc,
    char** argv,
    bool& primary200,
    bool& confirmedFrozen)
{
    primary200 =
        (BASELINE_DEFAULT_PRIMARY200 != 0);

    confirmedFrozen = false;

    if (argc <= 1) {
        return true;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mode") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --mode requires a value." << std::endl;
                return false;
            }

            const std::string mode = argv[++i];

            if (mode == "train-sanity" ||
                mode == "train_sanity")
            {
                primary200 = false;
            }
            else if (mode == "primary200" ||
                     mode == "test200")
            {
                primary200 = true;
            }
            else {
                std::cerr
                    << "[ERROR] Unknown mode: "
                    << mode
                    << std::endl;
                return false;
            }
        }
        else if (arg == "--confirm-levelA-fixed") {
            confirmedFrozen = true;
        }
        else if (arg == "--help" ||
                 arg == "-h")
        {
            printBaselineUsage(argv[0]);
            return false;
        }
        else {
            std::cerr
                << "[ERROR] Unknown argument: "
                << arg
                << std::endl;
            return false;
        }
    }

    return true;
}

bool loadBaselineDataset(
    const BaselineDatasetSpec& d,
    std::vector<cv::Mat>& imagesCPU,
    std::vector<cv::Mat>& masksCPU,
    std::vector<cv::cuda::GpuMat>& imagesGPU,
    std::vector<cv::cuda::GpuMat>& masksGPU)
{
    imagesCPU.resize(d.size);
    masksCPU.resize(d.size);
    imagesGPU.resize(d.size);
    masksGPU.resize(d.size);

    printf(
        "[DATA] mode=%s IDs=%d-%d n=%d\n",
        d.modeName.c_str(),
        d.startId,
        d.endId,
        d.size);

    for (int i = 0; i < d.size; ++i) {
        const int imageId = d.startId + i;
        const std::string filename =
            baselineCrackFilename(imageId);

        const std::string imagePath =
            d.imageDir +
            "/" +
            filename;

        const std::string maskPath =
            d.maskDir +
            "/" +
            filename;

        imagesCPU[i] =
            cv::imread(
                imagePath,
                cv::IMREAD_GRAYSCALE);

        masksCPU[i] =
            cv::imread(
                maskPath,
                cv::IMREAD_GRAYSCALE);

        if (imagesCPU[i].empty() ||
            masksCPU[i].empty())
        {
            std::cerr
                << "[ERROR] Could not load image/mask:\n"
                << "        image: "
                << imagePath
                << "\n"
                << "        mask : "
                << maskPath
                << std::endl;

            return false;
        }

        if (imagesCPU[i].type() != CV_8UC1 ||
            masksCPU[i].type() != CV_8UC1)
        {
            std::cerr
                << "[ERROR] Expected CV_8UC1: "
                << filename
                << std::endl;

            return false;
        }

        if (imagesCPU[i].size() !=
            masksCPU[i].size())
        {
            std::cerr
                << "[ERROR] Image/mask size mismatch: "
                << filename
                << std::endl;

            return false;
        }

        // IMPORTANT:
        // Training GT masks can contain values other than exact FG_PIXEL/BG_PIXEL.
        // We keep GT unchanged, matching the established GP metric semantics.
        // No thresholding or GT normalization is performed here.

#if USE_CUDA
        imagesGPU[i].upload(
            imagesCPU[i],
            getCudaStream());

        masksGPU[i].upload(
            masksCPU[i],
            getCudaStream());
#endif

        if ((i == 0) ||
            ((i + 1) % 20 == 0) ||
            (i + 1 == d.size))
        {
            printf(
                "[DATA] loaded %d/%d: %s\n",
                i + 1,
                d.size,
                filename.c_str());
        }
    }

#if USE_CUDA
    getCudaStream().waitForCompletion();
#endif

    return true;
}

double baselineCalcStandardF1(
    long long tp,
    long long fp,
    long long fn,
    double& precision,
    double& recall)
{
    precision =
        (tp + fp > 0) ?
        double(tp) /
        double(tp + fp) :
        0.0;

    recall =
        (tp + fn > 0) ?
        double(tp) /
        double(tp + fn) :
        0.0;

    const double denom =
        2.0 * double(tp) +
        double(fp) +
        double(fn);

    if (denom <= 0.0) {
        return 0.0;
    }

    return
        2.0 * double(tp) /
        denom;
}

double baselineCalcLegacyF1(
    long long tp,
    long long fp,
    long long fn,
    double& precision,
    double& recall)
{
    if (tp == 0) tp++;
    if (fp == 0) fp++;
    if (fn == 0) fn++;

    precision =
        double(tp) /
        double(tp + fp);

    recall =
        double(tp) /
        double(tp + fn);

    return
        calculateF1Score(
            precision,
            recall);
}

BaselineF1Stats baselineComputeStats(
    const std::vector<double>& values)
{
    BaselineF1Stats st;

    if (values.empty()) {
        return st;
    }

    st.minF1 = values[0];
    st.maxF1 = values[0];

    for (double v : values) {
        st.simpleSum += v;
        st.minF1 =
            std::min(
                st.minF1,
                v);
        st.maxF1 =
            std::max(
                st.maxF1,
                v);
    }

    st.meanAll =
        st.simpleSum /
        double(values.size());

    double var = 0.0;

    for (double v : values) {
        const double d =
            v - st.meanAll;
        var +=
            d * d;
    }

    // Match the previous GP primary-200 evaluator:
    // population SD across images within one deterministic method.
    st.stdAll =
        std::sqrt(
            var /
            double(values.size()));

    return st;
}

int main(int argc, char** argv)
{
#if !USE_CUDA
    std::cerr
        << "[FATAL] This baseline evaluator requires USE_CUDA=1."
        << std::endl;
    return 1;
#else

    omp_set_num_threads(1);

    bool primary200 = false;
    bool confirmedFrozen = false;

    if (!parseBaselineMode(
            argc,
            argv,
            primary200,
            confirmedFrozen))
    {
        if (argc > 1) {
            printBaselineUsage(argv[0]);
        }
        return 1;
    }

    if (primary200 && !confirmedFrozen) {
        std::cerr
            << "\n[LEVELA-PRIMARY200-BLOCKED] The primary test is intentionally protected.\n"
            << "Run training sanity first, freeze the four baseline definitions,\n"
            << "then use:\n"
            << "    "
            << argv[0]
            << " --mode primary200 --confirm-levelA-fixed\n\n";
        return 3;
    }

    initParamDesc();
    initParamDesc_safeVal();

    const std::vector<BaselineSpec> baselines =
        buildBaselineSpecs();

    const BaselineDatasetSpec dataSpec =
        buildBaselineDatasetSpec(
            primary200);

    printf("\n============================================================\n");
    printf(" Level A parameter-matched structural-control evaluator\n");
    printf(" mode=%s\n", dataSpec.modeName.c_str());
    printf(" baselines=%zu\n", baselines.size());
    printf(" IDs=%d-%d (%d images)\n",
        dataSpec.startId,
        dataSpec.endId,
        dataSpec.size);
    printf(" tree_dir=%s\n", BASELINE_TREE_DIR);
    printf(" image_dir=%s\n", dataSpec.imageDir.c_str());
    printf(" mask_dir=%s\n", dataSpec.maskDir.c_str());
    printf(" output_dir=%s\n", dataSpec.outputDir.c_str());
    printf(" USE_CUDA=%d USE_FUSED_METRICS=%d ENABLE_CC_FILTER=%d\n",
        USE_CUDA,
        USE_FUSED_METRICS,
        ENABLE_CC_FILTER);
    printf(" FG_PIXEL=%d BG_PIXEL=%d\n",
        FG_PIXEL,
        BG_PIXEL);
    printf(" metric=STANDARD F1=2TP/(2TP+FP+FN)\n");
    printf(" Level-A purpose=manual topology with GP-matched fixed parameters\n");
    printf(" Fixed params: Gaussian=(5,1.5), Median=19, Sobel=3, Canny=(100,200), Threshold=9, Erode=1, Dilate=1, CC=(5,3)\n");
    printf(" Parameter tuning=NONE\n");
    printf(" CSV schema=compatible with GP 200-image evaluator\n");
    printf(" baseline train_size field=0, seed field=0\n");
    printf("============================================================\n\n");

    if (!baselinePreflightTrees(
            baselines))
    {
        std::cerr
            << "[FATAL] Baseline tree preflight failed.\n"
            << "Copy the four provided *_printed_tree_sys.txt files to:\n"
            << "    "
            << BASELINE_TREE_DIR
            << std::endl;
        return 1;
    }

    if (!baselineCreateDirectoriesRecursive(
            dataSpec.outputDir))
    {
        return 1;
    }

#if SAVE_BASELINE_RESULT_IMAGES
    if (!baselineCreateDirectoriesRecursive(
            dataSpec.resultImageDir))
    {
        return 1;
    }
#endif

    std::vector<cv::Mat> imagesCPU;
    std::vector<cv::Mat> masksCPU;
    std::vector<cv::cuda::GpuMat> imagesGPU;
    std::vector<cv::cuda::GpuMat> masksGPU;

    if (!loadBaselineDataset(
            dataSpec,
            imagesCPU,
            masksCPU,
            imagesGPU,
            masksGPU))
    {
        std::cerr
            << "[FATAL] Dataset loading failed."
            << std::endl;
        return 1;
    }

    std::ofstream runCsv(
        dataSpec.runSummaryPath);

    std::ofstream perImageCsv(
        dataSpec.perImagePath);

    if (!runCsv.is_open() ||
        !perImageCsv.is_open())
    {
        std::cerr
            << "[FATAL] Cannot open output CSV files."
            << std::endl;
        return 1;
    }

    runCsv
        << std::fixed
        << std::setprecision(10);

    perImageCsv
        << std::fixed
        << std::setprecision(10);

    // EXACT same column names/order as the current GP 200-image summary.
    runCsv
        << "run_id,train_size,seed,test_size,"
        << "test_standard_simple_sum,test_standard_mean_f1,test_standard_std_f1,test_standard_min_f1,test_standard_max_f1,"
        << "test_legacy_simple_sum,test_legacy_mean_f1,test_legacy_std_f1,test_legacy_min_f1,test_legacy_max_f1,"
        << "elapsed_sec,tree_file\n";

    // EXACT same column names/order as the current GP 200-image per-image CSV.
    perImageCsv
        << "run_id,train_size,seed,image_id,tp,fp,fn,"
        << "precision_standard,recall_standard,f1_standard,"
        << "precision_legacy,recall_legacy,f1_legacy,result_image\n";

    int completedBaselines = 0;
    int failedBaselines = 0;
    long long totalEvaluations = 0;

    const auto allStart =
        std::chrono::high_resolution_clock::now();

    for (const auto& b : baselines)
    {
        printf("\n------------------------------------------------------------\n");
        printf("[LEVELA-START] %s\n",
            b.runId.c_str());
        printf("[PIPELINE] %s\n",
            b.pipelineName.c_str());
        printf("[TREE] %s\n",
            b.treeFile.c_str());

        const auto runStart =
            std::chrono::high_resolution_clock::now();

        std::shared_ptr<TreeNode> tree =
            loadTreeFromFile(
                b.treeFile);

        if (!tree) {
            std::cerr
                << "[ERROR] Cannot load baseline tree: "
                << b.treeFile
                << std::endl;
            failedBaselines++;
            continue;
        }

        confirmDepth(
            tree,
            MAX_DEPTH);

        std::vector<double> f1StandardValues;
        std::vector<double> f1LegacyValues;

        f1StandardValues.reserve(
            dataSpec.size);

        f1LegacyValues.reserve(
            dataSpec.size);

        int invalidImages = 0;

        for (int idx = 0;
             idx < dataSpec.size;
             ++idx)
        {
            const int imageId =
                dataSpec.startId +
                idx;

            CudaEvalContext ctx;

            cv::cuda::GpuMat pred =
                executeTreeCUDA(
                    tree,
                    imagesGPU[idx],
                    ctx);

            ctx.hold(
                pred);

            MetricsGPUFused m =
                calcMetricsOneGPUFused(
                    pred,
                    masksGPU[idx],
                    FG_PIXEL,
                    getCudaStream());

            const long long tp = m.tp;
            const long long fp = m.fp;
            const long long fn = m.fn;

            double precisionStandard = 0.0;
            double recallStandard = 0.0;
            double f1Standard = 0.01;

            double precisionLegacy = 0.0;
            double recallLegacy = 0.0;
            double f1Legacy = 0.01;

            if (m.invalid > 0) {
                invalidImages++;

                printf(
                    "[WARN] invalid non-binary output: baseline=%s image=crack_%05d invalid=%d -> F1 penalty=0.01\n",
                    b.runId.c_str(),
                    imageId,
                    m.invalid);
            }
            else {
                f1Standard =
                    baselineCalcStandardF1(
                        tp,
                        fp,
                        fn,
                        precisionStandard,
                        recallStandard);

                f1Legacy =
                    baselineCalcLegacyF1(
                        tp,
                        fp,
                        fn,
                        precisionLegacy,
                        recallLegacy);
            }

            f1StandardValues.push_back(
                f1Standard);

            f1LegacyValues.push_back(
                f1Legacy);

            std::string resultImagePath;

#if SAVE_BASELINE_RESULT_IMAGES
            {
                std::ostringstream outName;

                outName
                    << dataSpec.resultImageDir
                    << "/"
                    << b.runId
                    << "_"
                    << dataSpec.modeName
                    << "_crack_"
                    << std::setw(5)
                    << std::setfill('0')
                    << imageId
                    << ".png";

                resultImagePath =
                    outName.str();

                cv::Mat predCPU;

                pred.download(
                    predCPU,
                    getCudaStream());

                getCudaStream().waitForCompletion();

                if (!cv::imwrite(
                        resultImagePath,
                        predCPU))
                {
                    std::cerr
                        << "[WARN] Failed to save result image: "
                        << resultImagePath
                        << std::endl;
                }
            }
#endif

            perImageCsv
                << b.runId << ","
                << 0 << ","
                << 0 << ","
                << imageId << ","
                << tp << ","
                << fp << ","
                << fn << ","
                << precisionStandard << ","
                << recallStandard << ","
                << f1Standard << ","
                << precisionLegacy << ","
                << recallLegacy << ","
                << f1Legacy << ","
                << resultImagePath
                << "\n";

            const bool logThis =
                (idx == 0) ||
                ((idx + 1) % 20 == 0) ||
                (idx + 1 == dataSpec.size) ||
                (m.invalid > 0);

            if (logThis) {
                printf(
                    "[F1] baseline=%s image=crack_%05d progress=%d/%d "
                    "f1_standard=%.10f tp=%lld fp=%lld fn=%lld invalid=%d\n",
                    b.runId.c_str(),
                    imageId,
                    idx + 1,
                    dataSpec.size,
                    f1Standard,
                    tp,
                    fp,
                    fn,
                    m.invalid);
            }

            totalEvaluations++;
        }

        const BaselineF1Stats stStandard =
            baselineComputeStats(
                f1StandardValues);

        const BaselineF1Stats stLegacy =
            baselineComputeStats(
                f1LegacyValues);

        const auto runEnd =
            std::chrono::high_resolution_clock::now();

        const double elapsedSec =
            std::chrono::duration<double>(
                runEnd -
                runStart).count();

        runCsv
            << b.runId << ","
            << 0 << ","
            << 0 << ","
            << dataSpec.size << ","

            << stStandard.simpleSum << ","
            << stStandard.meanAll << ","
            << stStandard.stdAll << ","
            << stStandard.minF1 << ","
            << stStandard.maxF1 << ","

            << stLegacy.simpleSum << ","
            << stLegacy.meanAll << ","
            << stLegacy.stdAll << ","
            << stLegacy.minF1 << ","
            << stLegacy.maxF1 << ","

            << elapsedSec << ","
            << b.treeFile
            << "\n";

        runCsv.flush();
        perImageCsv.flush();

        printf(
            "[LEVELA-SUMMARY] %s mode=%s n=%d "
            "standardMeanF1=%.10f standardStdF1=%.10f "
            "standardMinF1=%.10f standardMaxF1=%.10f "
            "invalidImages=%d elapsedSec=%.3f\n",
            b.runId.c_str(),
            dataSpec.modeName.c_str(),
            dataSpec.size,
            stStandard.meanAll,
            stStandard.stdAll,
            stStandard.minF1,
            stStandard.maxF1,
            invalidImages,
            elapsedSec);

        completedBaselines++;
    }

    const auto allEnd =
        std::chrono::high_resolution_clock::now();

    const double totalElapsedSec =
        std::chrono::duration<double>(
            allEnd -
            allStart).count();

    runCsv.close();
    perImageCsv.close();

    const long long expectedEvaluations =
        (long long)baselines.size() *
        (long long)dataSpec.size;

    printf("\n============================================================\n");
    printf(
        "[LEVELA-DONE] mode=%s completed=%d failed=%d total=%zu\n",
        dataSpec.modeName.c_str(),
        completedBaselines,
        failedBaselines,
        baselines.size());

    printf(
        "[LEVELA-DONE] evaluations=%lld expected=%lld\n",
        totalEvaluations,
        expectedEvaluations);

    printf(
        "[LEVELA-DONE] elapsedSec=%.3f\n",
        totalElapsedSec);

    printf(
        "[LEVELA-DONE] run_summary=%s\n",
        dataSpec.runSummaryPath.c_str());

    printf(
        "[LEVELA-DONE] per_image=%s\n",
        dataSpec.perImagePath.c_str());

    if (!dataSpec.isPrimary200) {
        printf("\n[IMPORTANT] This was TRAINING SANITY only.\n");
        printf("Check implementation/polarity/binary outputs BEFORE freezing.\n");
        printf("Do not optimize repeatedly for training F1.\n");
        printf("After the baseline definitions are frozen, run:\n");
        printf("    %s --mode primary200 --confirm-levelA-fixed\n", argv[0]);
    }
    else {
        printf("\n[IMPORTANT] Primary-200 has now been evaluated.\n");
        printf("Do NOT modify baseline topology or parameters based on these results.\n");
    }

    printf("============================================================\n");

    return
        (failedBaselines == 0 &&
         completedBaselines == (int)baselines.size() &&
         totalEvaluations == expectedEvaluations) ?
        0 :
        2;

#endif
}
