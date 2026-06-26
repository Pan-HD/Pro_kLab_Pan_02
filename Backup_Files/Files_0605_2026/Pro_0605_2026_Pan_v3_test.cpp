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
// (!important) The foreground mode (pills: 0, crack: 1) must be confirmed before the program runs.
#include "SegmentationConfig.h" 

using namespace std;
using namespace cv;

#define IMG_START_IDX 211
#define TEST_SIZE 200

#define MAX_DEPTH 12 // { 0, 1, 2, ... } GP
#define NUM_TYPE_FUNC 16 // GP

#define CUDA_EQ_TEST_BILATERAL 0
#define CUDA_EQ_TEST_MED       1 // 6.80
#define CUDA_EQ_TEST_BLUR      1 // 6.78
#define CUDA_EQ_TEST_ERODE     1 // 6.75
#define CUDA_EQ_TEST_DILATE    1 // 6.77
#define CUDA_EQ_TEST_DIFF      1

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

/*
    CUDA Runtime
*/
// thread_local cv::cuda::Stream gCudaStream;
inline cv::cuda::Stream& getCudaStream()
{
    thread_local cv::cuda::Stream s;
    return s;
}

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

struct TreeNode {
    FilterType type = TERMINAL_INPUT;
    vector<shared_ptr<TreeNode>> children;
    vector<double> params;
};

struct Metrics {
    long long tp = 0;
    long long fp = 0;
    long long fn = 0;
};

struct TestScore {
    double precision;
    double recall;
    double f1;
    double iou;
};

struct MetricsGPU
{
    int tp;
    int fp;
    int fn;
};

static thread_local std::mt19937 rng(
#if USE_FIXED_SEED
    RANDOM_SEED + omp_get_thread_num()
#else
    (unsigned)(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + omp_get_thread_num()
#endif
);

static std::uniform_real_distribution<double> uni_real(0.0, 1.0);

inline int rand_int(int a, int b) { // inclusive
    std::uniform_int_distribution<int> d(a, b);
    return d(rng);
}

inline double rand_real() {
    return uni_real(rng);
}

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

static unordered_map<FilterType, vector<double>> g_paramDesc_safeVal;
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

bool isSafeValType(FilterType t) {
    if (t == THRESHOLD || t == ERODE || t == DILATE || t == SOBEL_X || t == SOBEL_Y) {
        return false;
    }
    else {
        return true;
    }
}

int getTreeMaxDepth(const shared_ptr<TreeNode>& node, int depth = 0) {
    if (!node) return depth;
    if (node->children.empty()) return depth;
    int maxChildDepth = depth;
    for (auto& child : node->children) maxChildDepth = max(maxChildDepth, getTreeMaxDepth(child, depth + 1));
    return maxChildDepth;
}

bool isTerminal(FilterType type) {
    return (type == TERMINAL_INPUT);
}

bool isBinaryFilter(FilterType type) {
    return (type == DIFF_PROCESS || type == BITWISE_AND || type == BITWISE_OR || type == BITWISE_XOR);
}

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

Mat executeContourProcessCPU(const Mat& srcMask, const vector<double>& params) { // Waiting for Module-Testing
    Mat maskImg = srcMask.clone();
    Mat contourInput;
#if FOREGROUND_WHITE
    bitwise_not(maskImg, contourInput);
#else
    contourInput = maskImg;
#endif

    int kk = params.size() > 0 ? int(params[0]) : 1;
    int k = ((kk) / 2) | 1;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
    int times = params.size() > 1 ? int(params[1]) / 2 : 0;
    for (int t = 0; t < times; ++t) {
        erode(contourInput, contourInput, kernel);
    }
    vector<vector<Point>> contours;
    findContours(contourInput, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Mat mask(contourInput.rows, contourInput.cols, CV_8UC1, Scalar(BG_PIXEL));
    int selType = 0;
    if (params.size() > 2) {
        selType = min(2, int(params[2] / 5));
    }
    for (const auto& contour : contours) {
        Rect bb = boundingRect(contour);
        double aspect_ratio = double(bb.width) / double(bb.height + 1e-9);
        if (selType == 0) {
            int range = params.size() > 3 ? int(params[3]) / 2 : 0;
            if (aspect_ratio >= (1 - range * 0.1) && aspect_ratio <= (1 + range * 0.1)) {
                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(FG_PIXEL), -1);
            }
        }
        else if (selType == 1) {
            int areaTh = params.size() > 4 ? int(params[4]) : 1;
            if (contourArea(contour) >= 100 * areaTh) {
                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(FG_PIXEL), -1);
            }
        }
        else {
            int range = params.size() > 3 ? int(params[3]) : 0;
            int areaTh = params.size() > 4 ? int(params[4]) : 1;
            if (aspect_ratio >= (1 - range * 0.1) && aspect_ratio <= (1 + range * 0.1) && contourArea(contour) >= 100 * areaTh) {
                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(FG_PIXEL), -1);
            }
        }
    }
#if FOREGROUND_WHITE
    bitwise_not(mask, mask);
#endif
    return mask;
}

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
        Mat child = executeTree(node->children[0], input);
        return executeContourProcessCPU(child, node->params);
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

cv::cuda::GpuMat executeTreeCUDA(const shared_ptr<TreeNode>& node, const cv::cuda::GpuMat& input)
{
    if (!node)
        return input;
    try
    {
        switch (node->type)
        {
        case TERMINAL_INPUT:
        {
            return input.clone();
        }
        case GAUSSIAN_BLUR:
        {
            auto child = executeTreeCUDA(node->children[0], input);
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
            return dst;
        }
        case MED_BLUR:
        {
#if CUDA_EQ_TEST_MED
            return fallbackCPU(node, input);
#endif

            auto child = executeTreeCUDA(node->children[0], input);
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
            return dst;
        }
        case BLUR:
        {
#if CUDA_EQ_TEST_BLUR
            return fallbackCPU(node, input);
#endif

            auto child = executeTreeCUDA(node->children[0], input);
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
            return dst;
        }
        case BILATERAL_FILTER:
        {
#if CUDA_EQ_TEST_BILATERAL
            return fallbackCPU(node, input);
#endif

            auto child = executeTreeCUDA(node->children[0], input);
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

            return dst;
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
            auto child = executeTreeCUDA(node->children[0], input);
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
            return grad8;
        }
        case SOBEL_Y:
        {
            auto child = executeTreeCUDA(node->children[0], input);
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
            return grad8;
        }
        case CANNY:
        {
            auto child = executeTreeCUDA(node->children[0], input);
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
            return dst;
        }
        case DIFF_PROCESS:
        {
#if CUDA_EQ_TEST_DIFF
            return fallbackCPU(node, input);
#endif

            auto a = executeTreeCUDA(node->children[0], input);
            auto b = executeTreeCUDA(node->children[1], input);

            cv::cuda::GpuMat dst;

            cv::cuda::subtract(
                a,
                b,
                dst,
                cv::noArray(),
                CV_8U,
                getCudaStream());

            return dst;
        }
        case THRESHOLD:
        {
            auto child =
                executeTreeCUDA(
                    node->children[0],
                    input);
            CV_Assert(child.type() == CV_8UC1);
            double th =
                node->params.size() > 0 ?
                node->params[0] : 127.0;
            cv::cuda::GpuMat dst;
            cv::cuda::threshold(child, dst, th, 255, THRESH_BINARY, getCudaStream());
            return dst;
        }
        case ERODE:
        {
#if CUDA_EQ_TEST_ERODE
            return fallbackCPU(node, input);
#endif

            auto child =
                executeTreeCUDA(
                    node->children[0],
                    input);
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
            return dst;
        }
        case DILATE:
        {
#if CUDA_EQ_TEST_DILATE
            return fallbackCPU(node, input);
#endif

            auto child =
                executeTreeCUDA(
                    node->children[0],
                    input);
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
            return dst;
        }
        case BITWISE_AND:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input);
            auto b =
                executeTreeCUDA(
                    node->children[1],
                    input);
            CV_Assert(a.type() == b.type());
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_and(a, b, dst, cv::noArray(), getCudaStream());
            return dst;
        }
        case BITWISE_OR:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input);
            auto b =
                executeTreeCUDA(
                    node->children[1],
                    input);
            CV_Assert(a.type() == b.type());
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_or(a, b, dst, cv::noArray(), getCudaStream());
            return dst;
        }
        case BITWISE_XOR:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input);
            auto b =
                executeTreeCUDA(
                    node->children[1],
                    input);
            CV_Assert(a.type() == b.type());
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_xor(a, b, dst, cv::noArray(), getCudaStream());
            return dst;
        }
        case BITWISE_NOT:
        {
            auto a =
                executeTreeCUDA(
                    node->children[0],
                    input);
            cv::cuda::GpuMat dst;
            cv::cuda::bitwise_not(a, dst, cv::noArray(), getCudaStream());
            return dst;
        }
        /*
            CONTOUR_PROCESS
            Remain Calculating in CPU
        */
        case CONTOUR_PROCESS: {
            auto child = executeTreeCUDA(node->children[0], input);
            Mat cpuMask;
            child.download(cpuMask, getCudaStream());
            getCudaStream().waitForCompletion();
            Mat cpuRes = executeContourProcessCPU(cpuMask, node->params);
            cv::cuda::GpuMat gpuRes;
            gpuRes.upload(cpuRes, getCudaStream());
            return gpuRes;
        }
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
        cerr << "[CUDA ERROR] "
            << e.what()
            << endl;
        return input.clone();
    }
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
        {"CONTOUR_PROCESS", CONTOUR_PROCESS},
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

Metrics calcMetrics(const Mat& pred, const Mat& gt) {
    Metrics m;
    for (int y = 0;y < pred.rows;y++) {
        const uchar* pPred = pred.ptr<uchar>(y);
        const uchar* pGT = gt.ptr<uchar>(y);
        for (int x = 0;x < pred.cols;x++) {
            bool predFG = (pPred[x] == FG_PIXEL);
            bool gtFG = (pGT[x] == FG_PIXEL);
            if (predFG && gtFG)
                m.tp++;
            else if (predFG && !gtFG)
                m.fp++;
            else if (!predFG && gtFG)
                m.fn++;
        }
    }
    return m;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

MetricsGPU calcMetricsOneGPU(const cv::cuda::GpuMat& pred, const cv::cuda::GpuMat& gt)
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

static TestScore calcScoreSameAsTraining(const MetricsGPU& m) {
    int tp = m.tp;
    int fp = m.fp;
    int fn = m.fn;
    if (tp == 0) tp++;
    if (fp == 0) fp++;
    if (fn == 0) fn++;
    TestScore s;
    s.precision = double(tp) / double(tp + fp);
    s.recall = double(tp) / double(tp + fn);
    s.f1 = calculateF1Score(s.precision, s.recall);
    s.iou = double(tp) / double(tp + fp + fn);
    return s;
}

void printResult(const Metrics& m, FILE* f1_evaluation)
{
    double precision = double(m.tp) / (m.tp + m.fp + 1e-10);
    double recall = double(m.tp) / (m.tp + m.fn + 1e-10);
    double f1 = 2.0 * precision * recall / (precision + recall + 1e-10);
    double iou = double(m.tp) / (m.tp + m.fp + m.fn + 1e-10);

    printf("\n");
    printf("Precision : %.6f\n", precision);
    printf("Recall    : %.6f\n", recall);
    printf("F1-Score  : %.6f\n", f1);
    printf("IoU       : %.6f\n", iou);

    if (f1_evaluation) {
        fprintf(f1_evaluation, "Precision : %.6f\n", precision);
        fprintf(f1_evaluation, "Recall    : %.6f\n", recall);
        fprintf(f1_evaluation, "F1-Score  : %.6f\n", f1);
        fprintf(f1_evaluation, "IoU       : %.6f\n", iou);
    }
}

int main(void) {
    FILE* fl_evaluation = nullptr;
    errno_t err = fopen_s(&fl_evaluation, "./imgs_0605_2026_v3/output/test_output/evaluation.txt", "a");
    if (err != 0 || fl_evaluation == nullptr) {
        perror("Cannot open the file");
    }

    initParamDesc();
    initParamDesc_safeVal();

    auto eliteTree = loadTreeFromFile("./imgs_0605_2026_v3/input/test/elite_tree_gp/printed_tree_sys.txt");
    if (!eliteTree) {
        printf("Load Tree Error\n");
        return -1;
    }

    double sumPrecision = 0.0;
    double sumRecall = 0.0;
    double sumF1 = 0.0;
    double sumIoU = 0.0;
    int imageCount = 0;

    // Metrics total;
    for (int i = 0; i < TEST_SIZE; i++) {
        char imgName[256];
        char gtName[256];
        sprintf_s(imgName, "./imgs_0605_2026_v3/input/test/images/crack_%05d.png", IMG_START_IDX + i);
        sprintf_s(gtName, "./imgs_0605_2026_v3/input/test/masks/crack_%05d.png", IMG_START_IDX + i);
        Mat src = imread(imgName, IMREAD_GRAYSCALE);
        Mat gt = imread(gtName, IMREAD_GRAYSCALE);

        cv::cuda::GpuMat gGT;
        gGT.upload(gt);
        cv::cuda::GpuMat gsrc;
        gsrc.upload(src);

        cv::cuda::GpuMat gres = executeTreeCUDA(eliteTree, gsrc);

        Mat pred;
        gres.download(pred);
        char outName[256];
        sprintf_s(outName, "./imgs_0605_2026_v3/output/test_output/resImgs/res_%05d.png", IMG_START_IDX + i);
        imwrite(outName, pred);

        MetricsGPU m = calcMetricsOneGPU(gres, gGT);
        TestScore score = calcScoreSameAsTraining(m);

        sumPrecision += score.precision;
        sumRecall += score.recall;
        sumF1 += score.f1;
        sumIoU += score.iou;
        imageCount++;

        /*
        Metrics m = calcMetrics(pred, gt);
        total.tp += m.tp;
        total.fp += m.fp;
        total.fn += m.fn;
        */

    }

    double avgPrecision = sumPrecision / imageCount;
    double avgRecall = sumRecall / imageCount;
    double avgF1 = sumF1 / imageCount;
    double avgIoU = sumIoU / imageCount;

    if (fl_evaluation) {
        fprintf(fl_evaluation, "Precision : %.6f\n", avgPrecision);
        fprintf(fl_evaluation, "Recall    : %.6f\n", avgRecall);
        fprintf(fl_evaluation, "F1-Score  : %.6f\n", avgF1);
        fprintf(fl_evaluation, "IoU       : %.6f\n", avgIoU);
    }

    // printResult(total, fl_evaluation);

    if (fl_evaluation) fclose(fl_evaluation);
    return 0;
}