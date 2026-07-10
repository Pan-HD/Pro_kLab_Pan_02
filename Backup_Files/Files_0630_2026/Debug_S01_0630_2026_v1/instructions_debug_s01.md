# 第一阶段：锁定正确性

目标：确认当前 CUDA baseline 是确定的、最优适应度不下降、CC_FILTER 不再引入不稳定。

## 1.1 添加重复评分测试函数

在 `GPU_OMP1_stable_baseline.cpp` 中新增一个调试函数：

```
void debugRepeatScore(
    const shared_ptr<TreeNode>& tree,
    Mat imgArr[][2],
    int repeatCount = 100)
{
    vector<double> scores;
    scores.reserve(repeatCount);

    for (int i = 0; i < repeatCount; ++i) {
        double s = CAL_SCORE(tree, imgArr, -1);
        scores.push_back(s);
        printf("[RepeatScore] %03d : %.12f\n", i, s);
    }

    double minS = *min_element(scores.begin(), scores.end());
    double maxS = *max_element(scores.begin(), scores.end());

    printf("[RepeatScore Summary] min=%.12f max=%.12f diff=%.12f\n",
        minS, maxS, maxS - minS);

    if (fabs(maxS - minS) > 1e-9) {
        printf("[ERROR] Non-deterministic CUDA score detected.\n");
    }
    else {
        printf("[OK] CUDA score is deterministic.\n");
    }
}
```

在 `main()` 中读取图像并上传 GPU 后，临时插入：

```
auto testTree = generateRandomTree();
confirmDepth(testTree);

debugRepeatScore(testTree, imgArr, 100);

return 0;
```

如果你想用真实 elite tree，也可以从 `printed_tree_sys.txt` 加载一棵树进行测试。

### 判定标准

```
100 次评分结果必须完全一致，或差值 <= 1e-9。
```

如果出现波动，先不要做阶段2，继续查 CUDA 生命周期或 CC_FILTER 同步。

```
Filename: debug_0708_01.cpp
测试条件：
- PC: PAN_MAIN (kLab-96)
- TRAIN_SIZE = 5
- USE_CUDA = 1
- GPU型号 RTX 20
- OMP线程数 1
- 总世代数 10000
- 总耗时 (START: 15:10, 07-02， END: xx:xx, xx-xx )
- 世代/小时 xx
- 最终elite fitness xx
```



------

## 1.2 检查 2000 代 eliteFValue 是否单调不下降

```
Filename: debug_0708_02.cpp (无需测试)
-> GPU_OMP1_stable_baseline.cpp
-> 已证明 2000 代 eliteFValue 单调不下降
```

------

## 1.3 开启/关闭 CC_FILTER 对比测试

目的不是做正式实验，而是确认不稳定是否来自 CC_FILTER。

在代码顶部加入：

```
#define ENABLE_CC_FILTER 1
```

然后在 `generateRandomTree()` 中不要直接：

```
int t_idx = rand_int(1, NUM_TYPE_FUNC);
```

改成：

```
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
```

然后把：

```
int t_idx = rand_int(1, NUM_TYPE_FUNC);
FilterType t = static_cast<FilterType>(t_idx);
```

改成：

```
FilterType t = randomFunctionType();
```

执行两个版本：

```
A: ENABLE_CC_FILTER = 1
B: ENABLE_CC_FILTER = 0
```

每个版本至少跑：

```
repeat score 100 次
训练 2000 代
记录耗时、eliteDropCount、best fitness
```

### 注意

`ENABLE_CC_FILTER = 0` **只能用于诊断或 ablation**，不能混入实验2正式结果。因为关闭 CC_FILTER 改变了 GP 的函数集合，也就是改变了搜索空间。稳定版中 CC_FILTER 前已经显式等待 OpenCV CUDA stream 完成，这是为了处理 raw CUDA / NPP / default stream 的同步边界。

```
Filename: debug_0708_03.cpp
START TIME: 11:50, 07-08, 2026

Type-A (ENABLE_CC_FILTER = 1): YANG_MAIN
  repeat score(100): √
  train 2000 generations: √
    END TIME: 14:08, 07-08, 2026
Type-B (ENABLE_CC_FILTER = 0): YANG_NOTEBOOK
  repeat score(100): √
  train 2000 generations:
    END TIME: 15:48, 07-08, 2026
```

```
RESULT:
Type-A (ENABLE_CC_FILTER = 1): 
  repeat score(100): 
    EliteDropCount: 0
  2000 generations:
    TIME: 2h18min
    EliteDropCount: 0
    best fitness: 3.15

Type-B (ENABLE_CC_FILTER = 0): 
  repeat score(100): 
    EliteDropCount: 0
  2000 generations:
    TIME: 4h
    EliteDropCount: 0
    best fitness: 2.84
```



------

## 第一阶段输出物

建议输出：

```
logs/stage1_correctness/
  repeat_score_enable_cc.txt
  repeat_score_disable_cc.txt
  train_2000_enable_cc.csv
  train_2000_disable_cc.csv
  summary.txt
```

`summary.txt` 写：

```
ENABLE_CC_FILTER=1:
  repeat_score_diff =
  eliteDropCount =
  time_per_generation =

ENABLE_CC_FILTER=0:
  repeat_score_diff =
  eliteDropCount =
  time_per_generation =
```

------

# 第二阶段：合并二值检查和 TP/FP/FN 计算

目标：用一个 fused CUDA kernel 替代：

```
isBinaryImageGPU(resImg)
calcMetricsOneGPU(resImg, gTarImgArr[idxSet])
```

稳定版当前是先判断输出是否为 0/255，再调用 `calcMetricsOneGPU()`，这会产生多次 OpenCV CUDA compare、bitwise 和同步。 修改目标是不改变语义，只减少 CUDA 调用和同步次数。

------

## 2.1 新增 `cuda_metrics_fused.cuh`

新建文件：

```
cuda_metrics_fused.cuh
```

内容：

```
#pragma once
#include <opencv2/core/cuda.hpp>

struct MetricsGPUFused
{
    long long tp = 0;
    long long fp = 0;
    long long fn = 0;
    long long invalid = 0;
};

MetricsGPUFused calcMetricsOneGPUFused(
    const cv::cuda::GpuMat& pred,
    const cv::cuda::GpuMat& gt,
    int fgPixel,
    cv::cuda::Stream& stream);
```

------

## 2.2 新增 `cuda_metrics_fused.cu`

新建：

```
cuda_metrics_fused.cu
```

代码示例：

```
#include "cuda_metrics_fused.cuh"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

__global__ void fusedMetricsKernel(
    const unsigned char* pred,
    size_t predStep,
    const unsigned char* gt,
    size_t gtStep,
    int rows,
    int cols,
    int fgPixel,
    unsigned long long* counters)
{
    __shared__ unsigned int s_tp;
    __shared__ unsigned int s_fp;
    __shared__ unsigned int s_fn;
    __shared__ unsigned int s_invalid;

    if (threadIdx.x == 0) {
        s_tp = 0;
        s_fp = 0;
        s_fn = 0;
        s_invalid = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int y = idx / cols;
        int x = idx % cols;

        unsigned char p = pred[y * predStep + x];
        unsigned char g = gt[y * gtStep + x];

        bool predInvalid = !(p == 0 || p == 255);
        bool predFG = (p == fgPixel);
        bool gtFG = (g == fgPixel);

        if (predInvalid) {
            atomicAdd(&s_invalid, 1);
        }
        else {
            if (predFG && gtFG) {
                atomicAdd(&s_tp, 1);
            }
            else if (predFG && !gtFG) {
                atomicAdd(&s_fp, 1);
            }
            else if (!predFG && gtFG) {
                atomicAdd(&s_fn, 1);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&counters[0], (unsigned long long)s_tp);
        atomicAdd(&counters[1], (unsigned long long)s_fp);
        atomicAdd(&counters[2], (unsigned long long)s_fn);
        atomicAdd(&counters[3], (unsigned long long)s_invalid);
    }
}

MetricsGPUFused calcMetricsOneGPUFused(
    const cv::cuda::GpuMat& pred,
    const cv::cuda::GpuMat& gt,
    int fgPixel,
    cv::cuda::Stream& stream)
{
    MetricsGPUFused out;

    unsigned long long* dCounters = nullptr;
    cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

    cudaMalloc(&dCounters, sizeof(unsigned long long) * 4);
    cudaMemsetAsync(dCounters, 0, sizeof(unsigned long long) * 4, s);

    int total = pred.rows * pred.cols;
    int block = 256;
    int grid = (total + block - 1) / block;

    fusedMetricsKernel<<<grid, block, 0, s>>>(
        pred.ptr<unsigned char>(),
        pred.step,
        gt.ptr<unsigned char>(),
        gt.step,
        pred.rows,
        pred.cols,
        fgPixel,
        dCounters);

    unsigned long long hCounters[4] = {0, 0, 0, 0};
    cudaMemcpyAsync(
        hCounters,
        dCounters,
        sizeof(unsigned long long) * 4,
        cudaMemcpyDeviceToHost,
        s);

    cudaStreamSynchronize(s);
    cudaFree(dCounters);

    out.tp = (long long)hCounters[0];
    out.fp = (long long)hCounters[1];
    out.fn = (long long)hCounters[2];
    out.invalid = (long long)hCounters[3];

    return out;
}
```

------

## 2.3 在主文件中加入开关

在 `GPU_OMP1_stable_baseline.cpp` 顶部加入：

```
#define USE_FUSED_METRICS 1
```

并 include：

```
#include "cuda_metrics_fused.cuh"
```

------

## 2.4 替换 `calScoreByIndCUDA()`

把当前逻辑：

```
if (!isBinaryImageGPU(resImg)) {
    return 0.01;
}

MetricsGPU m = calcMetricsOneGPU(resImg, gTarImgArr[idxSet]);
```

改成：

```
#if USE_FUSED_METRICS
MetricsGPUFused m = calcMetricsOneGPUFused(
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
#else
if (!isBinaryImageGPU(resImg)) {
    return 0.01;
}

MetricsGPU m = calcMetricsOneGPU(resImg, gTarImgArr[idxSet]);

long long tp = m.tp;
long long fp = m.fp;
long long fn = m.fn;
#endif
```

后面的 fitness 计算保持不变：

```
if (tp == 0) tp++;
if (fp == 0) fp++;
if (fn == 0) fn++;

double precision = double(tp) / double(tp + fp);
double recall = double(tp) / double(tp + fn);
double f1 = calculateF1Score(precision, recall);
```

训练阶段保留 `tp/fp/fn == 0` 时加 1 的逻辑，因为这是原 fitness 定义的一部分。测试阶段则继续使用你已加入的 formal score。

------

## 2.5 正确性验证

用同一批训练图、同一随机种子，分别跑：

```
USE_FUSED_METRICS = 0
USE_FUSED_METRICS = 1
```

先不要跑完整实验，只跑：

```
同一棵树重复评分 100 次
随机 100 棵树：old metrics vs fused metrics
训练 2000 代
```

新增一个对照函数：

```
void debugCompareOldAndFused(
    const shared_ptr<TreeNode>& tree,
    Mat imgArr[][2])
{
    double oldScore;
    double fusedScore;

#undef USE_FUSED_METRICS
    oldScore = calScoreByIndCUDA(tree, imgArr, -1);

#define USE_FUSED_METRICS 1
    fusedScore = calScoreByIndCUDA(tree, imgArr, -1);

    printf("[CompareMetrics] old=%.12f fused=%.12f diff=%.12f\n",
        oldScore,
        fusedScore,
        fabs(oldScore - fusedScore));
}
```

实际 C++ 宏不能在函数中这样动态切换，推荐做法是写两个函数：

```
double calScoreByIndCUDA_old(...);
double calScoreByIndCUDA_fused(...);
```

然后对比。

### 判定标准

```
oldScore 与 fusedScore 必须完全一致，或 diff <= 1e-9。
2000 代 eliteFValue 不下降。
速度有提升才进入阶段3。
```

```
Filename: debug_0708_04.cpp
START TIME: xx:xx, 07-08, 2026

Type-A (ENABLE_CC_FILTER = 1): 
  repeat score(100): ×　→　TESTING PART CURRENT
  ↓ Blocked temply
  (train 2000 generations: 
    END TIME: xx:xx, 07-08, 2026)
```

## 当前实验进度报告

本研究的总体目的，是在“小规模数据环境下的裂纹检测”任务中，完成遗传编程/遗传算法方法与深度学习方法的比较实验。其中，当前主要负责的是遗传方法部分的实现与稳定化，目标是在有限训练图像数量下，通过 GP 树结构组合图像处理算子，自动搜索能够较好分割裂纹区域的图像处理流程，并进一步评估训练数据量变化对检测性能的影响。

原定实验计划中，首先需要建立一个能够稳定运行的 GPU 版 GP 基准程序，确保在 CUDA 执行路径下，个体适应度评价结果具有确定性，尤其是最优个体适应度不会在进化过程中异常下降。在此基础上，再逐步进行性能优化。原计划的优化顺序包括：第一阶段，固定稳定 baseline 并验证 CUDA 评分确定性；第二阶段，将原有的二值检查与 TP/FP/FN 统计合并为 fused CUDA metrics kernel，以减少同步和多次 `countNonZero()` 调用；第三阶段，再考虑减少 `TERMINAL_INPUT` 的 GPU copy、启用 OpenCV CUDA BufferPool；之后再单独分析 `CC_FILTER`、多 stream 或 CUDA+OpenMP 并行评分等更高风险优化。

目前实验已经完成第一阶段稳定性验证。结果显示，在 `ENABLE_CC_FILTER = 1` 时，重复评分 100 次没有出现适应度下降，2000 代训练也没有出现 `EliteDropCount`，运行时间约为 2 小时 18 分钟，最佳适应度为 3.15；在 `ENABLE_CC_FILTER = 0` 时，同样未出现 `EliteDropCount`，2000 代运行时间约为 4 小时，最佳适应度为 2.84。由此可以判断，当前稳定 baseline 在 CUDA 执行路径和精英保持逻辑上已经基本可靠，可以作为后续优化前的正确性基准。

当前正在进行第二阶段，即 fused metrics 的替换与验证。该阶段的目的是将旧版 CUDA 评分中分离的二值检查、TP、FP、FN 统计过程合并为一次 CUDA kernel 执行，从而减少 GPU kernel 调度次数、同步次数以及 Host-Device 往返开销。初始 fused metrics 实现中曾出现 old/fused 评分不一致的问题，主要原因是 fused kernel 对 ground truth 背景像素的判断方式与旧版 `calcMetricsOneGPU()` 不完全一致。旧版逻辑只将 `gt == BG_PIXEL` 计为背景，而不是简单使用 `gt != FG_PIXEL`。修正后，在同一个 `resImg` 输出图像上，旧版 metrics 与 fused metrics 的 TP、FP、FN 和 F1 已经完全一致；`SameOutput` 测试中多组结果均显示 `diff = 0.000000000000`，说明 fused metrics 本身已经与旧版评分语义一致。

不过，在进一步进行 `TwoExec` 测试时发现，同一棵 GP 树对同一张输入图像重复执行两次，输出图像偶尔仍存在少量像素差异。初步统计显示，这类差异通常只涉及几十到数百个像素，相对于单张图像约 200704 个像素而言比例很小，但由于 GP 进化过程中个体适应度排序对结果敏感，因此仍不能直接忽略。进一步定位发现，所有出现非确定性输出的错误树均包含 `CC_FILTER` 节点。例如，出错树的统计信息均显示 `CC > 0`，其中还包括非常简单的结构 `CC_FILTER -> CANNY -> TERMINAL_INPUT`，这说明问题并非由复杂树结构组合引起，而是集中在 `CC_FILTER` 或其内部实现上。

随后进行了进一步排查。测试 A 中，关闭 `CC_FILTER` 后，`TwoExec` 测试通过，说明在不使用 `CC_FILTER` 的条件下，CUDA 执行路径基本具有确定性。测试 B 中，在保留 `CC_FILTER` 的情况下，即使在 `CC_FILTER` 返回后额外加入 `cudaDeviceSynchronize()`，仍然出现 `diffPixels`，说明问题不是简单的外层同步不足。测试 C 中，将 `CC_FILTER` 暂时改为 NOOP，即直接返回 child 输出，测试通过；测试 D 中，使用最小树 `CC_FILTER(CANNY(INPUT))` 仍然出现 `diffPixels`。因此可以基本确认，当前非确定性问题来源于 `executeCCFilterCUDA()` 内部，而不是 fused metrics、主 CUDA 执行框架或 CANNY 本身。

目前推测，`executeCCFilterCUDA()` 中使用的 NPP connected components 与 label compression 过程可能存在少量非确定性，或者后续基于 label 的 area、bbox、remove mask 统计流程中存在对 label 分配顺序、未初始化 buffer 或边界 component 的敏感性。由于外层同步已经不能解决问题，后续需要将 `CC_FILTER` 作为独立问题处理。可能的解决方案包括：第一，继续深入检查 `executeCCFilterCUDA()` 内部各阶段输出，包括 foreground mask、NPP label map、compressed label map、component stats、remove mask 和最终 output，确认差异首次出现在哪一步；第二，如果问题来自自定义 stats 或 remove mask kernel，则通过完整初始化 buffer、固定 label 范围、明确 label 0 处理等方式修复；第三，如果确认问题来自 NPP connected components 或 label compression 本身，则需要考虑绕开 NPP，改写为确定性的 CUDA connected-component labeling，或者在正式实验中将 `CC_FILTER` 改为 CPU fallback。

综合当前进度，第二阶段的 fused metrics 本身已经验证通过，但在 `ENABLE_CC_FILTER = 1` 的完整 GP 搜索空间下，仍被 GPU 版 `CC_FILTER` 的非确定性阻塞。因此，短期建议是：正式实验主线优先保证可重复性，保留 `ENABLE_CC_FILTER = 1`，但将 `CC_FILTER` 后端切换为 CPU deterministic fallback；与此同时，GPU 版 `CC_FILTER` 作为单独优化支线继续排查。若 CPU fallback 版本能够通过最小树 `TwoExec`、100 棵随机树 `TwoExec`、重复评分 100 次以及 2000 代无 `DROP-DETECTED` 测试，则可以将 fused metrics 合入正式实验版本。若后续希望恢复全 GPU 执行，则需要进一步实现或替换为确定性的 GPU connected components 算法，而不是仅依赖外层同步修补。

------

# 第三阶段：TERMINAL_INPUT 与 BufferPool

目标：减少无意义的 GpuMat copy 和分配开销。

------

## 3.1 修改 `TERMINAL_INPUT`

当前稳定版的 CUDA terminal 会复制输入：

```
cv::cuda::GpuMat dst;
input.copyTo(dst, getCudaStream());
return ctx.hold(dst);
```

这可以保证生命周期安全，但每个叶子都会增加一次 copy。

在阶段2 fused metrics 稳定后，尝试改成：

```
case TERMINAL_INPUT:
{
    return ctx.hold(input);
}
```

不要直接：

```
return input;
```

因为保持 `ctx.hold(input)` 更接近当前生命周期管理模式。

------

## 3.2 验证 TERMINAL_INPUT 修改是否改变结果

执行：

```
A: 原始 terminal copy
B: terminal return ctx.hold(input)
```

验证：

```
同一棵树重复评分 100 次
随机 100 棵树 old vs new score
训练 2000 代 eliteFValue
```

### 判定标准

```
score 完全一致
eliteFValue 不下降
速度有提升
```

如果结果波动，回退这个修改。

------

## 3.3 启用 OpenCV CUDA BufferPool

在程序启动后、任何 GpuMat 分配之前加入：

```
#if USE_CUDA
cv::cuda::setBufferPoolUsage(true);
cv::cuda::setBufferPoolConfig(
    cv::cuda::getDevice(),
    256 * 1024 * 1024,  // stack size，可从 128MB 或 256MB 开始
    2                   // stack count
);
#endif
```

建议放在 `main()` 中：

```
int main()
{
#if USE_CUDA
    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 256 * 1024 * 1024, 2);
#endif

    ...
}
```

### 注意

BufferPool 主要优化 OpenCV CUDA 的临时分配，对 `.cu` 中 raw `cudaMalloc/cudaFree` 的 CC_FILTER 帮助有限。

------

## 3.4 阶段3记录内容

每个版本记录：

```
time_per_generation
total_time_2000_gen
repeat_score_diff
eliteDropCount
GPU utilization
GPU memory usage
```

建议日志格式：

```
version,terminal_mode,buffer_pool,repeat_diff,elite_drop_count,time_2000_gen,gen_per_hour,best_fitness
v2,copy,off,0,0,...
v3,hold_input,off,0,0,...
v3,hold_input,on,0,0,...
```

------

# 第四阶段：单独处理 CC_FILTER

目标：确认 CC_FILTER 的实际开销，并决定是否优化它。正式实验中不能随意禁用 CC_FILTER，除非你把它作为 ablation 实验单独说明。

当前稳定版 CC_FILTER 进入前需要 `getCudaStream().waitForCompletion()`，因为 `executeCCFilterCUDA()` 内部使用 raw CUDA / NPP / default stream。 这是正确性优先的处理，但很可能是性能瓶颈。

------

## 4.1 给 CC_FILTER 加计时

在 `executeTreeCUDA()` 的 `CC_FILTER` 分支中加入计时。

```
case CC_FILTER:
{
    auto child = executeTreeCUDA(node->children[0], input, ctx);
    CV_Assert(child.type() == CV_8UC1);

    auto t0 = std::chrono::high_resolution_clock::now();

    getCudaStream().waitForCompletion();

    cv::cuda::GpuMat dst = executeCCFilterCUDA(child, node->params);

    auto t1 = std::chrono::high_resolution_clock::now();

    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    gCCFilterTimeMs += ms;
    gCCFilterCallCount++;

    return ctx.hold(dst);
}
```

在全局加：

```
double gCCFilterTimeMs = 0.0;
long long gCCFilterCallCount = 0;
```

每代结束时输出：

```
if (gCCFilterCallCount > 0) {
    printf("[CC_PROFILE] gen=%d calls=%lld total_ms=%.3f avg_ms=%.3f\n",
        idxGen,
        gCCFilterCallCount,
        gCCFilterTimeMs,
        gCCFilterTimeMs / gCCFilterCallCount);
}
```

每代结束后清零：

```
gCCFilterTimeMs = 0.0;
gCCFilterCallCount = 0;
```

------

## 4.2 统计 GP 树中 CC_FILTER 出现频率

新增函数：

```
int countNodeType(const shared_ptr<TreeNode>& node, FilterType target)
{
    if (!node) return 0;

    int count = (node->type == target) ? 1 : 0;

    for (auto& c : node->children) {
        count += countNodeType(c, target);
    }

    return count;
}
```

每代记录 elite tree 中 CC_FILTER 数量：

```
int eliteCCCount = countNodeType(curGenInfo.eliteTree, CC_FILTER);
```

输出到 CSV：

```
gen,elite_fitness,avg_fitness,dev_fitness,elite_cc_count,cc_calls,cc_total_ms,cc_avg_ms
```

------

## 4.3 诊断性禁用 CC_FILTER

只做诊断，不进入正式实验。

方式 A：生成树时不产生 CC_FILTER。使用第一阶段的 `ENABLE_CC_FILTER` 开关。

方式 B：执行时把 CC_FILTER 暂时变成 no-op：

```
#define CC_FILTER_NOOP 0

case CC_FILTER:
{
    auto child = executeTreeCUDA(node->children[0], input, ctx);

#if CC_FILTER_NOOP
    return ctx.hold(child);
#else
    getCudaStream().waitForCompletion();
    cv::cuda::GpuMat dst = executeCCFilterCUDA(child, node->params);
    return ctx.hold(dst);
#endif
}
```

两种诊断意义不同：

| 方法             | 作用                     | 是否改变搜索空间 |
| ---------------- | ------------------------ | ---------------- |
| 不生成 CC_FILTER | 测函数集合无 CC 时的速度 | 是               |
| CC_FILTER no-op  | 测跳过 CC 执行的速度影响 | 是，语义也改变   |

二者都不能直接作为实验2主结果。

------

## 4.4 后续优化方向

如果 CC_FILTER 占总时间很高，再考虑：

1. 给 `executeCCFilterCUDA()` 做 workspace 复用；
2. 避免每次 `cudaMalloc/cudaFree`；
3. 尝试让 NPP 使用指定 stream；
4. 减少 `cudaDeviceSynchronize()`；
5. 将多个 mask batch 化处理。

阶段4先只 profile，不急着改。

------

# 第五阶段：多 stream / CUDA + OpenMP

目标：只有在阶段1–4全部稳定后，才尝试恢复并行评分。

你之前的 GPU 利用率问题中，CPU 版反而比 GPU 快，而且 RTX 20xx 和 RTX 40xx 速度接近。 这说明瓶颈大概率在同步、host 调度、小 kernel 调用和 CC_FILTER，而不是 GPU 算力本身。因此多 stream 应该最后做。

------

## 5.1 准备多 stream 实验开关

加入：

```
#define CUDA_SCORE_PARALLEL 0
```

在 `getCurGenInfo()` 中改成：

```
#if USE_CUDA

#if CUDA_SCORE_PARALLEL
#pragma omp parallel for schedule(dynamic)
for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
    scoreArr[idxInd] = CAL_SCORE(population[idxInd], imgArr, -1);
}
#else
for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
    scoreArr[idxInd] = CAL_SCORE(population[idxInd], imgArr, -1);
}
#endif

#else
#pragma omp parallel for
for (int idxInd = 0; idxInd < POP_SIZE; idxInd++) {
    scoreArr[idxInd] = CAL_SCORE(population[idxInd], imgArr, -1);
}
#endif
```

当前稳定版 CUDA 分支是串行评分，CPU 分支才用 OpenMP。 所以这个改动必须用宏保护，不能直接覆盖。

------

## 5.2 限制 OpenMP 线程数

不要一开始就开满 CPU 线程。先测试：

```
OMP_NUM_THREADS=2
OMP_NUM_THREADS=4
OMP_NUM_THREADS=8
```

Windows 可以在程序中临时加：

```
omp_set_num_threads(2);
```

或者运行前设置环境变量。

------

## 5.3 多 stream 正确性验证

每个线程会使用自己的 `thread_local cv::cuda::Stream` 和 filter cache。理论上这有利于并发，但如果 CC_FILTER 内部仍有 default stream / global sync，可能反而拖慢或重新引入非确定性。

每个线程数都要执行：

```
重复评分 100 次
训练 2000 代
eliteDropCount 检查
速度统计
```

建议测试矩阵：

```
version,cc_filter,threads,repeat_diff,elite_drop_count,gen_per_hour,best_fitness
v5,on,1,...
v5,on,2,...
v5,on,4,...
v5,on,8,...
v5,off,1,...
v5,off,2,...
v5,off,4,...
v5,off,8,...
```

------

## 5.4 判定标准

只有同时满足下面条件，才保留 CUDA+OpenMP：

```
repeat_score_diff <= 1e-9
eliteDropCount = 0
gen_per_hour 明显提升，例如 > 15%
RTX 40xx 相对 RTX 20xx 开始出现合理优势
```

如果速度没提升，或者再次出现 elite drop，回退到串行 CUDA 评分。

------

# 实验2正式运行前的最终检查清单

等阶段1–3稳定后，阶段4–5可选优化完成，再开始正式实验2。正式实验2建议使用同一个固定版本，例如：

```
v3_terminal_bufferpool_verified
```

不要一边跑 S1，一边继续改代码。

## 训练侧检查

```
S1: TRAIN_SIZE = 5
S2: TRAIN_SIZE = 10
S3: TRAIN_SIZE = 20
S4: TRAIN_SIZE = 50
S5: TRAIN_SIZE = 100
每个条件 30 runs
每个 run 保存：
  printed_tree_sys.txt
  printed_tree_read.txt
  f_value.txt
  runtime.txt
  config.txt
```

`config.txt` 建议记录：

```
condition=S1
train_size=5
test_size=200
run_id=1
random_seed=...
USE_CUDA=1
USE_FUSED_METRICS=...
ENABLE_CC_FILTER=...
TERMINAL_INPUT_MODE=...
BUFFER_POOL=...
CUDA_SCORE_PARALLEL=...
```

------

## 测试侧检查

你现在的测试代码已经支持命令行参数和 CSV 输出，这适合批量测试。 每个 run 执行一次：

```
experiment_02_test_modify.exe S1 5 1 ^
  ./output/exp02/S1/run_01/printed_tree_sys.txt ^
  ./input/test/images ^
  ./input/test/masks ^
  ./output/exp02_test_results/results.csv ^
  ./output/exp02_test_results/S1/run_01/resImgs
```

每个条件循环 30 次。

最终 `results.csv` 再汇总：

```
condition,train_size,best_f1,mean_f1,std_f1,best_iou,mean_iou,std_iou
S1,5,...
S2,10,...
S3,20,...
S4,50,...
S5,100,...
```

------

# 推荐执行顺序总表

| 阶段  | 改动                                 | 是否改变算法语义       | 是否可进入正式实验  |
| ----- | ------------------------------------ | ---------------------- | ------------------- |
| 阶段0 | 修复测试代码、固定 baseline          | 否                     | 是                  |
| 阶段1 | 重复评分、elite drop、CC on/off 诊断 | CC off 会改变          | 只用 CC on 结果     |
| 阶段2 | fused metrics                        | 否，若实现正确         | 验证一致后可以      |
| 阶段3 | terminal input / BufferPool          | 否，若结果一致         | 验证一致后可以      |
| 阶段4 | CC_FILTER profile                    | 否                     | 可以                |
| 阶段4 | CC_FILTER no-op / disabled           | 是                     | 不可作为实验2主结果 |
| 阶段5 | CUDA+OpenMP / multi-stream           | 不应改变语义，但风险高 | 充分验证后才可以    |

最稳妥的正式实验版本建议是：

```
v3 = v0 baseline
   + fused metrics verified
   + terminal hold input verified
   + BufferPool verified
   + CUDA_SCORE_PARALLEL = 0
   + ENABLE_CC_FILTER = 1
```

也就是说，正式实验2不建议马上使用多 stream / CUDA+OpenMP 版本。先用确定性最强的版本完成 S1–S5，再把多 stream 作为后续加速实验或实现优化说明。