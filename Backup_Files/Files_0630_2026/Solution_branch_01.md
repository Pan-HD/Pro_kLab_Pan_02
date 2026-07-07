## 0. 先保存测试副本

建议不要直接覆盖当前文件，复制一份：

```
copy test_omp.cpp test_omp_debug.cpp
→　"./Debug_S01_0630_2026_v1/Debug_S01_0630_2026_v1_reDebug_01.cpp"
```

后续先在 `test_omp_debug.cpp` 上改。

------

## 1. 先修正 `getCurGenInfo()` 中 population[0] 重复评分问题

当前代码里先算了一次：

```
double firstScore = CAL_SCORE(population[0], imgArr, -1);
```

后面又在循环里重新算 `population[0]`。如果 GPU 评分有微小波动，这会让统计结果更混乱。把整个 `getCurGenInfo()` 替换成下面版本：

```
genType getCurGenInfo(vector<shared_ptr<TreeNode>>& population, Mat imgArr[][2]) {
    double scoreArr[POP_SIZE];

#if USE_CUDA
    // GPU版：单CPU线程顺序提交，避免多个CPU线程抢同一块GPU
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

    return curGenInfo;
}
```

这一步不是最终修复，只是先排除“同一代内 population[0] 被重复评分且结果不同”的干扰。

------

## 2. 加入“同一 elite 连续重评分”debug

在 `calcBias()` 函数后面、`multiProcess()` 前面加入这个函数：

```
void debugRescoreDrop(
    const vector<genType>& genInfo,
    Mat imgArr[][2],
    int numGen,
    FILE* fpDebug)
{
    if (genInfo.size() < 2) return;

    const genType& prevGen = genInfo[genInfo.size() - 2];
    const genType& curGen  = genInfo[genInfo.size() - 1];

    const double eps = 1e-9;

    if (curGen.eliteFValue + eps >= prevGen.eliteFValue) {
        return;
    }

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
```

然后在 `multiProcess()` 里打开 debug 日志。找到这些 `fopen_s()` 代码附近：

```
FILE* fl_logPrune = nullptr;
errno_t err6 = fopen_s(&fl_logPrune, "./imgs_0630_2026_v1/output/train_output/log_prune.txt", "a");
```

后面加：

```
FILE* fl_debugScore = nullptr;
errno_t err7 = fopen_s(
    &fl_debugScore,
    "./imgs_0630_2026_v1/output/train_output/debug_rescore.txt",
    "w");

if (err7 != 0 || fl_debugScore == nullptr) {
    perror("Cannot open debug_rescore.txt");
}
```

再找到这段：

```
genInfo.push_back(getCurGenInfo(population, imgArr));
double bias = calcBias(genInfo);
```

改成：

```
genInfo.push_back(getCurGenInfo(population, imgArr));
double bias = calcBias(genInfo);

debugRescoreDrop(genInfo, imgArr, numGen, fl_debugScore);
```

最后在 `multiProcess()` 结尾关闭文件。找到：

```
if (fl_logPrune) fclose(fl_logPrune);
```

后面加：

```
if (fl_debugScore) fclose(fl_debugScore);
```

然后先重新编译运行一次，不要先加 CUDA 同步。

Windows CMD 下运行：

```
set OMP_NUM_THREADS=1
test_omp_debug.exe
```

PowerShell 下运行：

```
$env:OMP_NUM_THREADS=1
.\test_omp_debug.exe
```

### 判断结果

如果 `debug_rescore.txt` 中同一棵 `prevElite` 连续 10 次评分不一致，例如：

```
prevElite run 00: 3.3020000000
prevElite run 01: 3.3014000000
prevElite run 02: 3.3019000000
```

就说明核心问题是 **GPU 评分非稳定**。

如果连续 10 次都完全一致，但 `curElite` 确实低于 `prevElite`，再查 GP 精英保留逻辑。

```
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

```
RESULT OF TEST: 连续 10 次评分不一致
---------generation: 2896---------
[DROP-DETECTED] gen=2896, prevElite=3.0115855019, curElite=3.0111213225
[RE-SCORE] previous elite tree:
  prevElite run 00: 3.0113533159
  prevElite run 01: 3.0111213225
  prevElite run 02: 3.0113533159
  prevElite run 03: 3.0115855019
  prevElite run 04: 3.0113533159
  prevElite run 05: 3.0108895215
  prevElite run 06: 3.0115611914
  prevElite run 07: 3.0113533159
  prevElite run 08: 3.0113533159
  prevElite run 09: 3.0111213225
[RE-SCORE] current elite tree:
  curElite  run 00: 3.0111213225
  curElite  run 01: 3.0110819398
  curElite  run 02: 3.0110973970
  curElite  run 03: 3.0111213225
  curElite  run 04: 3.0115611914
  curElite  run 05: 3.0115855019
  curElite  run 06: 3.0113533159
  curElite  run 07: 3.0111213225
  curElite  run 08: 3.0111213225
  curElite  run 09: 3.0115855019
(Res-GP)the idx of eliteInd: 10, the fitness of GP: 3.0111, the bias: 0.0783
```

------

## 3. 临时加 CUDA 同步，验证是否是异步生命周期问题

在 `getCudaStream()` 函数后面加入这个 helper：

```
#if USE_CUDA
inline cv::cuda::GpuMat syncReturnGpuMat(const cv::cuda::GpuMat& m)
{
    getCudaStream().waitForCompletion();
    return m;
}
#endif
```

然后只在 `executeTreeCUDA()` 函数内部做替换。不要全文件替换，避免影响 CPU 版 `executeTree()`。

把 `executeTreeCUDA()` 里的：

```
return dst;
```

改成：

```
return syncReturnGpuMat(dst);
```

把 Sobel 分支里的：

```
return grad8;
```

改成：

```
return syncReturnGpuMat(grad8);
```

把 CC 分支里的：

```
return executeCCFilterCUDA(child, node->params);
```

改成：

```
cv::cuda::GpuMat dst = executeCCFilterCUDA(child, node->params);
return syncReturnGpuMat(dst);
```

把 terminal 分支：

```
return input.clone();
```

建议改成：

```
cv::cuda::GpuMat dst = input.clone();
return syncReturnGpuMat(dst);
```

修改后重新编译运行：

```
set OMP_NUM_THREADS=1
test_omp_debug.exe
```

### 判断结果

如果加同步后：

```
the fitness of GP
```

不再向下波动，基本可以确认问题来自 `executeTreeCUDA()` 内部异步 CUDA 操作与局部 `GpuMat` 生命周期之间的冲突。

这一步会明显降低速度，只用于验证，不建议作为最终性能版。

```test_omp_sync_debug, Debug_S01_0630_2026_v1_reDebug_02.cpp```

- **同步验证版・実施中**

```
测试条件：
- PC: PAN_MAIN (kLab-96)
- TRAIN_SIZE = 5
- USE_CUDA = 1
- GPU型号 RTX 20
- OMP线程数 1
- 总世代数 10000
- 总耗时 (START: 15:02, 07-03， END: xx:xx, xx-xx )
- 世代/小时 xx
- 最终elite fitness xx
----------------------------------------------------
同步验证版运行至约 6400 代，未再观察到 elite fitness 向下波动；
此前同一 tree 连续重评分不一致的问题，在加 CUDA stream 同步后消失。
因此，原问题基本可归因于 executeTreeCUDA() 中异步 CUDA 操作与局部 GpuMat 生命周期不匹配。
-----------------------------------------------------
```



------

```
filename: Debug_S01_0630_2026_v1_reDebug_04.cpp
测试条件：
- PC: PAN_MAIN (kLab-96)
- TRAIN_SIZE = 5
- USE_CUDA = 1
- GPU型号 RTX 20
- OMP线程数 1
- 总世代数 2000
- 总耗时 (START: 21:10, 07-05， END: xx:xx, xx-xx )
- 世代/小时 xx
- 最终elite fitness xx
----------------------------------------------------------
测试条件：
- PC: YANG_MAIN
- TRAIN_SIZE = 5
- USE_CUDA = 1
- GPU型号 RTX 20
- OMP线程数 1
- 总世代数 10000
- 总耗时 (START: 21:10, 07-05， END: xx:xx, xx-xx )
- 世代/小时 xx
- 最终elite fitness xx
```


