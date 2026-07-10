# 4. 2000 代训练，无 DROP-DETECTED / EliteDropCount

## 目的

这一步验证正式 GP 训练主线：

```
deterministic GPU CC_FILTER
fused metrics
精英保持
选择 / 交叉 / 变异
debugRescoreDrop
完整 2000 generations
```

你的训练循环里已经在每代 `genInfo.push_back(getCurGenInfo(...))` 后调用 `debugRescoreDrop(...)`。如果当前代 elite fitness 比上一代下降，会打印 `[DROP-DETECTED]` 并重复评分 previous/current elite。

## 关闭前面所有 return 型测试

把开关设成：

```
#define RUN_TEST_RANDOM_TREE_TWOEXEC        0
#define RUN_TEST_REPEAT_SCORE_100           0
#define RUN_TEST_OLD_FUSED_SAME_OUTPUT      0
#define RUN_TEST_FULL_2000_GEN              1
```

然后在 `multiProcess()` 里改成：

```
void multiProcess(Mat imgArr[][2]) {
#if USE_CUDA
    omp_set_num_threads(1);
#endif

    initParamDesc();
    initParamDesc_safeVal();

#if USE_CUDA && RUN_TEST_RANDOM_TREE_TWOEXEC
    ...
#endif

#if USE_CUDA && RUN_TEST_REPEAT_SCORE_100
    ...
#endif

#if USE_CUDA && RUN_TEST_OLD_FUSED_SAME_OUTPUT
    ...
#endif

    // 正式训练从这里继续，不要 return
    Mat resImg[TRAIN_SIZE];
    Mat tarImg[TRAIN_SIZE];

    ...
}
```

也就是删除或注释掉现在这段：

```
auto t = makeCCFilterCannyTree();
for (int i = 0; i < 100; ++i) {
    debugCompareCCFilterOnly(t, imgArr, i);
}
return;
```

你当前文件里这段仍然在 `multiProcess()` 开头，会阻止后面的正式训练执行。

## 建议加入 EliteDropCount 计数

当前 `debugRescoreDrop()` 只打印，不返回是否 drop。建议加一个全局计数：

```
int gEliteDropCount = 0;
```

然后把 `debugRescoreDrop()` 里这段：

```
if (curGen.eliteFValue + eps >= prevGen.eliteFValue) {
    return;
}
```

后面加：

```
gEliteDropCount++;
```

完整位置：

```
if (curGen.eliteFValue + eps >= prevGen.eliteFValue) {
    return;
}

gEliteDropCount++;

printf("\n[DROP-DETECTED] gen=%d, prevElite=%.10f, curElite=%.10f\n",
    numGen + 1,
    prevGen.eliteFValue,
    curGen.eliteFValue);
```

训练结束处加总结：

```
printf("\n[TRAIN-SUMMARY] EliteDropCount=%d\n", gEliteDropCount);

if (fl_debugScore) {
    fprintf(fl_debugScore, "\n[TRAIN-SUMMARY] EliteDropCount=%d\n", gEliteDropCount);
    fflush(fl_debugScore);
}
```

## 运行前配置

正式验证建议使用：

```
#define USE_CUDA 1
#define USE_FUSED_METRICS 1
#define ENABLE_CC_FILTER 1
#define USE_FIXED_SEED 1
#define RANDOM_SEED 42
#define GENERATIONS 2000
```

`.cu` 中：

```
#define CC_FILTER_STAGE_DEBUG 0
```

否则训练会非常慢。

## 通过标准

控制台和 `debug_rescore.txt` 中必须满足：

```
没有 [DROP-DETECTED]
EliteDropCount=0
```

训练应该正常完成 2000 代，并输出最终最佳树和最佳分数。

---

## 运行情况

```
#define RUN_TEST_RANDOM_TREE_TWOEXEC        0
#define RUN_TEST_REPEAT_SCORE_100           0
#define RUN_TEST_OLD_FUSED_SAME_OUTPUT      0
#define RUN_TEST_FULL_2000_GEN              1

#define USE_CUDA 1
#define USE_FUSED_METRICS 1
#define ENABLE_CC_FILTER 1
#define GENERATIONS 2000

PC: YANG_MAIN
START_TIME: 16:13, 07-09, 2026
```

