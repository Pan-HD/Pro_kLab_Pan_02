# 两个月内 ViEW 级别实验方案（Week-01 标定结果反映版）

> 更新目的：本文件根据第一周实际运行结果修正此前 `info_experiment.md` 中基于粗略线性模型的实验规划。此前规划中的核心方向仍保留：**数据规模趋势 + 代表条件稳定性 + 压缩版模块消融 + 木结构分析**。主要变化是：运行时间估算改用实测值，GA 代码限定为 `lowmemGA_cooldownfix` 版本，`GP + PRUNING + GA` 不再使用过早触发的原始 0.31/0.31 策略作为正式配置。

---

## 0. 当前实验环境与稳定代码前提

当前正式训练主线采用：

```text
USE_CUDA = 1
USE_FUSED_METRICS = 1
ENABLE_CC_FILTER = 1
deterministic GPU CCL fallback
MAX_DEPTH = 12  # depth 从 0 开始计数，等价 13 层
```

第一周所有可完成的标定结果均满足：

```text
EliteDropCount = 0
```

因此当前 stable CUDA/GP 路径可继续作为 ViEW 实验基准。需要注意的是，含 GA 的旧版 `safeGA` 曾出现 GA 连续触发、单次 GA 耗时失控和显存压力问题；后续 GA 实验必须使用：

```text
lowmemGA_cooldownfix
MED_BLUR k 上限限制
禁用 MedianFilter cache
GA 成功或失败后都进入 cooldown
```

---

# 一、Week-01 实测结果摘要

## 1.1 TRAIN_SIZE 时间标定（GP-only）

该组实验用于替代旧规划中的粗略估算 `10000代 / run ≈ 2.75 × TRAIN_SIZE 小时`。由于各 TRAIN_SIZE 在不同机器上运行，下表更适合用于实际排程，而不是纯粹的算法复杂度分析。

| TRAIN_SIZE | 机器 | GPU | 标定代数 | 实测耗时 h | sec/gen | gen/min | 10000代 / run 外推 h | 6 run 外推 GPU h | final elite mean | EliteDropCount |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | pc02 | RTX 2080 SUPER | 500 | 0.73 | 5.2442 | 11.4413 | 14.57 | 87.40 | 0.512271 | 0 |
| 10 | pc03 | RTX 3060 Laptop | 500 | 1.29 | 9.3166 | 6.4401 | 25.88 | 155.28 | 0.434545 | 0 |
| 20 | pc04 | RTX 4080 | 1000 | 4.61 | 16.5901 | 3.6166 | 46.08 | 276.50 | 0.476020 | 0 |
| 50 | pc02 | RTX 2080 SUPER | 500 | 6.08 | 43.7659 | 1.3709 | 121.57 | 729.43 | 0.453085 | 0 |
| 100 | pc04 | RTX 4080 | 500 | 7.96 | 57.2837 | 1.0474 | 159.12 | 954.73 | 0.469061 | 0 |


> 注：TRAIN_SIZE=20 的正式排程估算优先采用 pc04 上 1000 代 GP-only 结果；pc01 上 500 代 GP-only 结果为 51.04 h / 10000代，可作为保守参考。

基于以上实测值，实验 A 的 5 个 TRAIN_SIZE、各 6 run 的总训练量更新为：

```text
实验 A ≈ 2203.3 GPU小时
```

旧规划估算为 3052.5 GPU小时；更新后约减少：

```text
3052.5 - 2203.3 = 849.2 GPU小时
```

---

## 1.2 pc04 上 TS20 模块标定结果

以下 4 组是在 pc04 / RTX 4080 上的同机结果，适合用于模块耗时倍率估算。

| 条件 | 1000代耗时 h | sec/gen | final elite mean | EliteDropCount | 10000代外推 h | 相对 GP-only 倍率 |
|---|---:|---:|---:|---:|---:|---:|
| GP | 4.61 | 16.5901 | 0.476020 | 0 | 46.08 | 1.00x |
| GP + GA lowmem/cooldownfix | 4.03 | 14.4931 | 0.555968 | 0 | 40.26 | 0.87x |
| GP + PRUNING | 3.97 | 14.2982 | 0.485614 | 0 | 39.72 | 0.86x |
| GP + PRUNING + GA lowmem/cooldownfix | 9.76 | 35.1276 | 0.465057 | 0 | 97.58 | 2.12x |


模块统计重点：

| 条件 | GA trigger | GA success | GA fail | GA blocked | GA total sec | GA avg sec | GA max sec | Prune attempt | Prune success | Prune total sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GP + GA lowmem/cooldownfix | 5 | 5 | 0 | 716 | 843.261 | 168.652 | 447.646 | 0 | 0 | 0 |
| GP + PRUNING | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 202 | 14 | 39.952 |
| GP + PRUNING + GA lowmem/cooldownfix | 5 | 5 | 0 | 952 | 3753.500 | 750.700 | 971.447 | 198 | 18 | 61.938 |

结论：

```text
GP + GA lowmem/cooldownfix：稳定，EliteDropCount=0，本次 final elite mean 最高。
GP + PRUNING：稳定，开销很小，本次 final elite mean 略高于 GP-only。
GP + PRUNING + GA：稳定完成，但耗时约为 GP-only 的 2.12 倍，且本次 final elite mean 低于 GP-only。
```

---

# 二、对旧规划的主要修正

## 2.1 耗时模型修正

旧模型：

```text
10000代 / run ≈ 2.75 × TRAIN_SIZE 小时
```

更新模型：优先使用实测机器排程值。

| TRAIN_SIZE | 旧估算 h/run | 当前推荐排程估算 h/run |
|---:|---:|---:|
| 5 | 13.75 | 14.57 |
| 10 | 27.50 | 25.88 |
| 20 | 55.00 | 46.08 |
| 50 | 137.50 | 121.57 |
| 100 | 275.00 | 159.12 |

当前 GPU-hours 估算明显低于旧规划，尤其是 pc04 上 TRAIN_SIZE=100 的标定结果显著改善总排程。

## 2.2 模块消融内容修正

旧规划中的压缩版模块消融包含：

```text
GP
GP + GA
GP + Pruning + Elite Injection
GP + Pruning + Elite Injection + GA
```

根据当前已经完成标定的代码路径，ViEW 主线建议改为：

```text
GP
GP + GA lowmem/cooldownfix
GP + PRUNING
GP + PRUNING + GA lowmem/cooldownfix
```

Elite Injection 暂不放入 ViEW 主线，除非后续单独完成稳定性和耗时标定。完整 Elite Injection 与 8 组全因子消融更适合 DIA 2027。

## 2.3 GA / PRUNING 触发策略修正

当前 `GP + PRUNING + GA` 使用的：

```text
GA_TRIGGER_MEAN_THRESH = 0.31
PRUNE_TRIGGER_MEAN_THRESH = 0.31
```

虽然能稳定完成，但结果显示：

```text
耗时高：M_PRUNING_GA ≈ 2.12x
收益不稳定：final elite mean = 0.465057，低于 GP-only 的 0.476020
```

因此判断：在 `GP + PRUNING + GA` 条件下，0.31/0.31 很可能导致 GA 和 PRUNING 过早介入，造成搜索结构过早收敛和 GA 优化对象变重。

建议后续正式 `GP + PRUNING + GA` 先追加一个 1000 代重标定版本：

```cpp
#define GA_TRIGGER_MEAN_THRESH 0.40
#define PRUNE_TRIGGER_MEAN_THRESH 0.45
#define GA_MIN_TRIGGER_GEN 200
#define PRUNE_MIN_TRIGGER_GEN 300
```

可选更保守版本：

```cpp
#define GA_TRIGGER_MEAN_THRESH 0.40
#define PRUNE_TRIGGER_MEAN_THRESH 0.50
#define GA_MIN_TRIGGER_GEN 200
#define PRUNE_MIN_TRIGGER_GEN 300
```

如果重标定结果优于当前 0.31/0.31 版本，则 ViEW 正式实验采用高阈值 + 最小世代限制版本。

---

# 三、ViEW 2026 推荐实验组合（修正版）

## 实验 A：训练图像数对性能影响

这是 ViEW 版最核心的实验。

| 条件 | 内容 |
|---|---|
| TRAIN_SIZE | 5, 10, 20, 50, 100 |
| 世代数 | 10000 |
| run 数 | 各 6 run |
| 方法 | stable GP-only |
| 测试图像 | 100 或 200 张 |
| 指标 | Precision, Recall, F1, IoU, Mean, Std |
| 预计训练量 | 约 2203.3 GPU小时 |

建议优先完成 5/10/20/50。TRAIN_SIZE=100 应尽早安排在 pc04 上运行，但如果后期时间不足，可作为参考条件而非主结论唯一依据。

---

## 实验 B：代表条件下稳定性实验

代表条件仍建议：

```text
TRAIN_SIZE = 20
GENERATIONS = 10000
GP-only
```

实验 A 中 TRAIN_SIZE=20 已有 6 run，因此追加 14 run：

```text
14 × 46.08 h = 645.2 GPU小时
```

目标：

```text
TRAIN_SIZE=20 GP-only 总 run 数 = 20
```

输出：

```text
F1 / IoU 的 mean, std, box plot
best fitness curve
异常 run 检查
EliteDropCount 检查
```

---

## 实验 C：压缩版模块消融实验

修正后的 ViEW 消融主线：

| Case | 方法 | 当前建议 | 10000代 / run 估算 |
|---|---|---|---:|
| A | GP | baseline，复用实验 A/B | 46.08 h |
| B | GP + GA lowmem/cooldownfix | 推荐保留 | 40.26 h |
| C | GP + PRUNING | 推荐保留 | 39.72 h |
| D | GP + PRUNING + GA lowmem/cooldownfix | 谨慎保留，先调高触发阈值后重标定 | 当前 0.31/0.31 版本为 97.58 h |

初始执行方案：

```text
TRAIN_SIZE = 20
GENERATIONS = 10000
Case B: 3 run
Case C: 3 run
Case D: 先 1 run 验证高阈值版本；通过后扩展到 3 run
```

如果按当前 pc04 标定直接估算，Case B/C/D 各 3 run 的新增训练量约：

```text
3 × 40.26 + 3 × 39.72 + 3 × 97.58
= 532.7 GPU小时
```

但 Case D 建议先使用高阈值重标定后再决定是否扩到 3 run。

---

## 实验 D：木结构与算子频度分析

该实验复用所有 best tree，几乎不增加训练时间。

分析内容：

```text
1. 高性能个体中的 operator 出现频度
2. CC_FILTER / CANNY / Sobel / Morphology / Bitwise 的使用频度
3. 常见 operator 组合
4. 木深度
5. 节点数
6. PRUNING 是否降低节点数与深度
7. GA 是否改变参数分布
```

---

# 四、更新后的总耗时估算

基础版本：

```text
实验 A：2203.3 GPU小时
实验 B：645.2 GPU小时
实验 C 初始版：532.7 GPU小时
实验 D：基本不新增训练时间
```

合计：

```text
2203.3 + 645.2 + 532.7 = 3381.2 GPU小时
```

若 Case B 与 Case C 从 3 run 扩展到 6 run，而 Case D 仍保持 3 run：

```text
约 3621.1 GPU小时
```

以 6 台机器、80% 有效率估算：

| 方案 | GPU小时 | 6台×80% 有效率所需天数 | 5台×80% 有效率所需天数 |
|---|---:|---:|---:|
| A+B+C 初始版 | 3381.2 | 29.4 天 | 35.2 天 |
| B/C 扩展到 6 run | 3621.1 | 31.4 天 | 37.7 天 |

与旧规划中的 4500–5600 GPU小时相比，当前修正版主实验量约为 3400–3650 GPU小时；即使考虑失败补跑、测试集评估和日志整理，两个月内完成 ViEW 版主实验更现实。

---

# 五、机器分配建议

| 机器 | 配置 | 推荐任务 |
|---|---|---|
| pc04 | RTX 4080 | TRAIN_SIZE=100、所有 GA 相关长任务、TS20 模块标定与补跑 |
| pc02 | RTX 2080 SUPER | TRAIN_SIZE=50、TRAIN_SIZE=5、Pruning-only 补跑 |
| pc01 | RTX 2060 SUPER | TRAIN_SIZE=20 GP-only 稳定性 run |
| pc03 | RTX 3060 Laptop | TRAIN_SIZE=10、短 run、日志检查；不建议承担 GA 长任务 |

原则：

```text
1. GA 任务优先 pc04。
2. TRAIN_SIZE=100 尽早启动，避免拖到最后一周。
3. pc03 避免跑 PRUNING+GA 或长 GA 任务。
4. 每个 run 独立 seed、独立输出目录，避免覆盖 CSV 和 best tree。
```

---

# 六、两个月执行计划（修正版）

## 第 1 周：已完成的标定与结论

已完成：

```text
1. TRAIN_SIZE=5/10/20/50/100 GP-only 时间标定
2. pc04 上 TS20_GP_1000
3. pc04 上 TS20_GA_1000 lowmemGA_cooldownfix
4. pc04 上 TS20_PRUNING_1000
5. pc04 上 TS20_PRUNING_GA_1000 lowmemGA_cooldownfix
```

关键结论：

```text
1. 全部完成项 EliteDropCount=0。
2. GA-only lowmem/cooldownfix 可用，且本次 final elite mean 最高。
3. Pruning-only 可用，开销很小。
4. Pruning+GA 可运行，但 0.31/0.31 触发策略耗时高且收益不稳定。
```

下一步优先补：

```text
TS20_PRUNING_GA_1000，高阈值 + 最小世代限制版本
```

---

## 第 2–4 周：实验 A 主跑

优先启动长任务：

```text
TRAIN_SIZE=100
TRAIN_SIZE=50
TRAIN_SIZE=20
```

最低完成目标：

```text
TRAIN_SIZE=5/10/20 各 6 run
TRAIN_SIZE=50 至少 3 run
TRAIN_SIZE=100 至少 2 run
```

理想完成目标：

```text
TRAIN_SIZE=5/10/20/50/100 各 6 run
```

**运行记录：**

```
TS05
  RUN01
    pc00(PAN_NOTEBOOK)
  RUN02
  RUN03
  RUN04
  RUN05
  RUN06
TS10
  RUN01
    pc03(YANG_NOTEBOOK)
  RUN02
  RUN03
  RUN04
  RUN05
  RUN06
TS20
  RUN01
    pc02(PAN_MAIN)
  RUN02
  RUN03
  RUN04
  RUN05
  RUN06
TS50
  RUN01
    pc04(FENG_MAIN)
  RUN02
  RUN03
  RUN04
  RUN05
  RUN06
TS100
  RUN01
    pc(YANG_MAIN)
  RUN02
  RUN03
  RUN04
  RUN05
  RUN06
```



---

## 第 5 周：实验 B 与实验 C

进行：

```text
1. TRAIN_SIZE=20 GP-only 补足到 20 run
2. GP + GA lowmem/cooldownfix，3 run
3. GP + PRUNING，3 run
4. GP + PRUNING + GA，高阈值版本，先 1 run，通过后扩到 3 run
```

判断 Case D 是否继续扩展的标准：

```text
1. EliteDropCount = 0
2. total_sec 不超过 GP-only 的 2.0–2.2 倍
3. final elite mean 不低于 GP-only 或 Pruning-only
4. GA 单次触发不超过 1 小时
5. 无 GPU OOM / MedianFilter / wavelet_matrix 错误
```

---

## 第 6 周：测试集评估与补跑

对所有 best tree 运行测试集：

```text
Precision
Recall
F1
IoU
推理时间
节点数
树深度
```

同时检查：

```text
EliteDropCount > 0 的 run 直接作废并补跑
日志损坏或 CSV 缺字段的 run 补跑
异常低分 run 保留但标注，不随意删除
```

---

## 第 7 周：统计与作图

生成：

```text
1. TRAIN_SIZE vs F1 mean/std
2. TRAIN_SIZE vs IoU mean/std
3. TRAIN_SIZE=20 stability box plot
4. best fitness curve
5. module ablation bar chart
6. operator frequency bar chart
7. tree depth / node count table
8. runtime table
```

建议论文图表：

```text
Table 1: GP parameters and GPU implementation settings
Table 2: dataset split
Table 3: runtime calibration
Table 4: TRAIN_SIZE performance
Table 5: ablation result
Figure 1: method overview
Figure 2: fitness curve
Figure 3: performance vs training size
Figure 4: operator frequency
Figure 5: qualitative examples
```

---

## 第 8 周：论文整理与小规模补跑

论文结构：

```text
1. Introduction
2. GP-based crack detection pipeline optimization
3. Deterministic GPU implementation
4. Experimental setup
5. Results
   5.1 Effect of training set size
   5.2 Stability analysis
   5.3 Ablation study
   5.4 Operator and tree-structure analysis
6. Discussion
7. Conclusion
```

第 8 周原则：

```text
不再启动 TRAIN_SIZE=50/100 的长 run。
只做缺失 seed、测试集评估、异常日志补跑和作图修正。
```

---

# 七、正式运行规则与停止条件

## 7.1 必须使用的 GA 版本

```text
lowmemGA_cooldownfix
```

不再使用：

```text
safeGA
```

原因：safeGA 已出现 GA 连续触发和单次 GA 耗时失控。

## 7.2 GA 停止/中断规则

出现以下情况时停止当前 run，不继续等待：

```text
1. GA 成功后下一代仍立即 GA-TRIGGER，说明 cooldown 失效。
2. 单次 GA 超过 1 小时仍未结束。
3. 出现 GPU Memory Alloc Error / out of memory。
4. 出现 OpenCV MedianFilter windowSize 错误。
5. 出现 wavelet_matrix 相关 OOM。
```

## 7.3 有效 run 判定

有效 run 必须满足：

```text
1. 正常完成指定 generations。
2. EliteDropCount = 0。
3. summary CSV 完整。
4. best tree 文件存在。
5. 测试集评估脚本可正常读取输出。
```

---

# 八、ViEW 最终实验清单（修正版）

```text
Experiment 1:
    TRAIN_SIZE = 5, 10, 20, 50, 100
    GP-only
    6 run each
    10000 generations

Experiment 2:
    TRAIN_SIZE = 20
    GP-only
    total 20 run
    stability analysis

Experiment 3:
    TRAIN_SIZE = 20
    GP
    GP + GA lowmem/cooldownfix
    GP + PRUNING
    GP + PRUNING + GA lowmem/cooldownfix with conservative trigger schedule
    3 run each initially

Experiment 4:
    Best-tree structural analysis
    operator frequency
    tree depth
    node count
    GA parameter distribution

Experiment 5:
    Qualitative examples on test images
```

---

# 九、DIA 2027 扩展内容

ViEW 不做或只做简化，DIA 2027 再扩展：

```text
1. 各 TRAIN_SIZE 从 6 run 扩展到 20 或 30 run
2. 完整模块消融，包括 Elite Injection
3. 完整 8 组或更多组合：
   GP
   GP + PRUNING
   GP + GA
   GP + Elite Injection
   GP + PRUNING + GA
   GP + PRUNING + Elite Injection
   GP + GA + Elite Injection
   GP + PRUNING + GA + Elite Injection
4. TRAIN_SIZE=50 下的完整模块消融
5. GP vs CNN 正式比较
6. GA 触发次数、成功率、耗时收益分析
7. Pruning 对节点数、深度、推理时间的影响
8. GA/Pruning 触发阈值敏感性分析
```

---

# 最终结论

根据 Week-01 实测结果，ViEW 级别两个月实验方案应从旧版的 4500–5600 GPU小时目标修正为：

```text
主实验初始版：约 3381 GPU小时
B/C 扩展版：约 3621 GPU小时
```

在 5–6 台 GPU 电脑并行运行的条件下，两个月内完成主实验、测试集评估、作图和论文初稿是可行的。

当前最重要的策略调整是：

```text
1. 使用实测时间表排程，不再使用 2.75 × TRAIN_SIZE 的粗略模型。
2. GA 只能使用 lowmemGA_cooldownfix。
3. GP + PRUNING + GA 不再使用 0.31/0.31 作为正式触发策略，需先测试高阈值 + 最小世代限制版本。
4. ViEW 主线优先保证 GP、GP+GA、GP+PRUNING 三组；GP+PRUNING+GA 作为谨慎扩展。
```
