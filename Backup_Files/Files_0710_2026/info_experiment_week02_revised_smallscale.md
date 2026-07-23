# 两个月内 ViEW 级别实验方案（Week-02 修正版：小样本主线）

> 更新目的：本文件在 `info_experiment(1).md` 的基础上再次修正。核心变化是：**不再把 TRAIN_SIZE=50 / 100 作为 ViEW 主实验条件**，因为 500/1000 代短标定已被 10000 代正式运行证明存在明显低估；ViEW 主线改为 **TRAIN_SIZE=5 / 10 / 20 的小样本 GP-only 多 seed 实验 + TS20 代表条件稳定性/模块消融 + best-tree 结构分析**。TRAIN_SIZE=50 / 100 移至参考实验或 DIA 2027 扩展。

---

## 0. 当前结论摘要

### 0.1 是否继续实验 A？

继续，但实验 A 的定义需要收窄：

```text
旧实验 A:
    TRAIN_SIZE = 5, 10, 20, 50, 100
    GP-only
    10000 generations
    各 6 run

新实验 A:
    TRAIN_SIZE = 5, 10, 20
    GP-only
    10000 generations
    各 6 run
```

### 0.2 TS50 / TS100 的处理

```text
TRAIN_SIZE=50:
    不再作为 ViEW 主实验的 6 run 条件。
    当前 pc04 上正在运行的 TS50_seed42 可作为 long-run pilot。
    如果最终完成，可作为参考结果或附录级结果；如果中止，只作为运行时间证据。

TRAIN_SIZE=100:
    不再启动 ViEW 阶段 10000代正式 run。
    移至 DIA 2027 扩展实验。
```

### 0.3 ViEW 论文定位

论文主题从“5–100 张训练图像的完整规模分析”收窄为：

```text
少数訓練画像条件における
GP によるクラック検出パイプライン最適化の実験的評価
```

中文定位：

```text
在 5～20 张少量训练图像条件下，评价 GP 自动构建裂缝检测图像处理树的性能、稳定性与结构特征。
```

---

# 一、为什么必须修正旧计划

## 1.1 500/1000 代短标定不能直接线性外推 10000 代

旧计划中的时间表主要基于 500 或 1000 代结果外推，例如 TS50 在 500 代上估算为 121.57 h/run，TS100 在 500 代上估算为 159.12 h/run。但正式训练中，tree 结构会随世代复杂化，`CAL_SCORE()` 单次开销会持续增加。

当前完整或长时间运行已经显示：

| 条件 | 机器 | seed | 状态 | total time / progress | 说明 |
|---|---|---:|---|---:|---|
| TS05 GP-only | PC00 / GTX1650 | 42 | 完成 | 20.07 h | 10000代有效，EliteDropCount=0 |
| TS05 GP-only | PC00 / GTX1650 | 43 | 完成 | 30.98 h | 同机同规模但比 seed42 慢约 54% |
| TS10 GP-only | PC03 / RTX3060 Laptop | 42 | 完成 | 56.78 h | 远高于 500代线性外推的 25.88 h |
| TS50 GP-only | pc04 / RTX4080 | 42 | 运行中 | gen3950 已约 86.3 h | sec/gen 持续上升，预计完整 run 可能超过 10 天 |

因此，后续排程必须改为：

```text
以完整 10000代 run 为主依据；
短标定只作为稳定性与早期速度参考；
每个 TRAIN_SIZE 的时间估算需要动态更新，不能用一次短标定线性外推。
```

## 1.2 同一 TRAIN_SIZE、同一机器也存在明显耗时波动

PC00 上 TS05 两次完整运行：

| TRAIN_SIZE | 机器 | seed | total_sec | total_time | sec/gen | final elite mean | EliteDropCount |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | PC00 | 42 | 72,234 | 20.07 h | 7.2234 | 0.633107 | 0 |
| 5 | PC00 | 43 | 111,532 | 30.98 h | 11.1532 | 0.625996 | 0 |

差异：

```text
30.98 h - 20.07 h = 10.91 h
11.1532 / 7.2234 ≈ 1.54x
```

说明运行时间不仅受 TRAIN_SIZE 和机器影响，还受 seed 演化出的 tree 结构影响。

## 1.3 TS50 证明长训练阶段成本上升明显

当前 TS50 / pc04 / seed42 的日志显示，累计平均 sec/gen 不是稳定值，而是随世代增加明显上升：

| generation | elapsedSec | cumulative sec/gen | eliteMean |
|---:|---:|---:|---:|
| 150 | 4,501.843 | 30.0123 | 0.3993 |
| 2350 | 143,110.196 | 60.8980 | 0.5298 |
| 2750 | 179,042.578 | 65.1064 | 0.5351 |
| 3900 | 303,498.917 | 77.8202 | 0.5673 |
| 3950 | 310,700.161 | 78.6583 | 约 0.568 |

这说明：

```text
TS50 / TS100 作为 10000代 × 6 run 主实验条件，已经不适合 ViEW 两个月窗口。
```

---

# 二、当前正式代码与运行前提

继续使用稳定主线：

```text
USE_CUDA = 1
USE_FUSED_METRICS = 1
ENABLE_CC_FILTER = 1
deterministic GPU CCL fallback
MAX_DEPTH = 12  # depth 从 0 开始，等价 13 层
POP_SIZE = 100
OFFSPRING_COUNT = 20
MUTATION_RATE = 0.9
```

实验 A 主跑必须保持：

```text
GA = OFF
PRUNING = OFF
WEEK01_ENABLE_GA = 0
WEEK01_ENABLE_PRUNING = 0
GENERATIONS = 10000
```

含 GA 的实验仍必须使用：

```text
lowmemGA_cooldownfix
```

不再使用：

```text
safeGA
```

---

# 三、ViEW 2026 主实验组合（Week-02 修正版）

## Experiment 1：小样本训练图像数影响（主实验）

这是论文最核心的定量实验。

| 项目 | 设置 |
|---|---|
| TRAIN_SIZE | 5, 10, 20 |
| Generations | 10000 |
| Method | GP-only |
| run 数 | 各 6 run |
| seed | 42, 43, 44, 45, 46, 47 |
| 测试集 | 100 或 200 张，固定不变 |
| 指标 | Precision, Recall, F1, IoU, mean, std |

目标回答的问题：

```text
在 5～20 张极少量训练图像条件下，增加训练图像数是否能提高 GP 自动生成裂缝检测 pipeline 的性能和稳定性？
```

输出图表：

```text
1. TRAIN_SIZE vs test F1 mean ± std
2. TRAIN_SIZE vs test IoU mean ± std
3. TRAIN_SIZE vs train final elite mean
4. representative fitness curves
5. per-run scatter / box plot
```

---

## Experiment 2：TS20 代表条件稳定性实验（可扩展主实验）

代表条件仍设为：

```text
TRAIN_SIZE = 20
GENERATIONS = 10000
GP-only
```

执行层级：

| 层级 | 目标 run 数 | 用途 |
|---|---:|---|
| 最低目标 | 6 run | 与 Experiment 1 一致，足以支撑主表 |
| 推荐目标 | 10–12 run | 用于稳定性 box plot 与失败率分析 |
| 理想目标 | 20 run | 若 TS20 完整 run 耗时可接受，再扩展 |

注意：旧计划中“TS20 补足到 20 run”不再作为硬性要求。原因是 TS10 的完整 run 已显示 500 代外推偏乐观，TS20 的真实 10000 代耗时需要等待至少 1 个完整 run 后重新估计。

输出：

```text
1. test F1 / IoU mean ± std
2. best fitness curve dispersion
3. EliteDropCount = 0 验证
4. runtime mean / std / min / max
5. tree node count / depth distribution
```

---

## Experiment 3：TS20 模块消融（压缩版）

ViEW 阶段不追求完整全因子模块消融。消融实验应作为辅助结论，而不是论文主结论唯一支撑。

| Case | 方法 | ViEW 建议 | 说明 |
|---|---|---|---|
| A | GP-only | 复用 Experiment 1/2 | baseline |
| B | GP + GA lowmem/cooldownfix | 推荐保留，1–3 run | 优先 pc04 |
| C | GP + PRUNING | 推荐保留，1–3 run | 开销小，适合保留 |
| D | GP + PRUNING + GA | 谨慎，仅 1 run 或放入 future work | 当前 0.31/0.31 触发策略不作为正式配置 |

执行优先级：

```text
优先级 1: GP-only 主实验完成。
优先级 2: GP + PRUNING，TS20，至少 1–3 run。
优先级 3: GP + GA lowmem/cooldownfix，TS20，至少 1–3 run。
优先级 4: GP + PRUNING + GA，仅高阈值版本 1 run 验证；不强求作为 ViEW 主表。
```

`GP + PRUNING + GA` 若继续测试，必须使用保守触发：

```cpp
#define GA_TRIGGER_MEAN_THRESH 0.40
#define PRUNE_TRIGGER_MEAN_THRESH 0.45
#define GA_MIN_TRIGGER_GEN 200
#define PRUNE_MIN_TRIGGER_GEN 300
```

如果代码暂时没有 `GA_MIN_TRIGGER_GEN` / `PRUNE_MIN_TRIGGER_GEN`，则先不把 Case D 纳入 ViEW 正式主表。

---

## Experiment 4：best-tree 结构分析（主实验的一部分）

该实验不新增训练时间，应作为 ViEW 的重要分析内容。

分析对象：

```text
Experiment 1 所有 TS5/TS10/TS20 best tree
Experiment 2 的 TS20 best tree
Experiment 3 的模块消融 best tree
```

分析内容：

```text
1. 节点数 node count
2. 最大深度 max depth
3. operator frequency
4. CC_FILTER / CANNY / SOBEL / BILATERAL / MORPHOLOGY / BITWISE 使用频度
5. 高性能 tree 中常见 operator 组合
6. test F1 与 node count / depth 的关系
7. runtime 与 tree complexity 的关系
```

建议图表：

```text
Figure: operator frequency bar chart
Figure: node count vs test F1 scatter
Figure: depth distribution by TRAIN_SIZE
Table: representative best trees
```

---

## Experiment 5：测试集定性结果

从每个 TRAIN_SIZE 中选择：

```text
best run
median run
failure-like / low-score run
```

展示：

```text
input image
ground truth
GP output mask
overlay result
false positive / false negative visualization
```

---

# 四、TS50 / TS100 的新处理方式

## 4.1 TS50

当前正在 pc04 上运行的：

```text
TS50_GP_G10000_seed42
```

处理建议：

```text
1. 不再启动新的 TS50 run。
2. 当前 run 若能完成，可作为参考实验：TRAIN_SIZE=50 single-run reference。
3. 若当前 run 占用 pc04 过久，可在合适节点中止，并作为 runtime pilot log 使用。
4. TS50 不进入 ViEW 主性能趋势图的 mean±std 主表。
```

可以在 Discussion 中写：

```text
TRAIN_SIZE=50 では木構造の複雑化に伴い学習時間が大幅に増加したため，
本稿では TRAIN_SIZE=5～20 の少数訓練画像条件に焦点を当て，
より大規模な訓練画像数条件は今後の課題とした。
```

## 4.2 TS100

```text
ViEW 阶段不启动 TS100 / 10000 generations。
TS100 移至 DIA 2027。
```

---

# 五、修正后的最低完成条件

ViEW 投稿所需的最低实验包：

```text
1. TS5 GP-only 10000代 × 6 run
2. TS10 GP-only 10000代 × 6 run
3. TS20 GP-only 10000代 × 6 run
4. 所有 best tree 在固定测试集上评价
5. best-tree 结构分析
6. 至少一个模块补充实验：GP + PRUNING 或 GP + GA
7. 定性检测结果图
```

推荐完成包：

```text
1. TS5/TS10/TS20 GP-only 各 6 run
2. TS20 GP-only 扩展到 10–12 run
3. TS20 GP + PRUNING 3 run
4. TS20 GP + GA lowmem/cooldownfix 3 run
5. TS50 当前 run 若完成，作为 single-run reference
```

理想完成包：

```text
1. 推荐完成包全部完成
2. TS20 GP-only 扩展到 20 run
3. GP + PRUNING + GA 高阈值版本 1–3 run
```

---

# 六、机器分配（Week-02 修正版）

| 机器 | 配置 | 新任务定位 |
|---|---|---|
| PC00 | i7-9750H + GTX1650 | TS05 专用，继续 seed44–47 |
| PC03 | i7-11800H + RTX3060 Laptop | TS10 主力，但按 55–60 h/run 保守估计 |
| pc01 | RTX2060 SUPER | TS20 GP-only 或 TS10 分担 |
| pc02 | RTX2080 SUPER | TS20 GP-only、PRUNING-only；不再长期占用 TS50 |
| pc04 | RTX4080 | TS20 长 run、GA 相关、当前 TS50 pilot；不再新开 TS100 |

调度原则：

```text
1. PC00 只跑 TS05。
2. PC03 继续 TS10，但不要把 TS10 全部依赖 PC03；pc01/pc02 空闲时可分担。
3. pc04 优先保留给 TS20 和 GA，不再启动新的 TS50/TS100。
4. 当前 TS50 是否继续，取决于 pc04 是否被主实验急需。
```

---

# 七、两个月执行计划（修正版）

## 第 2–3 周：主实验重新聚焦

目标：

```text
1. TS05 完成 6 run。
2. TS10 至少完成 3 run，理想完成 6 run。
3. TS20 至少启动 2–3 个 seed，尽快得到第一个完整 10000代实测耗时。
4. 不再启动新的 TS50 / TS100。
```

当前 TS50：

```text
若 pc04 暂时不急需，可继续观察到 gen5000 或完成。
若 pc04 需要转入 TS20/GA，则停止 TS50，并保存 log 作为 runtime pilot。
```

## 第 4–5 周：完成 TS5/TS10/TS20 主表

目标：

```text
TS5: 6/6
TS10: 6/6
TS20: 6/6
```

同时开始测试集评估脚本，对已完成 best tree 随完成随评估。

## 第 5–6 周：模块消融与结构统计

按优先级：

```text
1. TS20 GP + PRUNING: 1–3 run
2. TS20 GP + GA lowmem/cooldownfix: 1–3 run
3. TS20 GP + PRUNING + GA 高阈值版本: 仅 1 run 验证，可选
```

若主实验还没完成，模块消融让位于 TS5/TS10/TS20 GP-only。

## 第 6–7 周：测试集评价与作图

输出：

```text
1. test F1 / IoU table
2. training final elite mean table
3. runtime table: mean/std/min/max
4. fitness curves
5. operator frequency
6. tree depth/node count
7. qualitative examples
```

## 第 8 周：论文整理

第 8 周原则：

```text
不再启动任何 10000代新 run。
只做缺失测试、图表修正、异常日志确认、论文初稿整理。
```

---

# 八、运行与数据管理规则

## 8.1 每个 run 必须独立保存

文件名必须包含：

```text
TRAIN_SIZE
seed
machine name
method
```

例如：

```text
TS10_GP_G10000_seed43_PC03_summary.csv
TS10_GP_G10000_seed43_PC03_f_value.txt
TS10_GP_G10000_seed43_PC03_printed_tree_sys.txt
TS10_GP_G10000_seed43_PC03_printed_tree_read.txt
```

## 8.2 label 与 seed 必须一致

之前 TS05 seed43 的 summary 中出现：

```text
label = week02_TS05_GP_G10000_seed42
random_seed = 43
```

后续必须修正，避免汇总时混乱。

## 8.3 建议追加 checkpoint

长 run 尤其 TS20 以上建议添加：

```text
每 500 或 1000 代保存：
    current elite tree
    f_value partial
    generation
    elapsedSec
    segment sec/gen
    node count
    max depth
```

否则长 run 中途停止时，无法保留当前最优 tree。

## 8.4 建议追加分段耗时日志

当前已有累计 `secPerGen`，但建议再增加 segment 速度：

```text
[SEG-TIME] gen=1000 segmentSec=... segmentSecPerGen=...
```

用于论文分析：

```text
GP tree complexity growth vs runtime increase
```

---

# 九、论文结构（修正版）

建议论文结构：

```text
1. Introduction
2. GP-based crack detection pipeline optimization
3. Deterministic GPU implementation
4. Experimental setup
   4.1 Dataset and small training-size setting
   4.2 GP parameters
   4.3 Evaluation metrics
   4.4 Runtime measurement protocol
5. Results
   5.1 Effect of training size under small-sample conditions
   5.2 Stability across random seeds
   5.3 Ablation study on TS20
   5.4 Tree structure and operator frequency analysis
   5.5 Qualitative examples
6. Discussion
   6.1 Why TS50/TS100 were excluded from the main ViEW experiment
   6.2 Runtime growth caused by GP tree complexity
   6.3 Limitations and future work
7. Conclusion
```

Discussion 中明确说明：

```text
本研究聚焦 TRAIN_SIZE=5～20 的少数训练图像条件。
TRAIN_SIZE=50/100 的大规模条件由于 10000代训练中 tree 复杂度导致训练时间显著增加，
作为 DIA 2027 的扩展课题。
```

---

# 十、DIA 2027 扩展计划

ViEW 之后扩展：

```text
1. TRAIN_SIZE=50 / 100 的完整 10000代、多 seed 实验
2. TS20/TS50 的完整模块消融
3. Elite Injection 稳定版本加入消融
4. 8 组或更多全因子组合：
   GP
   GP + PRUNING
   GP + GA
   GP + Elite Injection
   GP + PRUNING + GA
   GP + PRUNING + Elite Injection
   GP + GA + Elite Injection
   GP + PRUNING + GA + Elite Injection
5. GP vs CNN 正式比较
6. tree complexity 控制或 parsimony pressure
7. checkpoint / resume 机制
8. CUDA batch 化与 memory pool 优化
```

---

# 最终结论

Week-02 修正后的 ViEW 方案是：

```text
主线：
    TS5 / TS10 / TS20
    GP-only
    10000 generations
    各 6 run

补充：
    TS20 稳定性扩展到 10–12 run，时间允许再到 20 run
    TS20 GP + PRUNING / GP + GA lowmem/cooldownfix 各 1–3 run
    TS50 当前 run 只作为参考或 runtime pilot

移出 ViEW 主线：
    TS50 × 6 run
    TS100 × 6 run
    GP + PRUNING + GA 的完整多 run 消融
    Elite Injection
```

这样仍然足以支撑 ViEW 论文，因为论文问题被明确限定为：

```text
少数训练图像条件下，GP 是否能稳定自动构建裂缝检测 pipeline，
以及训练图像数从 5 到 20 增加时性能与结构如何变化。
```

与原计划相比，新方案更聚焦、更可完成，也更符合当前 10000 代正式运行暴露出的真实计算成本。
