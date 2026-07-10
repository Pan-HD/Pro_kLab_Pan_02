好的，请整理生成 两个月内的 ViEW 级别实验方案。

可以。下面整理成一个**两个月内可完成的 ViEW 2026 级别实验方案**，目标是：先完成一套能投稿的核心实验，而不是把原始规划中所有 30 run × 全条件一次性跑完。

ViEW 2026 的開催日は 2026年12月3–4日，申込締切是 2026年9月11日，因此两个月实验期内需要同时考虑“训练、测试集评估、作图、论文初稿”这几件事。  原始方案中包含实验1到实验8，例如训练图像数影响、GP/CNN比较、稳定性、Pruning、Elite Injection、模块贡献、木结构分析等，其中很多实验默认 30 次运行、10000 代、训练图像 50 或 100 张。

## 两个月 ViEW 版总体定位

建议 ViEW 2026 的投稿主题定为：

```
クラック検出アルゴリズムのGPによる最適化と
小規模訓練データ条件における性能評価
```

核心主张放在三点：

```
1. GP 可以自动组合传统图像处理算子，构建 crack detection pipeline。
2. 在少量训练图像条件下，GP 的检测性能随训练数据规模变化具有可分析趋势。
3. 引入 deterministic GPU CC_FILTER 后，GPU 版训练稳定，适合大规模重复实验。
```

ViEW 阶段不建议把论文重点放在“完整 8 组全因子消融 + 30 run 统计”上。那更适合 DIA 2027。

------

# 一、实验规模设定

当前实测基准：

```
TRAIN_SIZE = 5
GENERATIONS = 2000
耗时 = 165 分钟
```

换算：

```
10000代，TRAIN_SIZE=5 ≈ 13.75 小时 / run
10000代单次运行小时数 ≈ 2.75 × TRAIN_SIZE
```

因此：

| TRAIN_SIZE | 10000代 / 1 run 预估 |
| ---------- | -------------------- |
| 5          | 13.75 h              |
| 10         | 27.5 h               |
| 20         | 55 h                 |
| 50         | 137.5 h              |
| 100        | 275 h                |

你有 5–6 台 20 系列以上 GPU 电脑。两个月理论容量约为：

```
5台：7200 GPU小时
6台：8640 GPU小时
```

考虑失败补跑、机器占用、日志整理，建议只按 70–85% 计算：

```
实际可用：约 5000–7300 GPU小时
```

所以两个月内可行的目标是 **约 4500–6000 GPU小时以内的核心实验包**。

------

# 二、ViEW 2026 推荐实验组合

## 实验 A：训练图像数对性能的影响

这是 ViEW 版最核心的实验。

| 条件       | 内容                                      |
| ---------- | ----------------------------------------- |
| TRAIN_SIZE | 5, 10, 20, 50, 100                        |
| 世代数     | 10000                                     |
| run 数     | 各 6 run                                  |
| 测试图像   | 100 或 200 张                             |
| 方法       | 当前 stable GP，先不启用 GA/Pruning/Elite |
| 指标       | Precision, Recall, F1, IoU, Mean, Std     |

预计耗时：

```
(5+10+20+50+100) × 2.75 × 6
= 3052.5 GPU小时
```

这部分直接对应原方案中的“学習データ数の変化がGPの検出性能に与える影響”。

------

## 实验 B：代表条件下的稳定性实验

选择一个代表性条件，建议：

```
TRAIN_SIZE = 20
```

理由是：5/10 太小，50/100 太耗时，20 比较适合 ViEW 阶段展示稳定性和可重复性。

| 条件       | 内容                                             |
| ---------- | ------------------------------------------------ |
| TRAIN_SIZE | 20                                               |
| 世代数     | 10000                                            |
| 总 run 数  | 补足到 20 run                                    |
| 方法       | 当前 stable GP                                   |
| 指标       | F1, IoU, Mean, Std, box plot, best fitness curve |

实验 A 中 TRAIN_SIZE=20 已经有 6 run，所以这里追加：

```
14 run × 55h = 770 GPU小时
```

这部分对应原方案中的“GPの収束安定性および実行結果のばらつき”。

------

## 实验 C：压缩版模块消融实验

模块包括：

```
P = Pruning
E = Elite Injection
G = GA parameter optimization
```

完整全因子是 8 组，但 ViEW 阶段建议压缩成 4 组：

| Case | 方法                                | 目的                       |
| ---- | ----------------------------------- | -------------------------- |
| A    | GP                                  | baseline，复用实验 A/B     |
| B    | GP + GA                             | 验证 GA 参数再优化是否有效 |
| C    | GP + Pruning + Elite Injection      | 验证结构性改良是否有效     |
| D    | GP + Pruning + Elite Injection + GA | 验证完整提案方法是否有效   |

推荐条件：

```
TRAIN_SIZE = 20
GENERATIONS = 10000
先各 3 run
时间允许再扩展到各 6 run
```

由于 GP baseline 可复用，新增耗时为：

```
Case C: 3 × 55 = 165 GPU小时
Case B + D: 2 × 3 × 55 × M_GA = 330 × M_GA GPU小时
```

其中 `M_GA` 是启用 GA 后相对普通 GP 的耗时倍率。

| GA 耗时倍率 | 3 run 压缩消融新增耗时 |
| ----------- | ---------------------- |
| 1.5x        | 660 GPU小时            |
| 2.0x        | 825 GPU小时            |
| 3.0x        | 1155 GPU小时           |
| 5.0x        | 1815 GPU小时           |

ViEW 阶段建议先按 **3 run** 执行。只有当 GA 标定结果显示 `M_GA <= 3`，并且主实验进度顺利时，再扩展到 6 run。

------

## 实验 D：木结构与算子频度分析

这个实验几乎不需要额外训练，复用前面所有 best tree。

分析内容：

```
1. 高性能个体中的 operator 出现频度
2. CC_FILTER / CANNY / Sobel / Morphology / Bitwise 的使用频度
3. 常见 operator 组合
4. 木深度
5. 节点数
6. 是否 Pruning 后树结构更简单
```

这部分对应原方案中的实验8。

------

# 三、总耗时估算

基础实验：

```
实验 A：3052.5 GPU小时
实验 B：770 GPU小时
实验 C：165 + 330 × M_GA GPU小时
实验 D：基本不新增训练时间
```

合计：

```
3987.5 + 330 × M_GA GPU小时
```

| GA 倍率 M_GA | 总 GPU小时 | 6台机器理论天数 | 按80%有效率 |
| ------------ | ---------- | --------------- | ----------- |
| 1.5x         | 4482.5 h   | 31.1 天         | 38.9 天     |
| 2.0x         | 4647.5 h   | 32.3 天         | 40.4 天     |
| 3.0x         | 4977.5 h   | 34.6 天         | 43.2 天     |
| 5.0x         | 5637.5 h   | 39.1 天         | 48.9 天     |

因此，如果控制在这个规模，两个月内完成实验部分是现实的。

------

# 四、两个月执行计划

## 第 1 周：标定与准备

目标：不要直接开跑全部实验，先确认时间模型和 GA 倍率。

运行：

```
1. TRAIN_SIZE=5/10/20/50/100，各 500 或 1000 代，1 run
2. TRAIN_SIZE=20，GP only，1000代，1 run
3. TRAIN_SIZE=20，GP + GA，1000代，1 run
```

记录：

```
sec/generation
GA_TRIGGER_COUNT
GA_SUCCESS_COUNT
GA_TOTAL_TIME_SEC
M_GA
EliteDropCount
best fitness
```

同时准备：

```
seed list
机器分配表
输出目录结构
日志命名规则
best tree 保存格式
测试集评估脚本
CSV 汇总脚本
```

建议 seed 固定为：

```
seed = 42, 43, 44, 45, 46, 47 ...
```

------

## 第 2–4 周：实验 A 主跑

优先启动大训练集，因为单 run 长。

机器分配示例：

| 机器  | 任务                      |
| ----- | ------------------------- |
| GPU-1 | TRAIN_SIZE=100            |
| GPU-2 | TRAIN_SIZE=50             |
| GPU-3 | TRAIN_SIZE=20             |
| GPU-4 | TRAIN_SIZE=10             |
| GPU-5 | TRAIN_SIZE=5              |
| GPU-6 | 补跑 / GA 标定 / 消融预跑 |

优先完成：

```
TRAIN_SIZE=5/10/20 各6 run
TRAIN_SIZE=50 至少3 run
TRAIN_SIZE=100 至少2 run
```

如果 TRAIN_SIZE=100 进度太慢，ViEW 论文中可以把 100 作为“追加条件”或“参考条件”，主结论集中在 5/10/20/50。

------

## 第 5 周：实验 B 与实验 C

进行：

```
1. TRAIN_SIZE=20 稳定性补足到 20 run
2. 消融 Case B: GP + GA，3 run
3. 消融 Case C: GP + P + E，3 run
4. 消融 Case D: GP + P + E + GA，3 run
```

这周的关键是判断 GA 是否值得在 ViEW 中重点展开。

如果 GA 非常耗时，例如：

```
M_GA > 5
```

则 ViEW 中只保留：

```
GP vs GP+GA 的少量结果
```

完整 GA 消融移到 DIA 2027。

------

## 第 6 周：测试集评估与补跑

对所有 best tree 运行测试集：

```
test images = 100 或 200
Precision
Recall
F1
IoU
```

同时检查：

```
是否有训练未完成
是否有 EliteDropCount > 0
是否有日志损坏
是否有异常低分 run
是否需要补跑 seed
```

------

## 第 7 周：统计与作图

生成 ViEW 所需图表：

```
1. TRAIN_SIZE vs F1 mean/std
2. TRAIN_SIZE vs IoU mean/std
3. TRAIN_SIZE=20 的 box plot
4. best fitness curve
5. 消融实验柱状图
6. operator frequency bar chart
7. tree depth / node count table
8. runtime table
```

建议 ViEW 论文至少包含：

```
Table 1: GP parameters
Table 2: training/test dataset split
Table 3: TRAIN_SIZE performance
Table 4: ablation result
Figure 1: method overview
Figure 2: fitness curve
Figure 3: performance vs training size
Figure 4: operator frequency
Figure 5: qualitative detection examples
```

------

## 第 8 周：论文整理与追加补跑

论文结构建议：

```
1. Introduction
2. Proposed GP-based crack detection optimization
3. Deterministic GPU implementation
4. Experimental setup
5. Results
   5.1 Effect of training set size
   5.2 Stability analysis
   5.3 Ablation study
   5.4 Operator analysis
6. Discussion
7. Conclusion
```

这周只做小规模补跑，不再开启长 run。

------

# 五、ViEW 版最终实验清单

建议最终锁定为：

```
Experiment 1:
    TRAIN_SIZE = 5, 10, 20, 50, 100
    GP only
    6 run each
    10000 generations

Experiment 2:
    TRAIN_SIZE = 20
    GP only
    total 20 run
    stability analysis

Experiment 3:
    TRAIN_SIZE = 20
    GP
    GP + GA
    GP + Pruning + Elite Injection
    GP + Pruning + Elite Injection + GA
    3 run each, 可能なら6 run

Experiment 4:
    Best-tree structural analysis
    operator frequency
    tree depth
    node count

Experiment 5:
    Qualitative examples on test images
```

------

# 六、DIA 2027 留下的扩展内容

ViEW 不做或只做简化，DIA 2027 再扩展：

```
1. 各 TRAIN_SIZE 从 6 run 扩展到 20 或 30 run
2. 完整 8 组全因子消融：
   GP
   GP+P
   GP+E
   GP+G
   GP+P+E
   GP+P+G
   GP+E+G
   GP+P+E+G

3. TRAIN_SIZE=50 下的完整模块消融
4. GP vs CNN 正式比较
5. GA 触发次数、成功率、耗时收益分析
6. Pruning 对节点数、深度、推理时间的影响
```

------

## 最终结论

两个月内完成 ViEW 级别实验是可行的。推荐目标不是“完整 30 run 全实验”，而是：

```
数据规模趋势 + 代表条件稳定性 + 压缩版模块消融 + 木结构分析
```

按当前速度和 5–6 台 GPU 资源，预计总训练量约：

```
4500–5600 GPU小时
```

大约可以在：

```
6台机器：40–50 天
5台机器：50–60 天
```

完成主实验和补跑。重点是第 1 周必须完成时间标定，尤其要测出 `M_GA`，否则 GA 模块可能把整体进度拖爆。