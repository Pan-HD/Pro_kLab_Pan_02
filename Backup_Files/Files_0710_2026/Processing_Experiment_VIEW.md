## 第 1 周：标定与准备

目标：不要直接开跑全部实验，先确认时间模型和 GA 倍率。

运行：

```
P-01. TRAIN_SIZE=5/10/20/50/100，各 500(selected) 或 1000 代，1 run
P-02. TRAIN_SIZE=20，GP only，1000代，1 run
P-03. TRAIN_SIZE=20，GP + GA，1000代，1 run
```

- PC id

  - pc-01: PAN_MAIN
  - pc_02: YANG_MAIN
  - pc_03: YANG_NOTEBOOK
  - pc_04: FENG_MAIN
  
- P01 (GP Only)

  - 01-TRAIN_SIZE=5

  ```
  experimentID: P01_01
  PC: YANG_MAIN
  START_TIME: 17:30, 07-10, 2026
  ```
  
  - 02-TRAIN_SIZE=10
  
  ```
  PC: YANG_NOTEBOOK
  START_TIME: XX:XX, XX-XX, 2026
  ```
  
  - 03-TRAIN_SIZE=20
  
  ```
  PC: PAN_MAIN
  START_TIME: XX:XX, XX-XX, 2026
  ```
  
  - 04-TRAIN_SIZE=50
  
  ```
  PC: YANG_MAIN
  START_TIME: XX:XX, XX-XX, 2026
  ```
  
  - 05-TRAIN_SIZE=100
  
  ```
  PC: FENG_MAIN
  START_TIME: XX:XX, XX-XX, 2026
  ```
  
- P02

  - 01-TRAIN_SIZE=20

  ```
  CALING
  experimentID: P01_02
  PC: pc_04 FENG_MAIN
  START_TIME: 10:30, 07-14, 2026
  TS20_GP_1000.cpp
  ```
  
  ```
  FINISHED
  experimentID: P01_02
  PC: pc_04 FENG_MAIN
  START_TIME: 10:50, 07-13, 2026
  pro_train_week01_TS20_GA_1000_0710_2026_lowmemGA_cooldownfix.cpp
  ```
  
  ```
  WAITING
  experimentID: P01_02
  PC: pc_04 FENG_MAIN
  START_TIME: xx:xx, 07-xx, 2026
  TS20_PRUNING_GA_1000_lowmemGA_cooldownfix
  ```
  
  ```
  FINISHED
  GP + Pruning
  experimentID: P01_02
  PC: pc_02 YANG_MAIN
  START_TIME: 22:45, 07-12, 2026
  pro_train_week01_TS20_PRUNING_1000_0710_2026.cpp
  ```
  

## 第 2 周 (EXPERIMENT-01)

优先启动长任务：

```text
TRAIN_SIZE=20
```

完成目标：

```text
TRAIN_SIZE=5/10/20 各 6 run
```

**运行记录：**

```
TS05
  RUN01 finished
      pc00(PAN_NOTEBOOK)
  RUN02 finished (recovered)
      pc02(YANG_MAIN)
      END_TIME_PREDICT: xx:xx, xx-xx
  RUN03 finished
      pc00(PAN_NOTEBOOK)
      END_TIME_PREDICT: xx:xx, 07-xx
  RUN04 finished
      pc02(YANG_MAIN)
      END_TIME_PREDICT: xx:xx, xx-xx
  RUN05 ing
      pc00(PAN_NOTEBOOK)
  RUN06 ing
      pc02(YANG_MAIN)
      END_TIME_PREDICT: xx:xx, xx-xx
TS10
  RUN01 finished
      pc03(YANG_NOTEBOOK)
      END_TIME_PREDICT: 21:30, 07-17
  RUN02 finished
      pc03(YANG_NOTEBOOK)
      END_TIME_PREDICT: xx:xx, 07-xx
  RUN03 ing
      pc03(YANG_NOTEBOOK)
      END_TIME_PREDICT: 21:30, 07-17
  RUN04
  
  RUN05
  RUN06
TS20
  RUN01 finished
    pc01(PAN_MAIN)
    END_TIME_PREDICT: 00:00, 07-23
  RUN02
  RUN03
  RUN04
  RUN05
  RUN06
```

