# 11. 实验2的运行方式建议

整理目录时建议使用这种结构：

```
output/
  exp02/
    S1/
      run_01/
        printed_tree_sys.txt
      run_02/
      ...
      run_30/
    S2/
    S3/
    S4/
    S5/

  exp02_test_results/
    results.csv
    S1/
      run_01/resImgs/
      run_02/resImgs/
    S2/
    ...
```

单次测试命令例：

```
experiment_02_test.exe S1 5 1 ^
  ./imgs_0630_2026_v1/output/exp02/S1/run_01/printed_tree_sys.txt ^
  ./imgs_0630_2026_v1/input/test/images ^
  ./imgs_0630_2026_v1/input/test/masks ^
  ./imgs_0630_2026_v1/output/exp02_test_results/results.csv ^
  ./imgs_0630_2026_v1/output/exp02_test_results/S1/run_01/resImgs
```

之后用脚本循环：

```
S1: train_size=5,   run_id=1..30
S2: train_size=10,  run_id=1..30
S3: train_size=20,  run_id=1..30
S4: train_size=50,  run_id=1..30
S5: train_size=100, run_id=1..30
```

------

# 12. 最终汇总表应由 CSV 再计算

`experiment_02_test.cpp` 只需要输出每次 run 的结果。实验2的最终表格建议由 Python / Excel / C++ 汇总：

```
condition,train_size,best_f1,mean_f1,std_f1,best_iou,mean_iou,std_iou
S1,5,...
S2,10,...
S3,20,...
S4,50,...
S5,100,...
```

这样更符合实验2“各条件 Best / Mean / Std”的要求。