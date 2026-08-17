# Level A — Structural-Control Baseline

This package evaluates manually specified pipelines using the **same fixed
numeric operator parameters as the current GP configuration**.

## 1. Copy the tree files

Copy all four files from `levelA_fixedparam_tree/` into:

`./imgs_0710_2026_v1/input/test/levelA_fixedparam_tree/`

The four definitions are:

- M0: GaussianBlur → Threshold → BITWISE_NOT
- M1: GaussianBlur → Canny → Dilate → CC_FILTER
- M2: GaussianBlur → Sobel-X/Y → OR → Threshold → Dilate → CC_FILTER
- M3: MedianBlur → Threshold → BITWISE_NOT → Erode → Dilate → CC_FILTER

Numeric values are GP-matched and must not be tuned.

## 2. Compile

Compile:

`levelA_manual_fixedparam_structural_control_standardF1.cpp`

using the same Visual Studio/CUDA configuration as the successful GP and
traditional-baseline evaluators.

Keep the same deterministic CC_FILTER `.cu`, fused metric implementation,
OpenCV CUDA libraries, and `SegmentationConfig.h`.

## 3. Training sanity

Run:

`Pro_kLab_Pan_Main.exe --mode train-sanity`

Expected:

`4 methods × 20 images = 80 evaluations`

Do not modify parameters based on the sanity F1 values. This mode is only for
tree-loading, CUDA, polarity and binary-output checks.

Outputs:

`./imgs_0710_2026_v1/output/traditional_baseline/levelA_fixedparam/train_sanity/`

## 4. Primary-200 structural control

Run:

`Pro_kLab_Pan_Main.exe --mode primary200 --confirm-levelA-fixed`

Expected:

`4 methods × 200 images = 800 evaluations`

Outputs:

`./imgs_0710_2026_v1/output/traditional_baseline/levelA_fixedparam/primary200/`

Main files:

- `levelA_fixedparam_primary200_run_summary_standardF1.csv`
- `levelA_fixedparam_primary200_per_image_standardF1.csv`

## 5. Paper interpretation

Level A answers a narrow structural-control question:

> GP topology search vs manual topology, with numeric operator parameters
> matched.

It does **not** replace the training-calibrated B0 practical baseline.
Report both:

- Level A = parameter-matched structural control.
- Calibrated B0–B3 = practical traditional baseline.

Because the broader primary-200 results were already known before this Level-A
control was run, label this analysis as post-hoc/supplementary and do not
redesign M0–M3 after inspecting its results.
