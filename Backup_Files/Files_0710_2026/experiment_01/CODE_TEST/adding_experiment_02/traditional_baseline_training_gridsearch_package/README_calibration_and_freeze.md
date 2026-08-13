# B0–B3 training-only grid-search calibration protocol

## Purpose

This package replaces ad-hoc manual tuning with a predeclared calibration
procedure that uses **only training images 111–130**.  The calibration
executable contains no primary-test (211–410) image path.

## Selection rule

For each baseline:

1. maximize macro mean standard F1 over the 20 training images;
2. if tied within 1e-12, maximize the minimum per-image standard F1;
3. if still tied, choose the smaller candidate ID.

Standard F1 is `2TP / (2TP + FP + FN)` with no +1 smoothing.

## Fixed topologies

- **B0**: GaussianBlur → Threshold → BITWISE_NOT
- **B1**: GaussianBlur → Canny → Dilate → CC_FILTER
- **B2**: GaussianBlur → Sobel-X/Y → BITWISE_OR → Threshold → Dilate → CC_FILTER
- **B3**: MedianBlur → Threshold → BITWISE_NOT → Erode → Dilate → CC_FILTER

The BITWISE_NOT nodes in B0/B3 are fixed semantic choices for the dark-crack
raw-intensity baselines, based on the pre-primary training-sanity polarity
check. They are not test-selected topology changes.

## Search size

- B0: 90 candidates
- B1: 45 detector candidates + 36 postprocessing candidates
- B2: 270 detector candidates + 36 postprocessing candidates
- B3: 50 detector candidates + 108 postprocessing candidates
- **Total: 635 candidates**
- Candidate-image evaluations: **12,700**
- Final-winner replay: **80**
- Total metric evaluations: **12,780**

The two-stage B1/B2/B3 grids are predeclared. Stage 2 includes the Stage-1
default postprocessing settings, so the selected Stage-2 result cannot be
worse merely because the Stage-1 configuration disappeared from the grid.

## Inputs

- images: `./imgs_0710_2026_v1/input/train/positive/images`
- masks: `./imgs_0710_2026_v1/input/train/positive/masks`
- IDs: 111–130 only

Training GT masks are kept unchanged; the metric preserves the established
exact FG_PIXEL/BG_PIXEL semantics.

## Outputs

Calibration records:

`./imgs_0710_2026_v1/output/traditional_baseline/calibration/`

- `baseline_calibration_all_candidates.csv`
- `baseline_calibration_best_summary.csv`
- `baseline_calibration_best_per_image.csv`
- `baseline_calibration_resolved_protocol.txt`
- `best_resImgs/` (80 images by default)

Frozen trees are written automatically to:

`./imgs_0710_2026_v1/input/test/baseline_tree_frozen/`

## Run sequence

1. Compile `traditional_baseline_training_gridsearch_calibration.cpp` in the
   same Visual Studio/CUDA project used by the working GP evaluators.
2. Link the same deterministic CC_FILTER CUDA implementation and fused metric
   code.
3. Run the calibration executable once.
4. Confirm the final console says `candidates=635 expected=635`.
5. Inspect the four frozen tree files and a few `best_resImgs/` masks for
   implementation sanity only.
6. Do **not** manually retune after selection.
7. Compile/run `traditional_baseline_primary200_frozen_standardF1.cpp`.
8. Primary test command remains:
   `program.exe --mode primary200 --confirm-frozen`
9. After the primary-200 results exist, the baseline topology/parameters are
   locked permanently for this experiment.

## Re-running protection

By default the calibration program refuses to overwrite existing frozen tree
files. If a pre-primary implementation failure requires a rerun, delete/rename
the old frozen files first. Do not enable overwrite after primary-test results
have been inspected.
