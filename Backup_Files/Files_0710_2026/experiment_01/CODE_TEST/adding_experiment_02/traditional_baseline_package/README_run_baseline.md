# Traditional baseline package

## Files to copy

Copy the four files in `baseline_tree/` to:

`./imgs_0710_2026_v1/input/test/baseline_tree/`

Keep `traditional_baseline_200images_standardF1.cpp` in the same Visual Studio /
CUDA project configuration used by the successful GP 200-image evaluator.

Continue linking the same deterministic `CC_FILTER` CUDA `.cu` implementation
and fused-metrics CUDA code as before.

## Mode 1 — training sanity (RUN THIS FIRST)

No arguments are also accepted and default to training sanity.

Command:

`program.exe --mode train-sanity`

Input:
- images: `./imgs_0710_2026_v1/input/train/positive/images`
- masks:  `./imgs_0710_2026_v1/input/train/positive/masks`
- IDs: 111–130

Expected evaluations:

`4 baselines × 20 images = 80`

Expected final console:

`[BASELINE-DONE] mode=train_sanity completed=4 failed=0 total=4`

`[BASELINE-DONE] evaluations=80 expected=80`

Check:
- no CUDA exceptions,
- no missing tree/image/mask,
- ideally `invalidImages=0`,
- inspect several saved masks,
- confirm foreground polarity is logically correct.

Do not use the training sanity results for repeated manual optimization.

## Freeze point

After sanity passes, do not change the four tree definitions/parameters.
Keep `baseline_protocol_pre_primary.txt` with the experiment records.

If a genuine polarity/implementation error is found during training sanity,
fix it before primary-200 and record exactly what changed. Recompute file hashes
before calling the baseline frozen.

## Mode 2 — primary 200 test

The executable deliberately blocks the primary test unless an explicit freeze
confirmation is supplied.

Command:

`program.exe --mode primary200 --confirm-frozen`

Input:
- images: `./imgs_0710_2026_v1/input/test/images`
- masks:  `./imgs_0710_2026_v1/input/test/masks`
- IDs: 211–410

Expected evaluations:

`4 baselines × 200 images = 800`

Expected final console:

`[BASELINE-DONE] mode=primary200 completed=4 failed=0 total=4`

`[BASELINE-DONE] evaluations=800 expected=800`

After this run, do not modify the baseline definitions based on test results.

## CSV compatibility

Primary output:
- `traditional_baseline_primary200_run_summary_standardF1.csv`
- `traditional_baseline_primary200_per_image_f1_long_standardF1.csv`

The column names/order are intentionally identical to the existing GP
primary-200 CSVs.

For baseline rows:
- `train_size = 0`
- `seed = 0`

Use `run_id` to distinguish B0–B3.

## Saved predictions

By default:

`#define SAVE_BASELINE_RESULT_IMAGES 1`

This saves 80 sanity masks and 800 primary-test masks. Change it to 0 before
any run if disk I/O is undesirable; this does not change metric computation.
