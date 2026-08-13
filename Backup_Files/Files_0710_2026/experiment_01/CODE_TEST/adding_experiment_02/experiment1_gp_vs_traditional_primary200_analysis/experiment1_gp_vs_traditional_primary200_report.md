# Experiment 1 — GP vs calibrated traditional baseline on primary 200-image test

## Integrity

The traditional-baseline primary evaluation is complete: **800 rows = 4 pipelines × 200 images**.
Each baseline contains image IDs **211–410 exactly once**. Standard F1 recomputes from TP/FP/FN
with maximum absolute discrepancy below `1e-9`.

## Traditional baseline results

| Method | Macro mean F1 | Median F1 | Macro precision | Macro recall | Micro F1 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1987 | 0.1602 | 0.1243 | 0.8232 | 0.1839 |
| B1 | 0.0304 | 0.0105 | 0.0201 | 0.1383 | 0.0254 |
| B2 | 0.0303 | 0.0121 | 0.0180 | 0.1724 | 0.0291 |
| B3 | 0.0482 | 0.0332 | 0.0252 | 0.9882 | 0.0486 |

B0 is clearly the strongest of the four calibrated handcrafted pipelines. B1–B3 are weak supplementary references.

## GP versus primary traditional baseline B0

B0 primary-200 macro mean F1 = **0.1987**.

| GP condition | GP mean ± between-run SD | Δ(GP−B0) | hierarchical bootstrap 95% CI | GP runs above B0 |
|---|---:|---:|---:|---:|
| TS05 | 0.1952 ± 0.0596 | -0.0035 | [-0.0554, 0.0400] | 5/6 |
| TS10 | 0.1827 ± 0.0229 | -0.0160 | [-0.0375, 0.0063] | 2/6 |
| TS20 | 0.1783 ± 0.0410 | -0.0204 | [-0.0544, 0.0165] | 2/6 |

The hierarchical bootstrap resamples both the six GP runs and the 200 shared test images.
All three confidence intervals include zero. Therefore the present data do **not** support a
claim that GP reliably outperforms B0, nor a strong claim that B0 reliably outperforms GP after
accounting for GP run-to-run variability.

The important numerical result is that TS05 is essentially tied with B0 on average
(0.1952 versus 0.1987), while TS10 and TS20 are lower numerically. TS05 is highly variable:
**5/6 individual TS05 runs exceed the B0 macro mean**, but seed46 is much worse and pulls down
the six-run average.

## Same-20-training-set comparison

B0 calibration used the 20 images 111–130, which is the same formal image set used by GP TS20.
B0 moves from training mean F1 **0.3543** to test F1
**0.1987**, a gap of **0.1556**.
GP TS20 previously had a substantially larger average train–test gap. This is useful descriptive
evidence that the GP configuration fits its training images much more strongly without obtaining
a corresponding primary-test advantage. Because GP optimized the robust training objective
rather than simple mean F1 directly, this should be framed as a generalization-gap observation,
not as a pure controlled test of overfitting mechanism.

## Paper-level conclusion

A defensible statement is:

> The calibrated handcrafted intensity baseline (B0) achieved a primary-test mean F1 comparable
> to the GP configurations. GP did not show a reliable average advantage over this baseline once
> run-to-run variability was considered. In particular, the small-data TS05 condition was
> competitive with B0 but exhibited substantial stochastic variability.

Do not claim that GP beats the conventional baseline. The stronger research contribution is now
automatic generation of executable/interpretable pipelines under very small training sets,
together with an explicit characterization of their generalization limitations.
