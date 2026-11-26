# A combined XGBoost, and how close it gets to the best result on every task · English v6

Version: EN v6 (2026-11-26). Chinese version: [KNOB_XGB_REPORT_CN_v6.md](KNOB_XGB_REPORT_CN_v6.md). [EN v5](KNOB_XGB_REPORT_EN_v5.md) changed one component at a time. This version combines every piece that helped somewhere and lets validation choose per task, with the aim of the best result on all nine Kronos_SupervisedAD tasks.

## Conclusion

**The combined XGBoost is not the best method on every task.** It is ahead of every reference on six of nine, level on one where nothing works, and behind the gradient-boosting control of Kronos_SupervisedAD on two.

- **Ahead (6):** Sapienza, SD-FMM stock, SD-FMM crypto, TSTR, Elliptic, IBM AML. On TSTR and Elliptic the lead over the gradient-boosting control is inside the bootstrap interval, so those two are better described as level.
- **Level (1):** TabFormer, 0.0070 against 0.0074 for Kronos-mini; neither is usable.
- **Behind (2):** PaySim (0.970 against 0.975) and ULB (0.762 against 0.776). Both gaps are small, and both bootstrap intervals exclude zero.
- **Against plain XGBoost** the combined model is better on all nine tasks; mean test AP rises from 0.623 to 0.660.
- **Two attempts to close the remaining gaps made things worse.** A wider search and a different selection rule both lowered the mean. The limit is the validation split, which holds 33–100 anomalies on most tasks.

"Best" here means the strongest result in this comparison on identical splits: recorded Kronos-mini, the HistGradientBoosting control of Kronos_SupervisedAD, and XGBoost. Published results on these datasets use other splits and are not comparable.

## The combined model

No class weight. Then four choices, made one at a time by mean validation AP over three seeds:

1. Starting score: fitted prior, or 0.5.
2. Regularisation: default (depth 6), depth 2, depth 4, `min_child_weight` 30, or `reg_lambda` 30.
3. Novelty feature (the label-free real-versus-swapped XGBoost of v5): off or on.
4. Seed ensemble: one booster, or five averaged.

Test labels never enter a choice.

## Results

Test AP; XGBoost columns are means over three seeds. "Best reference" is the higher of Kronos-mini repaired and the gradient-boosting control.

| Task | Kronos-mini repaired | HistGB control | Plain XGBoost | **Combined XGBoost** | Against best reference | Chosen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sapienza 15S | 0.912 | 0.954 | 0.965 | **0.966 ± 0.002** | +0.012 | prior, default, novelty |
| SD-FMM stock | 0.630 | 0.443 | 0.602 | **0.664 ± 0.003** | +0.034 | prior, depth 2, ensemble |
| SD-FMM crypto | 0.480 | 0.648 | 0.657 | **0.680 ± 0.012** | +0.032 | 0.5 start, depth 4, novelty |
| TSTR | 0.486 | 0.826 | 0.830 | **0.841 ± 0.002** | +0.015 | prior, default |
| Elliptic | 0.435 | 0.542 | 0.537 | **0.545 ± 0.001** | +0.004 | 0.5 start, depth 4, ensemble |
| IBM AML | 0.266 | 0.242 | 0.330 | **0.501 ± 0.028** | +0.235 | 0.5 start, default |
| TabFormer | **0.0074** | not run | 0.0052 | 0.0070 ± 0.0001 | −0.0003 | prior, `reg_lambda` 30 |
| PaySim | 0.452 | **0.975** | 0.924 | 0.970 ± 0.002 | −0.005 | 0.5 start, default |
| ULB | 0.754 | **0.776** | 0.755 | 0.762 ± 0.001 | −0.014 | prior, default, ensemble |

Detection at the validation-F1 threshold, plain XGBoost → combined:

| Task | F1 | Misses | False alerts |
| --- | ---: | ---: | ---: |
| Sapienza | 0.910 → 0.911 | 9.7 → 9.0 | 3.7 → 4.3 |
| SD-FMM stock | 0.617 → 0.685 | 39.7 → 37.0 | 34 → 20 |
| SD-FMM crypto | 0.609 → 0.641 | 31.7 → 27.0 | 15 → 19 |
| TSTR | 0.756 → 0.767 | 21.7 → 22.7 | 25.3 → 20.7 |
| ULB | 0.764 → 0.812 | 15.7 → 14.7 | 6.7 → 2.7 |
| Elliptic | 0.562 → 0.581 | 223 → 220 | 64 → 52 |
| PaySim | 0.754 → 0.842 | 1,543 → 1,076 | 54 → 13 |
| IBM AML | 0.421 → 0.442 | 1,130 → 1,107 | 69 → 50 |
| TabFormer | 0.010 → 0.010 | 4,430 → 4,429 | 492 → 334 |

F1 does not fall on any task. Most of the gain is fewer false alerts.

### How firm are the comparisons with the gradient-boosting control

Paired bootstrap over test rows (500 resamples, seed 42 models), combined XGBoost minus the control:

| Task | Test anomalies | Difference in AP | 95 % interval |
| --- | ---: | ---: | --- |
| IBM AML | 1,567 | +0.302 | +0.281 to +0.321 |
| TSTR | 94 | +0.014 | −0.017 to +0.048 |
| Elliptic | 408 | +0.003 | −0.002 to +0.008 |
| PaySim | 4,010 | −0.008 | −0.010 to −0.005 |
| ULB | 52 | −0.015 | −0.033 to −0.000 |

Rows are resampled independently, which ignores dependence within sequences, so the intervals are if anything too narrow. The control was not run on TabFormer, and the three market tasks were not bootstrapped; there the combined model leads the control by 0.012 to 0.22.

## Two attempts to close the gaps, both worse

After the first run I tried two changes aimed at PaySim and ULB, applied to all nine tasks.

| Scheme | What changed | Mean test AP | PaySim | ULB |
| --- | --- | ---: | ---: | ---: |
| 1. First design (reported above) | — | **0.660** | 0.970 | 0.762 |
| 2. Wider search | adds leaf-wise trees with 15 leaves (as the control uses) and a choice of feature variant | 0.654 | 0.921 | 0.760 |
| 3. Pooled selection | scheme 2's candidates, scored by cross-validation on training rows pooled with validation | 0.656 | 0.950 | 0.753 |

- **Wider search hurt.** On PaySim validation preferred the leaf-wise trees by 0.0025 AP; on test they are 0.03–0.05 worse and unstable across seeds.
- **Pooling training cross-validation into the choice did not help either.** It improved SD-FMM crypto (0.704) and Sapienza (0.967) and lowered PaySim, ULB, SD-FMM stock and TSTR.
- **Scheme 1 is reported as the result because it was designed and run first**, not because it scored best afterwards. That it also has the highest mean is consistent with the simpler search overfitting validation less.

## Why two tasks stay behind

- **PaySim**: validation holds 33 anomalies and its AP is around 0.75 while test AP is around 0.97, so validation is a poor guide there. The control does not use validation at all: it runs a fixed 100 rounds. Its lead is 0.005.
- **ULB**: the test split has 52 anomalies and the gap is 0.014, with an interval that only just excludes zero. Every XGBoost variant tried since v2 lands between 0.75 and 0.77.

For these two tasks the gradient-boosting control is the model to use.

## Limitations

- **The numbers are optimistic.** The search space was designed after seeing test results in v3–v5, on these same nine tasks. A clean estimate needs data that played no part in the design.
- **Kronos numbers are records**, mostly single seed, and were not rerun; SD-FMM stock's lead over Kronos (0.664 against 0.630) rests on a single recorded Kronos run and six test sequences.
- **TabFormer is unusable for every method** with these single-transaction features; an earlier run with per-card history features reached 0.078, so the remedy there is features, not the model.
- **IBM AML's gain comes entirely from the starting score**, whose mechanism v5 could not establish.
- XGBoost 3.4.1; three seeds; one job at a time on two CPU cores.

## Reproduce

Data preparation as in v2; flags as in v4.

```bash
python -m baselines.auto_xgb    --dataset NAME [--variant V --flat FOLDER]   # schemes 1 and 2 (scheme 2 when --variant is omitted for --flat tasks)
python -m baselines.auto_xgb_cv --dataset NAME [--variant V --flat FOLDER]   # scheme 3
python -m baselines.auto_vs_hgb --dataset NAME --flat FOLDER                 # paired bootstrap against the control
python -m baselines.auto_aggregate
```

Scheme 1 results are in `results/auto_xgb_first/`, scheme 2 and the bootstrap in `results/auto_xgb/`, scheme 3 in `results/auto_xgb_cv/`; the combined table is `results/auto_xgb/final_table.csv`. Scheme 1 was produced with the candidate list before leaf-wise trees were added; to rerun it exactly, remove `leaves15` from `REG` in `baselines/auto_xgb.py` and pass `--variant`.
