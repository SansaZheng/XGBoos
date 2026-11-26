# Combined XGBoost against ten methods from 2025–2026 · English v7

Version: EN v7 (2026-11-27). Chinese version: [KNOB_XGB_REPORT_CN_v7.md](KNOB_XGB_REPORT_CN_v7.md). Earlier: [EN v6](KNOB_XGB_REPORT_EN_v6.md) (the combined model and the comparison inside Kronos_SupervisedAD). This version compares it with the methods recorded in [Financial_unsuperviesdAD](https://github.com/SansaZheng/Financial_unsuperviesdAD), on that repository's own task partitions.

## Conclusion

**On ranking quality (test AP) the combined XGBoost is clearly ahead of all ten recent methods, and of every other method recorded there (classical ones included), on 9 of 11 tasks; it is level on one and behind on one. It is not a state-of-the-art result in the strict sense, because the comparison is not like for like: our model is trained with labels, and all of theirs are label-free.**

- **Clearly ahead (9 tasks)**: ULB, BankSim, Sapienza 5S / 15S / 25S, SD-FMM stock / crypto / synthetic, and the BankSim account task. Margins over the best recorded method run from +0.075 to +0.495 AP.
- **Level (1 task)**: S-FFSD accounts, 0.913 ± 0.019 against Audit-HCL 0.904 ± 0.022. The gap is inside the seed spread.
- **Behind (1 task)**: S-FFSD transactions, 0.290 against HBOS 0.561. Our model is worse than random here (test ROC-AUC 0.30, anomaly share of the test rows 0.389): what the labels teach on the earlier period points the wrong way on the later period. Using labels is a liability on this task.
- **How many labels the lead needs**: with only 10 labelled anomalies the plain no-weight model already beats the best recorded method on 7 of 11 tasks; with 50, on 8.
- **What this does not show**: that the method beats published supervised methods on these datasets. No supervised competitor was run on these partitions, and pretrained tabular foundation models still cannot be run in this environment.

## What was compared

**Their side** (numbers read from `results/roleweight_20261117` of that repository, never re-run here): 22 method variants on nine transaction / point tasks, 9 on two account tasks. The ten methods from 2025–2026:

| Method | Source | Tasks it has numbers for |
|---|---|---|
| CAE-IF | ICMLA 2025 | all 11 |
| RGLD-ID | arXiv, June 2026 | 9 transaction / point |
| RTTAD-TTA | arXiv, May 2026 | 9 transaction / point |
| KronosAD-M | Kronos foundation model, 2025 | 9 transaction / point |
| Kronos-S | Kronos foundation model, 2025 | 9 transaction / point |
| HIMVH-Zero | WWW 2026 | BankSim, S-FFSD transactions |
| SD-FMM-Zero | IPM 2026 | Sapienza ×3, SD-FMM ×3 |
| HUGE | AAAI 2025 | 2 account tasks |
| EB-GAD | arXiv 2026 | 2 account tasks |
| Audit-HCL | Scientific Reports 2026 | 2 account tasks |

Several of these are that repository's label-free adaptations of the published methods, not the authors' releases. Where a method has several recorded variants, the best one per task is used, which favours their side.

**Our side**: the same combined XGBoost as v6, unchanged (no class weight; start score, regularisation, novelty feature and seed ensemble chosen on validation AP), plus plain XGBoost. Seeds 42, 13, 2026.

**Same data**: the partitions are rebuilt with that repository's own scripts (`scripts/prepare_repo_data.py`, `finad_zero.graph_data`). Test rows and positives match its tables exactly (for example ULB 92,575 rows, BankSim accounts 4,082 with 648 positive, S-FFSD accounts 1,070 with 936 positive). Its calibration part is our validation part.

**Not the same information**, and this decides how to read every number below:

| | Their methods | Ours |
|---|---|---|
| Training labels | none | all of the training part |
| Training rows | at most 8,192 | all |
| Validation (calibration) labels | none | used for early stopping, model choice and the F1 threshold |
| Threshold | POT or 99 % quantile, label-free | best validation F1 |

## Main result: test AP

Mean of three seeds; ± is the sample standard deviation over seeds.

| Task | Test rows / anomalies | Best recorded (any method) | Best of the ten recent | Plain XGBoost | Combined XGBoost | Combined − best recorded |
|---|---:|---|---|---:|---:|---:|
| ULB | 92,575 / 115 | 0.301 PCA-SPE | 0.121 KronosAD-M, Kronos-S | 0.778 | **0.796** ± 0.001 | +0.495 |
| BankSim | 118,638 / 1,280 | 0.409 HBOS | 0.207 CAE-IF | 0.874 | **0.876** ± 0.000 | +0.467 |
| S-FFSD | 6,159 / 2,394 | **0.561** HBOS | 0.423 RGLD-ID | 0.302 | 0.290 ± 0.001 | −0.271 |
| Sapienza 5S | 207,267 / 77 | 0.762 kNN | 0.670 RGLD-ID | 0.928 | **0.928** ± 0.001 | +0.166 |
| Sapienza 15S | 145,550 / 77 | 0.826 kNN | 0.670 RGLD-ID | 0.964 | **0.964** ± 0.001 | +0.139 |
| Sapienza 25S | 119,469 / 77 | 0.840 kNN | 0.733 RGLD-ID | 0.963 | **0.967** ± 0.001 | +0.126 |
| SD-FMM stock | 960 / 99 | 0.639 Isolation Forest | 0.628 RGLD-ID | 0.391 | **0.713** ± 0.012 | +0.075 |
| SD-FMM crypto | 10,880 / 68 | 0.337 persistence control | 0.332 Kronos-S | 0.657 | **0.662** ± 0.026 | +0.326 |
| SD-FMM synthetic | 32,000 / 1,589 | 0.771 RGLD-ID | 0.771 RGLD-ID | 0.906 | **0.908** ± 0.000 | +0.136 |
| BankSim accounts | 4,082 / 648 | 0.613 EB-GAD | 0.613 EB-GAD | 0.795 | **0.805** ± 0.001 | +0.193 |
| S-FFSD accounts | 1,070 / 936 | 0.904 ± 0.022 Audit-HCL | 0.904 Audit-HCL | 0.909 | 0.913 ± 0.019 | +0.009 |

Notes on single rows:

- **SD-FMM stock** has six test sequences. The lead is real on these rows but rests on very little data. It also comes entirely from the combination: plain XGBoost (0.391) is well below Isolation Forest (0.639).
- **S-FFSD accounts**: 87 % of the evaluated accounts are positive, so every method scores high and the task separates methods poorly.
- **S-FFSD transactions**: among known-label rows the anomaly share is 12 % in training and 39 % in test. Validation AP of our model is 0.40, test AP 0.29. Most label-free methods are also near the anomaly share of 0.389; HBOS and one-class SVM are the only ones above 0.5.

Full per-method table of the ten recent methods: [`results/finunsup_compare/recent_methods_AP.csv`](../results/finunsup_compare/recent_methods_AP.csv).

## How much of the lead is just labels

The no-weight XGBoost (start 0.5, no selection) with fewer labels. "8,192 rows" is the row budget their methods get, here with labels; the number in brackets is how many labelled anomalies that sample contains on average.

| Task | Best recorded | 10 labelled anomalies | 50 labelled anomalies | 8,192 labelled rows | All labels |
|---|---:|---:|---:|---:|---:|
| ULB | 0.301 | 0.689 | 0.768 | 0.689 (16) | 0.775 |
| BankSim | 0.409 | 0.011 | 0.731 | 0.790 (102) | 0.876 |
| S-FFSD | 0.561 | 0.386 | 0.442 | 0.391 (990) | 0.321 |
| Sapienza 5S | 0.762 | 0.863 | 0.906 | 0.763 (2) | 0.928 |
| Sapienza 15S | 0.826 | 0.908 | 0.937 | 0.893 (6) | 0.964 |
| Sapienza 25S | 0.840 | 0.939 | 0.958 | 0.934 (4) | 0.967 |
| SD-FMM stock | 0.639 | 0.600 | 0.594 | 0.493 (all 333) | 0.493 |
| SD-FMM crypto | 0.337 | 0.547 | 0.644 | 0.657 (57) | 0.626 |
| SD-FMM synthetic | 0.771 | 0.833 | 0.860 | 0.888 (416) | 0.907 |
| BankSim accounts | 0.613 | 0.676 | 0.769 | 0.792 (all 1,129) | 0.792 |
| S-FFSD accounts | 0.904 | 0.887 | 0.894 | 0.914 (all 229) | 0.914 |

- Ten labelled anomalies are enough to pass the best label-free result on 7 of 11 tasks; fifty on 8 (BankSim joins).
- On BankSim with 10 anomalies the model stops at zero trees and returns a constant score (AP equals the anomaly share, 0.011).
- On SD-FMM stock fewer anomaly labels give a higher AP than all of them (0.60 against 0.49). With six test sequences this is not a stable finding.
- On S-FFSD fewer labels are also better than all labels, consistent with the labels pointing the wrong way on the test period.
- These runs still use all validation labels for early stopping, so they understate the true label cost.

## Alerts: F1

| Task | Best recorded F1 (POT or 99 % quantile) | Combined XGBoost, validation-F1 threshold | No-weight XGBoost at their 99 % quantile rule |
|---|---:|---:|---:|
| ULB | 0.205 | 0.808 | 0.260 |
| BankSim | 0.489 | 0.811 | 0.808 |
| S-FFSD | 0.031 | 0.002 | 0.000 |
| Sapienza 5S | 0.086 | 0.857 | 0.073 |
| Sapienza 15S | 0.113 | 0.896 | 0.100 |
| Sapienza 25S | 0.195 | 0.926 | 0.115 |
| SD-FMM stock | 0.262 | 0.665 | 0.240 |
| SD-FMM crypto | 0.347 | 0.627 | 0.535 |
| SD-FMM synthetic | 0.422 | 0.857 | 0.358 |
| BankSim accounts | 0.496 | 0.745 | 0.145 |
| S-FFSD accounts | 0.478 | 0.883 | 0.020 |

Most of the F1 gap comes from the threshold, not the ranking. The 99 % quantile rule alerts on 1 % of rows whatever the anomaly share is (0.04 % on Sapienza, 87 % on S-FFSD accounts), so with that rule our F1 falls to about their level on most tasks. A threshold set with validation labels is a second use of labels that their methods do not have.

## Is it state of the art?

| Claim | Supported? |
|---|---|
| Better ranking than the ten 2025–2026 methods and the classical methods recorded in that repository | Yes on 9 of 11 tasks, with labels against no labels; level on S-FFSD accounts; worse on S-FFSD transactions |
| State of the art for label-free financial anomaly detection | No. Our method is not label-free |
| State of the art for supervised detection on these datasets | Not shown. No supervised competitor was run on these partitions |

What the comparison does establish: on these financial tasks a small number of labels is worth more than any of the recent label-free methods as adapted there, except where the label relationship changes between periods (S-FFSD), where a label-free detector is the safer choice.

## Limits

- Their numbers are read from the repository's tables; seeds differ (theirs 42 / 43 / 44).
- The combined model was designed in v5–v6 after looking at test results of other tasks. On the partitions used here it was applied unchanged, so these numbers are less exposed to that than v6's.
- Model selection inside the combined model does not always help: on S-FFSD it chose a configuration with lower test AP than plain XGBoost.
- SD-FMM stock (6 test sequences) and S-FFSD accounts (87 % positive) are weak tests.
- The account tasks use 4,096 hash-selected paying accounts per phase, as defined there, not all accounts.

## Reproduce

```bash
# in a checkout of Financial_unsuperviesdAD
python scripts/prepare_repo_data.py --output CACHE --datasets ulb banksim s_ffsd sapienza_5S sapienza_15S sapienza_25S sd_stock sd_crypto sd_synth
python -m finad_zero.graph_data --output GRAPHS
# in this repository
python -m baselines.finunsup_data --tabular CACHE --graphs GRAPHS --output FLAT
python -m baselines.finunsup_compare --flat FLAT --dataset fu_ulb          # one call per task
python -m baselines.auto_xgb --dataset fu_ulb --variant raw --flat FLAT/fu_ulb --out finunsup_auto
python -m baselines.finunsup_aggregate --repo PATH_TO_Financial_unsuperviesdAD
```

Results: [`results/finunsup_compare/`](../results/finunsup_compare/) (plain and reduced-label runs, `table.csv`), [`results/finunsup_auto/`](../results/finunsup_auto/) (combined model, every candidate).
