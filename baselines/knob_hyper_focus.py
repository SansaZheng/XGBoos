"""Focused analysis of the hyper-parameter knobs: fine dose-response on three seeds, and what each
setting does inside the model.

Usage: python -m baselines.knob_hyper_focus --dataset NAME [--variant V --flat FOLDER]

Knobs and grids (baseline value marked *):
  spw_alpha         scale_pos_weight = (negatives / positives) ** alpha, alpha in 0, 0.25, 0.5*, 0.75, 1, 1.25
  max_depth         2, 3, 4, 6*, 9
  min_child_weight  1*, 3, 10, 30, 100
  reg_lambda        0.1, 1*, 10, 30, 100
Everything else stays at the baseline (600 rounds max, early stopping 50 on validation aucpr, eta 0.05,
row / column sampling 0.8).  Small tasks: every value on three seeds.  Large tasks (test > 500k rows):
spw_alpha on three seeds, the other three knobs on seed 42.
Per fit: validation / test AP, F1, precision, recall, FN, FP at the validation-F1 threshold; trees kept by
early stopping; class separation of the margin (d', mean anomaly and normal margin, recall at 0.1 % FPR);
and, from TreeSHAP on validation rows, how the anomaly-minus-normal margin gap is distributed
(share of the largest feature, effective number of features, share of the first half of the trees).
"""
import argparse
import json
import time

import numpy as np
import xgboost as xgb

from . import knob_xgb as K
from .common import RESULTS, SEEDS, metrics, pick_threshold

OUT = RESULTS / 'knob_hyper'
GRID = dict(spw_alpha=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25), max_depth=(2, 3, 4, 6, 9),
            min_child_weight=(1, 3, 10, 30, 100), reg_lambda=(0.1, 1.0, 10.0, 30.0, 100.0))
BASE = dict(spw_alpha=0.5, max_depth=6, min_child_weight=1, reg_lambda=1.0)


def one(seed, data, names, setting):
    Xtr, ytr, _, Xva, yva, Xte, yte = data
    s = dict(BASE, **setting)
    ratio = (ytr == 0).sum() / max(ytr.sum(), 1)
    start = time.time()
    model = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=int(s['max_depth']), subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=s['min_child_weight'], reg_lambda=s['reg_lambda'],
                              scale_pos_weight=float(ratio ** s['spw_alpha']), tree_method='hist', eval_metric='aucpr',
                              early_stopping_rounds=50, random_state=seed, n_jobs=2)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    trees = int(model.best_iteration + 1)
    booster = model.get_booster()
    mva, mte = K.margins(model, Xva, trees), K.margins(model, Xte, trees)
    thr = pick_threshold(yva, mva)
    v, t = metrics(yva, mva, thr), metrics(yte, mte, thr)
    a, n = mte[yte == 1], mte[yte == 0]
    cut = np.quantile(n, 0.999)
    budget = np.sort(mte)[::-1][:max(int(yte.sum()), 1)]            # as many alerts as there are anomalies
    # distribution of the class gap over features and over boosting depth (validation rows)
    rows = np.arange(len(yva)) if len(yva) <= 120_000 else np.sort(np.random.default_rng(0).choice(len(yva), 120_000, replace=False))
    dv = xgb.DMatrix(Xva[rows]); yv = yva[rows]
    C = booster.predict(dv, pred_contribs=True, iteration_range=(0, trees))[:, :-1]
    half = booster.predict(dv, pred_contribs=True, iteration_range=(0, max(trees // 2, 1)))[:, :-1]
    gap = C[yv == 1].mean(0) - C[yv == 0].mean(0)
    total = float(gap.sum())
    p = np.clip(gap, 0, None); p = p / max(p.sum(), 1e-12)
    top = int(np.argmax(np.abs(gap)))
    return dict(seed=seed, setting=s, trees=trees, seconds=round(time.time() - start, 1), scale_pos_weight=float(ratio ** s['spw_alpha']),
                val_AP=v['AP'], test_AP=t['AP'], test_ROC_AUC=t['ROC_AUC'], test_F1=t['F1'], test_precision=t['precision'],
                test_recall=t['recall'], test_TP=t['TP'], test_FP=t['FP'], test_FN=t['FN'], val_F1=v['F1'],
                threshold=thr, anomaly_margin_mean=float(a.mean()), normal_margin_mean=float(n.mean()),
                anomaly_margin_sd=float(a.std()), normal_margin_sd=float(n.std()),
                d_prime=float((a.mean() - n.mean()) / np.sqrt((a.var() + n.var()) / 2 + 1e-12)),
                recall_at_fpr_0p1pct=float((a > cut).mean()), recall_at_k_positives=float((a >= budget[-1]).mean()),
                anomalies_below_normal_median=float((a < np.median(n)).mean()),
                val_gap=total, top_gap_feature=names[top], top_gap_share=float(gap[top] / total) if total else None,
                effective_features=float(1 / max((p ** 2).sum(), 1e-12)),
                first_half_tree_share=float((half[yv == 1].mean(0) - half[yv == 0].mean(0)).sum() / total) if total else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--variant', default=None)
    p.add_argument('--flat', default=None)
    args = p.parse_args()
    if args.dataset in K.MARKET and not args.flat:
        variant = args.variant or K.MARKET[args.dataset]
        data, names = K.market_data(args.dataset, variant)
    else:
        variant = args.variant
        data, names = K.flat_data(args.flat, variant)
    large = len(data[5]) > 500_000
    folder = OUT; folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{args.dataset}__{variant}.json'
    records, cache = [], {}
    for knob, values in GRID.items():
        seeds = SEEDS if (knob == 'spw_alpha' or not large) else SEEDS[:1]
        for value in values:
            for seed in seeds:
                setting = {knob: value}
                key = (seed, tuple(sorted(dict(BASE, **setting).items())))
                if key not in cache:
                    cache[key] = one(seed, data, names, setting)
                r = dict(cache[key], knob=knob, value=value, is_baseline=value == BASE[knob])
                records.append(r)
                print(f'{args.dataset} {knob}={value} seed={seed} trees={r["trees"]} val AP {r["val_AP"]:.4f} test AP {r["test_AP"]:.4f} '
                      f'F1 {r["test_F1"]:.3f} FN/FP {r["test_FN"]}/{r["test_FP"]} d\' {r["d_prime"]:.2f} top share '
                      f'{r["top_gap_share"] if r["top_gap_share"] is not None else float("nan"):.2f} ({r["top_gap_feature"][:24]}) {r["seconds"]}s', flush=True)
        path.write_text(json.dumps(dict(dataset=args.dataset, variant=variant, large=large, train_rows=int(len(data[1])),
                                        train_positives=int(data[1].sum()), validation_positives=int(data[4].sum()),
                                        test_positives=int(data[6].sum()), grid={k: list(v) for k, v in GRID.items()},
                                        baseline=BASE, records=records), indent=1) + '\n')


if __name__ == '__main__':
    main()
