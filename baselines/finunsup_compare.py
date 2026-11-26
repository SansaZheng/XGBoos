"""Plain and reduced-label XGBoost on the task partitions of SansaZheng/Financial_unsuperviesdAD.

Usage: python -m baselines.finunsup_compare --flat FOLDER --dataset fu_ulb

Variants (three seeds each, validation = that repository's calibration part):
  plain        XGBoost with the usual class weight sqrt(neg/pos), fitted starting score
  noweight     no class weight, starting score 0.5   (the removal found earlier in this project)
  rows8192     noweight, but only 8192 sampled training rows keep their labels and are used
               (the row budget that repository gives its label-free detectors)
  pos10/pos50  noweight on all normal training rows plus only 10 / 50 sampled labelled anomalies
The combined model is run separately:  python -m baselines.auto_xgb --flat ... --out finunsup_auto
Besides AP, F1 is reported at two thresholds: best validation F1 (uses validation labels) and the 99 % quantile of
validation scores (the label-free rule that repository reports).
"""
import argparse
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from . import knob_xgb as K
from .common import RESULTS, SEEDS, pick_threshold
from .dual_xgb import booster, margin

OUT = RESULTS / 'finunsup_compare'


def fit(seed, Xtr, ytr, Xva, yva, weight, half):
    m = booster(seed, weight)
    if half:
        m.set_params(base_score=0.5)
    return m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


def row(variant, seed, m, Xva, yva, Xte, yte, n_train, n_pos):
    (sv, trees), (st, _) = margin(m, Xva), margin(m, Xte)
    thr, q99 = pick_threshold(yva, sv), np.quantile(sv, .99)
    return dict(variant=variant, seed=seed, train_rows=int(n_train), train_pos=int(n_pos), trees=trees,
                val_AP=float(average_precision_score(yva, sv)), test_AP=float(average_precision_score(yte, st)),
                test_ROC_AUC=float(roc_auc_score(yte, st)), test_F1=float(f1_score(yte, st >= thr)),
                test_F1_q99=float(f1_score(yte, st >= q99)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flat', required=True)
    p.add_argument('--dataset', required=True)
    a = p.parse_args()
    (Xtr, ytr, _, Xva, yva, Xte, yte), _ = K.flat_data(f'{a.flat}/{a.dataset}', 'raw')
    Xte = np.ascontiguousarray(Xte)
    started, rows = time.time(), []
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    for seed in SEEDS:
        rows.append(row('plain', seed, fit(seed, Xtr, ytr, Xva, yva, float(np.sqrt(len(neg) / len(pos))), False), Xva, yva, Xte, yte, len(ytr), len(pos)))
        rows.append(row('noweight', seed, fit(seed, Xtr, ytr, Xva, yva, 1.0, True), Xva, yva, Xte, yte, len(ytr), len(pos)))
        rng = np.random.default_rng(seed)
        subsets = {'rows8192': np.sort(rng.choice(len(ytr), min(8192, len(ytr)), replace=False))}
        for k in (10, 50):
            subsets[f'pos{k}'] = np.sort(np.r_[neg, rng.choice(pos, min(k, len(pos)), replace=False)])
        for name, idx in subsets.items():
            y = ytr[idx]
            if y.sum() == 0 or y.sum() == len(y):       # no labelled anomaly in the sample: nothing to learn from
                rows.append(dict(variant=name, seed=seed, train_rows=int(len(y)), train_pos=int(y.sum()), trees=0, val_AP=float(yva.mean()),
                                 test_AP=float(yte.mean()), test_ROC_AUC=0.5, test_F1=0.0, test_F1_q99=0.0, degenerate=True))
            else:
                rows.append(row(name, seed, fit(seed, np.ascontiguousarray(Xtr[idx]), y, Xva, yva, 1.0, True), Xva, yva, Xte, yte, len(y), y.sum()))
        print(a.dataset, seed, ' '.join(f"{r['variant']} {r['test_AP']:.4f}" for r in rows[-5:]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{a.dataset}.json').write_text(json.dumps(dict(dataset=a.dataset, test_rows=int(len(yte)), test_pos=int(yte.sum()),
                                                           rows=rows, seconds=round(time.time() - started, 1)), indent=1) + '\n')


if __name__ == '__main__':
    main()
