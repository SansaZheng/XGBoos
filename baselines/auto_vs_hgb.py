"""Paired bootstrap of test AP: validation-chosen XGBoost (baselines.auto_xgb, first search space) against the
HistGradientBoosting control of Kronos_SupervisedAD, on identical test rows.

Usage: python -m baselines.auto_vs_hgb --dataset NAME --flat FOLDER [--resamples 500]
Rows are resampled independently, which ignores dependence inside sequences, so intervals are if anything too narrow.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from . import knob_xgb as K
from .auto_xgb import REG
from .common import RESULTS
from .dual_xgb import MEMBERS, margin, novelty, with_extra

OUT = RESULTS / 'auto_xgb'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--flat', required=True)
    p.add_argument('--resamples', type=int, default=500)
    args = p.parse_args()
    rec = json.loads(next((RESULTS / 'auto_xgb_first').glob(f'{args.dataset}__*.json')).read_text())
    c, variant, seed = rec['chosen'], rec['variant'], 42
    data, names = K.flat_data(args.flat, variant)
    Xtr, ytr, _, Xva, yva, Xte, yte = data
    etr = eva = ete = None
    if c['novelty']:
        etr, (eva, ete), _ = novelty(seed, Xtr, ytr, [Xva, Xte], names)
    ours = 0.0
    for ms in ([seed * 1000 + k for k in range(MEMBERS)] if c['ensemble'] else [seed]):
        params = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                      scale_pos_weight=1.0, tree_method='hist', eval_metric='aucpr', early_stopping_rounds=50, random_state=ms,
                      n_jobs=2, base_score=0.5 if c['start'] == 'half' else None)
        params.update(REG[c['reg']])
        m = xgb.XGBClassifier(**params).fit(with_extra(Xtr, etr), ytr, eval_set=[(with_extra(Xva, eva), yva)], verbose=False)
        ours = ours + margin(m, Xte, ete)[0]
    info = json.loads((Path(args.flat) / 'info.json').read_text())
    hv = 'summary' if info['task'] == 'sequence' else 'temporal'
    (Htr, hy, _, _, _, Hte, hyte), _ = K.flat_data(args.flat, hv)
    assert np.array_equal(hyte, yte)
    h = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1., class_weight='balanced',
                                       early_stopping=False, random_state=seed).fit(Htr, hy)
    hgb = np.concatenate([h.predict_proba(Hte[b:b + 500_000])[:, 1] for b in range(0, len(Hte), 500_000)])
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    diffs = []
    for _ in range(args.resamples):       # resample anomalies and normals separately (keeps both classes present)
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        diffs.append(average_precision_score(yte[i], ours[i]) - average_precision_score(yte[i], hgb[i]))
    diffs = np.array(diffs)
    out = dict(dataset=args.dataset, chosen=c, variant=variant, seed=seed, test_positives=int(len(pos)),
               ours_AP=float(average_precision_score(yte, ours)), hgb_AP=float(average_precision_score(yte, hgb)),
               diff=float(average_precision_score(yte, ours) - average_precision_score(yte, hgb)),
               ci95=[float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
               share_resamples_ours_higher=float((diffs > 0).mean()), resamples=args.resamples)
    (OUT / f'vs_hgb__{args.dataset}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(out, flush=True)


if __name__ == '__main__':
    main()
