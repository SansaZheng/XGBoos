"""The starting score (base_score) as a key component of XGBoost under extreme class imbalance.

Usage: python -m baselines.intercept_xgb --dataset NAME [--variant V --flat FOLDER]

XGBoost >= 2.0 fits the starting score of binary:logistic to the class prior.  With 0.1 % anomalies every
row then starts at p ~ 0.001, so its hessian p(1-p) ~ 0.001: the defaults min_child_weight = 1 and
reg_lambda = 1 are measured in those units and become very strong (a leaf needs about 1/p rows).
Starting from p = 0.5 instead gives every row hessian 0.25 and removes that hidden constraint.

Variants (built-in objective, same splits, threshold rule and seeds as everywhere else):
  prior|sqrt       baseline: fitted prior, scale_pos_weight = sqrt(neg/pos)
  prior|1          fitted prior, no class weight
  half|sqrt        base_score 0.5, sqrt weight
  half|1           base_score 0.5, no class weight                 <- "delete the prior and the class weight"
  prior|1|mcw0     fitted prior, no class weight, min_child_weight 0 (removes the leaf-size part only)
  prior|sqrt|mcw0  baseline with min_child_weight 0
  half|1|ens       five half|1 boosters with different seeds, margins averaged
"""
import argparse
import json
import time

import numpy as np
import xgboost as xgb

from . import knob_xgb as K
from .common import RESULTS, SEEDS
from .dual_xgb import MEMBERS, margin, report

OUT = RESULTS / 'intercept_xgb'
VARIANTS = dict([('prior|sqrt', (None, 'sqrt', 1)), ('prior|1', (None, 'one', 1)), ('half|sqrt', (0.5, 'sqrt', 1)),
                 ('half|1', (0.5, 'one', 1)), ('prior|1|mcw0', (None, 'one', 0)), ('prior|sqrt|mcw0', (None, 'sqrt', 0))])


def fit(seed, data, base_score, weight, mcw):
    Xtr, ytr, _, Xva, yva, Xte, _ = data
    w = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1))) if weight == 'sqrt' else 1.0
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=mcw, scale_pos_weight=w, base_score=base_score, tree_method='hist',
                          eval_metric='aucpr', early_stopping_rounds=50, random_state=seed, n_jobs=2)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    (sv, trees), (st, _) = margin(m, Xva), margin(m, Xte)
    leaves = m.get_booster().trees_to_dataframe()
    leaves = leaves[(leaves.Feature == 'Leaf') & (leaves.Tree < trees)]
    return sv, st, trees, float(leaves.groupby('Tree').size().mean()), float(leaves.Cover.median())


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
    _, ytr, _, _, yva, _, yte = data
    for seed in SEEDS:
        res, clock, base = {}, {}, None
        for name, (bs, weight, mcw) in VARIANTS.items():
            start = time.time()
            sv, st, trees, leaves, cover = fit(seed, data, bs, weight, mcw)
            res[name], alert = report(yva, sv, yte, st, base)
            res[name].update(trees=trees, leaves_per_tree=leaves, median_leaf_cover=cover)
            if base is None:
                base = dict(missed=(yte == 1) & ~alert, caught=(yte == 1) & alert, alerts=alert.sum())
            clock[name] = round(time.time() - start, 1)
        start = time.time(); sv, st, trees = 0.0, 0.0, []
        for k in range(MEMBERS):
            a, b, t, _, _ = fit(seed * 1000 + k, data, 0.5, 'one', 1)
            sv, st = sv + a / MEMBERS, st + b / MEMBERS; trees.append(t)
        res['half|1|ens'], _ = report(yva, sv, yte, st, base); res['half|1|ens']['trees'] = float(np.mean(trees))
        clock['half|1|ens'] = round(time.time() - start, 1)
        folder = OUT / args.dataset; folder.mkdir(parents=True, exist_ok=True)
        (folder / f'intercept__{variant}__seed{seed}.json').write_text(json.dumps(
            dict(dataset=args.dataset, variant=variant, seed=seed, prevalence=float(ytr.mean()), train_positives=int(ytr.sum()),
                 results=res, seconds=clock), indent=1) + '\n')
        print(f'[{args.dataset} seed={seed}] ' + ' | '.join(
            f'{k}: AP {v["test_AP"]:.4f} F1 {v["test_F1"]:.3f} FN/FP {v["test_FN"]}/{v["test_FP"]} t{v["trees"]:.0f}' +
            (f' L{v["leaves_per_tree"]:.0f}' if 'leaves_per_tree' in v else '') for k, v in res.items()), flush=True)


if __name__ == '__main__':
    main()
