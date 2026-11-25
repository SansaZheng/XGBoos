"""Follow-up to baselines.dual_xgb: the loss swap in detail.

Usage: python -m baselines.focal_xgb --dataset NAME [--variant V --flat FOLDER]

  focal g=G       single booster, focal loss with gamma G, no class weight.  G=0 is plain unweighted log-loss
                  through the same custom-objective code path (control for anything but gamma).
  focal-ens       K=5 focal boosters (gamma 2) with different seeds, margins averaged
  focal-ens+N     the same with the novelty feature of baselines.dual_xgb appended
"""
import argparse
import json
import time

import numpy as np
import xgboost as xgb

from . import knob_xgb as K
from .common import RESULTS, SEEDS
from .dual_xgb import MEMBERS, focal_objective, margin, novelty, report, with_extra

OUT = RESULTS / 'dual_xgb'
GAMMAS = (0.0, 0.5, 1.0, 2.0, 3.0)


def focal(seed, data, gamma, extra=None):
    Xtr, ytr, _, Xva, yva, Xte, _ = data
    etr, eva, ete = extra if extra else (None, None, None)
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=1, tree_method='hist', eval_metric='aucpr', early_stopping_rounds=50,
                          objective=focal_objective(gamma), random_state=seed, n_jobs=2)
    m.fit(with_extra(Xtr, etr), ytr, eval_set=[(with_extra(Xva, eva), yva)], verbose=False)
    (sv, trees), (st, _) = margin(m, Xva, eva), margin(m, Xte, ete)
    return sv, st, trees


def ensemble(seed, data, gamma, extra=None):
    sv, st, trees = 0.0, 0.0, []
    for k in range(MEMBERS):
        a, b, t = focal(seed * 1000 + k, data, gamma, extra)
        sv, st = sv + a / MEMBERS, st + b / MEMBERS
        trees.append(t)
    return sv, st, trees


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
    Xtr, ytr, _, Xva, yva, Xte, yte = data
    for seed in SEEDS:
        res, clock = {}, {}
        for g in GAMMAS:
            start = time.time()
            sv, st, trees = focal(seed, data, g)
            res[f'focal g={g:g}'], _ = report(yva, sv, yte, st)
            res[f'focal g={g:g}']['trees'] = [trees]; clock[f'focal g={g:g}'] = round(time.time() - start, 1)
        start = time.time()
        sv, st, trees = ensemble(seed, data, 2.0)
        res['focal-ens'], _ = report(yva, sv, yte, st); res['focal-ens']['trees'] = trees
        clock['focal-ens'] = round(time.time() - start, 1)
        start = time.time()
        ntr, (nva, nte), _ = novelty(seed, Xtr, ytr, [Xva, Xte], names)
        sv, st, trees = ensemble(seed, data, 2.0, (ntr, nva, nte))
        res['focal-ens+N'], _ = report(yva, sv, yte, st); res['focal-ens+N']['trees'] = trees
        clock['focal-ens+N'] = round(time.time() - start, 1)
        folder = OUT / args.dataset; folder.mkdir(parents=True, exist_ok=True)
        (folder / f'focal__{variant}__seed{seed}.json').write_text(json.dumps(
            dict(dataset=args.dataset, variant=variant, seed=seed, results=res, seconds=clock), indent=1) + '\n')
        print(f'[{args.dataset} seed={seed}] ' + ' | '.join(
            f'{k}: AP {v["test_AP"]:.4f} F1 {v["test_F1"]:.3f} FN/FP {v["test_FN"]}/{v["test_FP"]}' for k, v in res.items()), flush=True)


if __name__ == '__main__':
    main()
