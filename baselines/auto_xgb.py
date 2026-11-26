"""Combined XGBoost for anomaly detection: every piece that helped somewhere, chosen per task on validation.

Usage: python -m baselines.auto_xgb --dataset NAME [--variant V --flat FOLDER]

Fixed:   no class weight (scale_pos_weight = 1).
Chosen, one step at a time, by mean validation AP over three seeds (the simpler option wins ties):
  1. starting score      fitted prior  |  0.5
  0. feature variant     (tasks prepared in two variants) current-observation features | temporal features
  2. regularisation      default (depth 6)  |  depth 2  |  depth 4  |  min_child_weight 30  |  reg_lambda 30
                         |  leaf-wise trees with 15 leaves
  3. novelty feature     off  |  on   (label-free real-vs-swapped XGBoost margin, see baselines.dual_xgb)
  4. seed ensemble       single booster  |  five boosters with different seeds, margins averaged
Test labels are used only to report each candidate; they never enter a choice.
"""
import argparse
import json
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from . import knob_xgb as K
from .common import RESULTS, SEEDS
from .dual_xgb import MEMBERS, margin, novelty, report, with_extra

OUT = RESULTS / 'auto_xgb'
REG = dict([('default', {}), ('depth2', dict(max_depth=2)), ('depth4', dict(max_depth=4)),
            ('mcw30', dict(min_child_weight=30)), ('lam30', dict(reg_lambda=30.0)),
            ('leaves15', dict(max_depth=0, grow_policy='lossguide', max_leaves=15))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--variant', default=None)
    p.add_argument('--flat', default=None)
    p.add_argument('--out', default=None, help='results subfolder (default auto_xgb)')
    args = p.parse_args()
    out_dir = RESULTS / args.out if args.out else OUT
    if args.dataset in K.MARKET and not args.flat:
        variant = args.variant or K.MARKET[args.dataset]
        data, names = K.market_data(args.dataset, variant)
    else:
        variant = args.variant
        variant_trace = []
        if variant is None:        # step 0: pick the feature variant with the plain no-weight model
            from pathlib import Path
            options = json.loads((Path(args.flat) / 'info.json').read_text())['variants']
            for v in options:
                d, _ = K.flat_data(args.flat, v)
                val, test = [], []
                for seed in SEEDS:
                    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                          min_child_weight=1, scale_pos_weight=1.0, tree_method='hist', eval_metric='aucpr',
                                          early_stopping_rounds=50, random_state=seed, n_jobs=2).fit(d[0], d[1], eval_set=[(d[3], d[4])], verbose=False)
                    val.append(float(average_precision_score(d[4], margin(m, d[3])[0]))); test.append(float(average_precision_score(d[6], margin(m, d[5])[0])))
                variant_trace.append(dict(variant=v, val_AP=float(np.mean(val)), test_AP=float(np.mean(test))))
                print(f'  {args.dataset} step 0 variant {v}: val {np.mean(val):.4f} test {np.mean(test):.4f}', flush=True)
                del d
            variant = max(variant_trace, key=lambda r: r['val_AP'])['variant']
        data, names = K.flat_data(args.flat, variant)
    Xtr, ytr, _, Xva, yva, Xte, yte = data
    started = time.time()
    nov, cache = {}, {}

    def extra(seed):
        if seed not in nov:
            ntr, (nva, nte), _ = novelty(seed, Xtr, ytr, [Xva, Xte], names)
            nov[seed] = (ntr, nva, nte)
        return nov[seed]

    def fit(seed, start, reg, use_nov, model_seed=None):
        key = (seed, start, reg, use_nov, model_seed)
        if key not in cache:
            etr, eva, ete = extra(seed) if use_nov else (None, None, None)
            params = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=1, scale_pos_weight=1.0, tree_method='hist', eval_metric='aucpr',
                          early_stopping_rounds=50, random_state=seed if model_seed is None else model_seed, n_jobs=2,
                          base_score=0.5 if start == 'half' else None)
            params.update(REG[reg])
            m = xgb.XGBClassifier(**params).fit(with_extra(Xtr, etr), ytr, eval_set=[(with_extra(Xva, eva), yva)], verbose=False)
            (sv, trees), (st, _) = margin(m, Xva, eva), margin(m, Xte, ete)
            cache[key] = (sv, st, trees)
        return cache[key]

    def scores(seed, cfg):
        start, reg, use_nov, ens = cfg
        if not ens:
            return fit(seed, start, reg, use_nov)
        parts = [fit(seed, start, reg, use_nov, seed * 1000 + k) for k in range(MEMBERS)]
        return (np.mean([a for a, _, _ in parts], 0), np.mean([b for _, b, _ in parts], 0), float(np.mean([t for _, _, t in parts])))

    trace = []

    def evaluate(step, cfg):
        runs = [scores(s, cfg) for s in SEEDS]
        val = [float(average_precision_score(yva, a)) for a, _, _ in runs]
        test = [float(average_precision_score(yte, b)) for _, b, _ in runs]
        row = dict(step=step, start=cfg[0], reg=cfg[1], novelty=cfg[2], ensemble=cfg[3], val_AP=float(np.mean(val)),
                   val_AP_seeds=val, test_AP=float(np.mean(test)), test_AP_sd=float(np.std(test, ddof=1)), test_AP_seeds=test,
                   trees=float(np.mean([t for _, _, t in runs])))
        trace.append(row)
        print(f'  {args.dataset} step {step} {cfg}: val {row["val_AP"]:.4f} test {row["test_AP"]:.4f} ± {row["test_AP_sd"]:.4f} trees {row["trees"]:.0f}', flush=True)
        return row['val_AP']

    best = ('prior', 'default', False, False); best_val = evaluate(1, best)
    def consider(step, cfg):
        nonlocal best, best_val
        v = evaluate(step, cfg)
        if v > best_val + 1e-12:
            best, best_val = cfg, v
    consider(1, ('half', 'default', False, False))
    start = best[0]
    for reg in list(REG)[1:]:
        consider(2, (start, reg, False, False))
    consider(3, (best[0], best[1], True, False))
    consider(4, (best[0], best[1], best[2], True))

    final = []
    for seed in SEEDS:
        sv, st, trees = scores(seed, best)
        r, _ = report(yva, sv, yte, st)
        final.append(dict(seed=seed, trees=float(np.mean(trees)), **r))
    out = dict(dataset=args.dataset, variant=variant, chosen=dict(start=best[0], reg=best[1], novelty=best[2], ensemble=best[3]),
               chosen_val_AP=best_val, variant_trace=locals().get('variant_trace', []), trace=trace, final=final, seconds=round(time.time() - started, 1),
               best_test_candidate=max(trace, key=lambda r: r['test_AP']))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{args.dataset}__{variant}.json').write_text(json.dumps(out, indent=1) + '\n')
    m = lambda k: float(np.mean([f[k] for f in final]))
    print(f'[{args.dataset}] chosen {out["chosen"]} -> test AP {m("test_AP"):.4f} ± {np.std([f["test_AP"] for f in final], ddof=1):.4f} '
          f'F1 {m("test_F1"):.4f} FN {m("test_FN"):.1f} FP {m("test_FP"):.1f} | best candidate on test '
          f'{out["best_test_candidate"]["test_AP"]:.4f} | {out["seconds"]}s', flush=True)


if __name__ == '__main__':
    main()
