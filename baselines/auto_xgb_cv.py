"""baselines.auto_xgb with a selection rule that does not rest on the small validation split alone.

Usage: python -m baselines.auto_xgb_cv --dataset NAME [--variant V --flat FOLDER]

Same candidates as the extended search of baselines.auto_xgb (starting score; regularisation incl. leaf-wise
trees; novelty feature; seed ensemble).  Steps 1-3 are scored by
    (P_train * grouped 3-fold CV AP on the training rows  +  P_val * validation AP) / (P_train + P_val)
where P is the number of anomalies in each part, so the part with more anomalies counts more.  Fold models are
early-stopped on the validation split like every other model.  Step 4 (ensemble or not) uses mean validation
AP over three seeds, as before.  Test labels never enter a choice.
"""
import argparse
import json
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from . import knob_xgb as K
from .auto_xgb import REG
from .common import RESULTS, SEEDS
from .dual_xgb import MEMBERS, margin, novelty, report, with_extra

OUT = RESULTS / 'auto_xgb_cv'


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
    Xtr, ytr, gtr, Xva, yva, Xte, yte = data
    started = time.time()
    nov, cache = {}, {}
    folds = list(GroupKFold(min(3, len(np.unique(gtr)))).split(Xtr, ytr, gtr))
    p_tr, p_va = int(ytr.sum()), int(yva.sum())

    def extra(seed):
        if seed not in nov:
            ntr, (nva, nte), _ = novelty(seed, Xtr, ytr, [Xva, Xte], names)
            nov[seed] = (ntr, nva, nte)
        return nov[seed]

    def model(start, reg, model_seed):
        params = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                      scale_pos_weight=1.0, tree_method='hist', eval_metric='aucpr', early_stopping_rounds=50,
                      random_state=model_seed, n_jobs=2, base_score=0.5 if start == 'half' else None)
        params.update(REG[reg])
        return xgb.XGBClassifier(**params)

    def fit(seed, start, reg, use_nov, model_seed=None):
        key = (seed, start, reg, use_nov, model_seed)
        if key not in cache:
            etr, eva, ete = extra(seed) if use_nov else (None, None, None)
            m = model(start, reg, seed if model_seed is None else model_seed).fit(
                with_extra(Xtr, etr), ytr, eval_set=[(with_extra(Xva, eva), yva)], verbose=False)
            (sv, trees), (st, _) = margin(m, Xva, eva), margin(m, Xte, ete)
            cache[key] = (sv, st, trees)
        return cache[key]

    def cv_ap(start, reg, use_nov, seed=SEEDS[0]):
        etr, eva, _ = extra(seed) if use_nov else (None, None, None)
        Va = with_extra(Xva, eva)
        aps = []
        for a, b in folds:
            if ytr[b].sum() == 0 or ytr[a].sum() == 0:
                continue
            m = model(start, reg, seed).fit(with_extra(Xtr, etr, a), ytr[a], eval_set=[(Va, yva)], verbose=False)
            s, _ = margin(m, Xtr[b] if etr is None else with_extra(Xtr, etr, b))
            aps.append(float(average_precision_score(ytr[b], s)))
        return float(np.mean(aps))

    trace = []

    def evaluate(step, cfg):
        start, reg, use_nov, _ = cfg
        sv, st, trees = fit(SEEDS[0], start, reg, use_nov)
        val, cv = float(average_precision_score(yva, sv)), cv_ap(start, reg, use_nov)
        score = (p_tr * cv + p_va * val) / (p_tr + p_va)
        row = dict(step=step, start=start, reg=reg, novelty=use_nov, ensemble=False, cv_AP=cv, val_AP=val, score=score,
                   test_AP_seed42=float(average_precision_score(yte, st)), trees=trees)
        trace.append(row)
        print(f'  {args.dataset} step {step} {cfg}: cv {cv:.4f} val {val:.4f} score {score:.4f} | test(seed 42) {row["test_AP_seed42"]:.4f}', flush=True)
        return score

    best = ('prior', 'default', False, False); best_score = evaluate(1, best)
    for step, make in ((1, lambda: [('half', 'default', False, False)]),
                       (2, lambda: [(best[0], r, False, False) for r in list(REG)[1:]]),
                       (3, lambda: [(best[0], best[1], True, False)])):
        for cfg in make():
            s = evaluate(step, cfg)
            if s > best_score + 1e-12:
                best, best_score = cfg, s

    def scores(seed, cfg):
        start, reg, use_nov, ens = cfg
        if not ens:
            return fit(seed, start, reg, use_nov)
        parts = [fit(seed, start, reg, use_nov, seed * 1000 + k) for k in range(MEMBERS)]
        return np.mean([a for a, _, _ in parts], 0), np.mean([b for _, b, _ in parts], 0), float(np.mean([t for _, _, t in parts]))
    single = [scores(s, best) for s in SEEDS]
    ens_cfg = (best[0], best[1], best[2], True)
    ens = [scores(s, ens_cfg) for s in SEEDS]
    val_single = float(np.mean([average_precision_score(yva, a) for a, _, _ in single]))
    val_ens = float(np.mean([average_precision_score(yva, a) for a, _, _ in ens]))
    step4 = dict(val_single=val_single, val_ensemble=val_ens,
                 test_single=float(np.mean([average_precision_score(yte, b) for _, b, _ in single])),
                 test_ensemble=float(np.mean([average_precision_score(yte, b) for _, b, _ in ens])))
    chosen, runs = (ens_cfg, ens) if val_ens > val_single + 1e-12 else (best, single)
    final = []
    for seed, (sv, st, trees) in zip(SEEDS, runs):
        r, _ = report(yva, sv, yte, st)
        final.append(dict(seed=seed, trees=float(np.mean(trees)), **r))
    out = dict(dataset=args.dataset, variant=variant, train_positives=p_tr, validation_positives=p_va,
               chosen=dict(start=chosen[0], reg=chosen[1], novelty=chosen[2], ensemble=chosen[3]), trace=trace, step4=step4,
               final=final, seconds=round(time.time() - started, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{args.dataset}__{variant}.json').write_text(json.dumps(out, indent=1) + '\n')
    m = lambda k: float(np.mean([f[k] for f in final]))
    print(f'[{args.dataset}] chosen {out["chosen"]} -> test AP {m("test_AP"):.4f} ± {np.std([f["test_AP"] for f in final], ddof=1):.4f} '
          f'F1 {m("test_F1"):.4f} FN {m("test_FN"):.1f} FP {m("test_FP"):.1f} | {out["seconds"]}s', flush=True)


if __name__ == '__main__':
    main()
