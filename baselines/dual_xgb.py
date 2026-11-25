"""Structural changes to XGBoost for anomaly detection, with an ablation of every piece.

Usage: python -m baselines.dual_xgb --dataset NAME [--variant V --flat FOLDER]

Removed from the baseline
  scale_pos_weight          - earlier analysis: its baseline value is the least stable setting across seeds
  the single early-stopped  - one booster stopped on a validation split with 33-100 positives
  model
Added
  balanced bagging (B)      - K boosters, each fitted on ALL anomalies plus a fresh random subset of normals
                              (max(10 x anomalies, 20,000) rows), weight 1 on both classes, margins averaged
  novelty branch (N)        - a second XGBoost that never sees an anomaly label: it is trained to tell real
                              NORMAL rows from synthetic rows made by swapping whole feature groups between
                              two normal rows (marginals kept, joint structure broken).  Its margin says how
                              far a row is from the joint structure of normal data.  Cross-fitted on the
                              training rows and appended as one extra input feature.
Variants (same splits, same threshold rule, three seeds)
  base        baseline XGBoost (sqrt class-ratio weight, early stopping)
  ens         control: K baseline boosters with different seeds, margins averaged (ensembling alone)
  bag         B
  focal       loss swapped: focal loss (gamma 2) in place of weighted log-loss, single booster, no class weight
  base+N      baseline with the novelty feature
  bag+N       B and N together (the proposed model)
  nov         the novelty branch alone, used as an unsupervised score
"""
import argparse
import json
import re
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from . import knob_xgb as K
from .common import RESULTS, SEEDS, metrics, pick_threshold

OUT = RESULTS / 'dual_xgb'
MEMBERS = 5
CHUNK = 500_000


def booster(seed, weight, rounds=600, early=True):
    return xgb.XGBClassifier(n_estimators=rounds, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=1, scale_pos_weight=weight, tree_method='hist', eval_metric='aucpr',
                             early_stopping_rounds=50 if early else None, random_state=seed, n_jobs=2)


def with_extra(X, extra, rows=None):
    X = X if rows is None else X[rows]
    return X if extra is None else np.column_stack([X, extra if rows is None else extra[rows]]).astype(np.float32)


def margin(model, X, extra=None):
    trees = model.best_iteration + 1 if model.get_params()['early_stopping_rounds'] else model.get_params()['n_estimators']
    b = model.get_booster()
    out = [b.predict(xgb.DMatrix(with_extra(X[i:i + CHUNK], None if extra is None else extra[i:i + CHUNK])),
                     output_margin=True, iteration_range=(0, trees)).astype(np.float64) for i in range(0, len(X), CHUNK)]
    return np.concatenate(out), int(trees)


# ----------------------------------------------------------------------------- novelty branch
def feature_groups(names):
    """Columns that describe the same raw field move together (one-hot levels, temporal variants)."""
    def key(n):
        n = re.sub(r'-(lag\d+|mean\d+)$', '', n)
        m = re.match(r'std\d+\((.*)\)$', n)
        n = m.group(1) if m else n
        if n.startswith('category_'):
            return 'category_' + n.split('_')[1]
        return n
    keys = [key(n) for n in names]
    uniq = sorted(set(keys))
    groups = [[i for i, k in enumerate(keys) if k == u] for u in uniq]
    return groups if len(groups) >= 3 else [[i] for i in range(len(names))]


def synthetic(real, groups, rng):
    """Swap about half of the feature groups of every row with those of another real row."""
    out = real.copy()
    partner = rng.permutation(len(real))
    for g in groups:
        take = rng.random(len(real)) < 0.5
        out[np.ix_(take, g)] = real[partner][np.ix_(take, g)]
    return out


def novelty(seed, Xtr, ytr, others, names, cap=100_000):
    """Cross-fitted novelty margin for the training rows, averaged margin for the other splits."""
    rng = np.random.default_rng(seed + 7)
    groups = feature_groups(names)
    normal = np.flatnonzero(ytr == 0)
    pick = rng.permutation(normal)[:min(cap, len(normal))]
    halves = [np.sort(pick[::2]), np.sort(pick[1::2])]
    models = []
    for h in halves:
        real = np.ascontiguousarray(Xtr[h])
        X = np.vstack([real, synthetic(real, groups, rng)])
        y = np.r_[np.zeros(len(real)), np.ones(len(real))]
        m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              tree_method='hist', random_state=seed, n_jobs=2).fit(X, y)
        models.append(m)
    score = lambda m, X: np.concatenate([m.get_booster().predict(xgb.DMatrix(X[i:i + CHUNK]), output_margin=True)
                                         for i in range(0, len(X), CHUNK)]).astype(np.float32)
    s = [score(m, Xtr) for m in models]
    tr = (s[0] + s[1]) / 2
    tr[halves[0]] = s[1][halves[0]]          # a row used to fit one model is scored by the other
    tr[halves[1]] = s[0][halves[1]]
    return tr, [(score(models[0], X) + score(models[1], X)) / 2 for X in others], len(groups)


# ----------------------------------------------------------------------------- supervised branches
def fit_single(seed, data, extra=None):
    Xtr, ytr, _, Xva, yva, Xte, _ = data
    etr, eva, ete = extra if extra else (None, None, None)
    w = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1)))
    m = booster(seed, w).fit(with_extra(Xtr, etr), ytr, eval_set=[(with_extra(Xva, eva), yva)], verbose=False)
    (sv, trees), (st, _) = margin(m, Xva, eva), margin(m, Xte, ete)
    return sv, st, [trees]


def focal_objective(gamma=2.0):
    """Focal loss -(1 - pt)^gamma log(pt) on the raw margin; gradient and (clipped) hessian."""
    def objective(y, z):
        s = 2.0 * y - 1.0
        pt = np.clip(1.0 / (1.0 + np.exp(-s * z)), 1e-9, 1 - 1e-9)
        u = (1 - pt) ** gamma
        v = gamma * pt * np.log(pt) - (1 - pt)
        grad = s * u * v
        hess = u * pt * ((1 - pt) * (gamma * np.log(pt) + gamma + 1) - gamma * v)
        return grad, np.maximum(hess, 1e-6)
    return objective


def fit_focal(seed, data):
    Xtr, ytr, _, Xva, yva, Xte, _ = data
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=1, tree_method='hist', eval_metric='aucpr', early_stopping_rounds=50,
                          objective=focal_objective(), random_state=seed, n_jobs=2)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    (sv, trees), (st, _) = margin(m, Xva), margin(m, Xte)
    return sv, st, [trees]


def fit_ensemble(seed, data, extra=None, balanced=False):
    Xtr, ytr, _, Xva, yva, Xte, _ = data
    etr, eva, ete = extra if extra else (None, None, None)
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    Va = with_extra(Xva, eva)
    sv, st, trees = 0.0, 0.0, []
    for k in range(MEMBERS):
        s = seed * 1000 + k
        if balanced:
            n = min(len(neg), max(10 * len(pos), 20_000))
            rows = np.sort(np.r_[pos, np.random.default_rng(s).choice(neg, n, replace=False)])
            m = booster(s, 1.0).fit(with_extra(Xtr, etr, rows), ytr[rows], eval_set=[(Va, yva)], verbose=False)
        else:
            w = float(np.sqrt(len(neg) / max(len(pos), 1)))
            m = booster(s, w).fit(with_extra(Xtr, etr), ytr, eval_set=[(Va, yva)], verbose=False)
        (a, t), (b, _) = margin(m, Xva, eva), margin(m, Xte, ete)
        sv, st = sv + a / MEMBERS, st + b / MEMBERS
        trees.append(t)
    return sv, st, trees


def report(yva, sv, yte, st, base=None):
    thr = pick_threshold(yva, sv)
    v, t = metrics(yva, sv, thr), metrics(yte, st, thr)
    out = dict(val_AP=v['AP'], test_AP=t['AP'], test_ROC_AUC=t['ROC_AUC'], test_F1=t['F1'], test_precision=t['precision'],
               test_recall=t['recall'], test_TP=t['TP'], test_FP=t['FP'], test_FN=t['FN'])
    alert = st >= thr
    if base is not None:
        out['missed_by_base_recovered'] = int((base['missed'] & alert).sum())
        out['caught_by_base_lost'] = int((base['caught'] & ~alert).sum())
        k = int(base['alerts'])
        top = np.argsort(-st, kind='stable')[:k]
        out['recall_at_base_alert_budget'] = float(yte[top].sum() / max(yte.sum(), 1))
    return out, alert


def run(dataset, variant, seed, data, names):
    Xtr, ytr, _, Xva, yva, Xte, yte = data
    res, clock = {}, {}

    def timed(name, fn):
        start = time.time(); out = fn(); clock[name] = round(time.time() - start, 1); return out

    sv, st, trees = timed('base', lambda: fit_single(seed, data))
    res['base'], alert = report(yva, sv, yte, st)
    res['base']['trees'] = trees
    base = dict(missed=(yte == 1) & ~alert, caught=(yte == 1) & alert, alerts=alert.sum())
    res['base']['recall_at_base_alert_budget'] = res['base']['test_recall']
    ntr, (nva, nte), n_groups = timed('novelty', lambda: novelty(seed, Xtr, ytr, [Xva, Xte], names))
    extra = (ntr, nva, nte)
    res['nov'], _ = report(yva, nva.astype(np.float64), yte, nte.astype(np.float64), base)
    for name, fn in (('ens', lambda: fit_ensemble(seed, data)),
                     ('bag', lambda: fit_ensemble(seed, data, balanced=True)),
                     ('focal', lambda: fit_focal(seed, data)),
                     ('base+N', lambda: fit_single(seed, data, extra)),
                     ('bag+N', lambda: fit_ensemble(seed, data, extra, balanced=True))):
        sv, st, trees = timed(name, fn)
        res[name], _ = report(yva, sv, yte, st, base)
        res[name]['trees'] = trees
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())
    out = dict(dataset=dataset, variant=variant, seed=seed, features=int(Xtr.shape[1]), feature_groups=n_groups,
               train_rows=len(ytr), train_positives=pos, bag_negatives=min(neg, max(10 * pos, 20_000)),
               members=MEMBERS, seconds=clock, results=res,
               novelty_val_ROC_AUC=float(roc_auc_score(yva, nva)), novelty_val_AP=float(average_precision_score(yva, nva)))
    folder = OUT / dataset; folder.mkdir(parents=True, exist_ok=True)
    (folder / f'dual__{variant}__seed{seed}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f'[{dataset} {variant} seed={seed}] ' + ' | '.join(
        f'{k}: AP {v["test_AP"]:.4f} F1 {v["test_F1"]:.3f} FN/FP {v["test_FN"]}/{v["test_FP"]}' for k, v in res.items())
          + f' | novelty AUROC {res["nov"]["test_ROC_AUC"]:.3f} | {clock}', flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--variant', default=None)
    p.add_argument('--flat', default=None)
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    args = p.parse_args()
    if args.dataset in K.MARKET and not args.flat:
        variant = args.variant or K.MARKET[args.dataset]
        data, names = K.market_data(args.dataset, variant)
    else:
        variant = args.variant
        data, names = K.flat_data(args.flat, variant)
    for seed in map(int, args.seeds.split(',')):
        run(args.dataset, variant, seed, data, names)


if __name__ == '__main__':
    main()
