"""Which XGBoost parameters control the anomaly / normal distinction, and what turning them does.

Usage: python -m baselines.knob_params --dataset NAME [--variant V --flat FOLDER] [--part model|hyper|both]

Two readings of "parameter", both treated as knobs.

A. Parameters inside the fitted model (part=model).  The Knob method of Huang, Mao, Jiang & Cambria with
   the boundary changed from correct/incorrect to ANOMALY / NORMAL:
     hidden state   H[l, j] = TreeSHAP contribution of feature j from boosting block l (L blocks);
                    margin = bias + sum H
     boundary       multi-view CNN / MLP / LR probes predict the label from H, out-of-fold on training rows
     knobs          features ranked by s_j = sum_l |dA_lj| (vertical CNN map); k_knob = first k whose probe
                    is within 1 % of the full map; random subsets of the same size as control
     share          how much of the anomaly-minus-normal margin gap each feature / block carries
     turning        a gain g on a knob: margin' = margin + (g - 1) * (that feature's or block's contribution).
                    A block gain is exactly a rescaling of the leaf weights of that block's trees.
                    Gains are tuned greedily on validation AP; random features tuned the same way are the control.
B. Training hyper-parameters (part=hyper).  One-at-a-time scan around the baseline setting; the sensitivity
   of a hyper-parameter is the range of validation AP it produces.  The two most sensitive ones are tuned
   jointly on validation AP and the chosen setting is rerun on three seeds.
Test labels are used only for the reported metrics.
"""
import argparse
import itertools
import json
import math
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import knob_xgb as K
from .common import RESULTS, SEEDS, metrics, pick_threshold

OUT = RESULTS / 'knob_params'
GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
BASE = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, reg_lambda=1.0, gamma=0.0, max_delta_step=0.0, early_stopping_rounds=50)
SCAN = dict(scale_pos_weight=('one', 'ratio'), max_depth=(3, 9), min_child_weight=(5, 20),
            learning_rate=(0.02, 0.2), subsample=(0.5, 1.0), colsample_bytree=(0.5, 1.0),
            reg_lambda=(0.1, 10.0), gamma=(1.0, 5.0), max_delta_step=(1.0, 5.0), rounds=(100, 300))


# ----------------------------------------------------------------------------- shared helpers
def build(seed, ytr, **over):
    ratio = (ytr == 0).sum() / max(ytr.sum(), 1)
    p = dict(BASE, tree_method='hist', eval_metric='aucpr', random_state=seed, n_jobs=2,
             scale_pos_weight=float(np.sqrt(ratio)))
    for k, v in over.items():
        if k == 'scale_pos_weight':
            p[k] = {'one': 1.0, 'sqrt': float(np.sqrt(ratio)), 'ratio': float(ratio)}[v]
        elif k == 'rounds':           # fixed number of rounds, no early stopping
            if v != 'early':
                p['n_estimators'], p['early_stopping_rounds'] = int(v), None
        else:
            p[k] = v
    return xgb.XGBClassifier(**p)


def fit(seed, data, **over):
    Xtr, ytr, _, Xva, yva, Xte, yte = data
    model = build(seed, ytr, **over).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    trees = model.best_iteration + 1 if model.get_params()['early_stopping_rounds'] else model.get_params()['n_estimators']
    return model, int(trees)


def summary(yva, sva, yte, ste):
    thr = pick_threshold(yva, sva)
    v, t = metrics(yva, sva, thr), metrics(yte, ste, thr)
    gap = lambda y, s: float((s[y == 1].mean() - s[y == 0].mean()) / (s.std() + 1e-12))
    return dict(val_AP=v['AP'], test_AP=t['AP'], test_F1=t['F1'], test_precision=t['precision'],
                test_recall=t['recall'], test_TP=t['TP'], test_FP=t['FP'], test_FN=t['FN'],
                test_ROC_AUC=t['ROC_AUC'], val_margin_gap_sd=gap(yva, sva), test_margin_gap_sd=gap(yte, ste))


def sums(model, X, trees, layers):
    """Per-row contribution of every feature (summed over blocks) and of every block (summed over features)."""
    chunk = max(20_000, int(4e7 / (layers * X.shape[1])))
    F, B, M = [], [], []
    for b in range(0, len(X), chunk):
        H, m = K.contribution_map(model, X[b:b + chunk], trees, layers)
        F.append(H.sum(1)); B.append(H.sum(2)); M.append(m)
    return np.concatenate(F), np.concatenate(B), np.concatenate(M)


def tune(C_va, m_va, y_va, cols):
    """Greedy gain per knob, in rank order, by validation AP (gain 1 = untouched; ties keep 1)."""
    gains, shift = {}, np.zeros(len(m_va))
    best = average_precision_score(y_va, m_va)
    for j in cols:
        pick = 1.0
        for g in GAINS:
            ap = average_precision_score(y_va, m_va + shift + (g - 1) * C_va[:, j])
            if ap > best + 1e-12:
                best, pick = ap, g
        gains[int(j)] = pick
        shift = shift + (pick - 1) * C_va[:, j]
    return gains


def apply(C, m, gains):
    out = m.copy()
    for j, g in gains.items():
        out += (g - 1) * C[:, j]
    return out


# ----------------------------------------------------------------------------- A. knobs inside the model
def model_knobs(dataset, variant, seed, data, names, layers=K.LAYERS):
    Xtr, ytr, gtr, Xva, yva, Xte, yte = data
    K.SLICED = len(Xte) > 500_000
    rng = np.random.default_rng(seed)
    model, trees = fit(seed, data)
    L = len(np.unique(np.linspace(0, trees, layers + 1).round().astype(int))) - 1
    d = Xtr.shape[1]
    Fva, Bva, mva = sums(model, Xva, trees, layers)
    Fte, Bte, mte = sums(model, Xte, trees, layers)
    base = summary(yva, mva, yte, mte)
    print(f'[{dataset} {variant} seed={seed}] trees={trees} L={L} d={d} baseline val AP={base["val_AP"]:.4f} '
          f'test AP={base["test_AP"]:.4f} F1={base["test_F1"]:.4f} FN={base["test_FN"]}', flush=True)

    # anomaly / normal boundary on out-of-fold hidden states
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    idx = np.sort(np.r_[rng.choice(pos, min(len(pos), 3000), replace=False),
                        rng.choice(neg, min(len(neg), 3000), replace=False)])
    Hp = np.zeros((len(idx), L, d), dtype=np.float32)
    pw = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1)))
    for a, b in GroupKFold(min(5, len(np.unique(gtr)))).split(Xtr, ytr, gtr):
        inside = np.isin(idx, b)
        if inside.any():
            fold = K.xgb_model(seed, pw, rounds=trees, early=False).fit(Xtr[a], ytr[a], verbose=False)
            Hp[inside], _ = K.contribution_map(fold, Xtr[idx[inside]], trees, layers)
    yp, gp = ytr[idx], gtr[idx]
    wp = np.where(yp == 1, 0.5 / max(yp.sum(), 1), 0.5 / max((yp == 0).sum(), 1)) * len(yp)
    folds = list(GroupKFold(min(5, len(np.unique(gp)))).split(Hp, yp, gp))
    acc_full = K.cv_accuracy(('cnn', 'mlp', 'lr'), seed, Hp, yp, wp, gp, folds)
    _, delta = K.probe('cnn', seed, Hp, yp, wp, Hp)
    s_feat, s_block = np.abs(delta).sum(0), np.abs(delta).sum(1)
    order = np.argsort(-s_feat)
    grid = range(1, d + 1) if d <= 16 else sorted({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, d} & set(range(1, d + 1)))
    acc_k, k_knob = {}, None
    for k in grid:
        acc_k[k] = max(K.cv_accuracy(('mlp', 'lr'), seed, Hp[:, :, order[:k]], yp, wp, gp, folds).values())
        if k_knob is None and acc_k[k] >= max(acc_full.values()) - K.DELTA:
            k_knob = k
        if k_knob is not None and k >= min(d, max(k_knob + 2, 4)):
            break
    k_knob = k_knob or d
    J = [int(j) for j in order[:k_knob]]
    randoms = []
    if math.comb(d, k_knob) > 1:
        seen = {tuple(sorted(J))}
        for _ in range(200):
            c = tuple(sorted(int(j) for j in rng.choice(d, k_knob, replace=False)))
            if c not in seen:
                seen.add(c); randoms.append(list(c))
            if len(randoms) == 5:
                break
    acc_random = [max(K.cv_accuracy(('mlp', 'lr'), seed, Hp[:, :, c], yp, wp, gp, folds).values()) for c in randoms]
    del Hp

    # share of the anomaly-minus-normal margin gap carried by each feature and each block (validation rows)
    gap = lambda C: C[yva == 1].mean(0) - C[yva == 0].mean(0)
    total = float(mva[yva == 1].mean() - mva[yva == 0].mean())
    share_feat, share_block = gap(Fva) / total, gap(Bva) / total
    by_share = [int(j) for j in np.argsort(-np.abs(share_feat))]

    # turning the knobs: greedy gains on validation, reported on test
    def turned(Cva, Cte, cols):
        g = tune(Cva, mva, yva, cols)
        return dict(gains=g, **summary(yva, apply(Cva, mva, g), yte, apply(Cte, mte, g)))
    knob_gain = turned(Fva, Fte, J)
    random_gain = [turned(Fva, Fte, c) for c in randoms]
    share_gain = turned(Fva, Fte, by_share[:k_knob])
    block_gain = turned(Bva, Bte, [int(b) for b in np.argsort(-s_block)])
    # dose-response of the single strongest knob feature and of every block
    dose = lambda Cva, Cte, j: [dict(gain=g, **summary(yva, mva + (g - 1) * Cva[:, j], yte, mte + (g - 1) * Cte[:, j]))
                                for g in GAINS]
    out = dict(dataset=dataset, variant=variant, seed=seed, trees=trees, layers=L, features=d, baseline=base,
               boundary_cv_balanced_accuracy=acc_full, acc_top_k=acc_k, k_knob=k_knob,
               knob_features=[names[j] for j in J], acc_knob=acc_k[k_knob],
               acc_random=acc_random, random_features=[[names[j] for j in c] for c in randoms],
               importance={names[j]: float(s_feat[j]) for j in order[:20]},
               block_importance=s_block.tolist(), late_ratio=float(s_block[L // 2:].sum() / s_block.sum()),
               margin_gap=total, share_by_feature={names[j]: float(share_feat[j]) for j in by_share[:20]},
               share_of_knobs=float(share_feat[J].sum()), share_by_block=share_block.tolist(),
               rank_of_knobs_by_share=[by_share.index(j) + 1 for j in J],
               knob_gain=dict(knob_gain, names={names[j]: g for j, g in knob_gain['gains'].items()}),
               random_gain=random_gain,
               share_gain=dict(share_gain, names={names[j]: g for j, g in share_gain['gains'].items()}),
               block_gain=block_gain,
               dose_top_knob=dict(feature=names[J[0]], curve=dose(Fva, Fte, J[0])),
               dose_blocks=[dose(Bva, Bte, b) for b in range(L)])
    folder = OUT / dataset; folder.mkdir(parents=True, exist_ok=True)
    (folder / f'model_knobs__{variant}__seed{seed}.json').write_text(json.dumps(out, indent=1) + '\n')
    r = np.mean([x['test_AP'] for x in random_gain]) if random_gain else float('nan')
    print(f'  boundary acc {({k: round(v, 3) for k, v in acc_full.items()})} | knobs k={k_knob} {out["knob_features"][:6]} '
          f'acc {acc_k[k_knob]:.3f} random {np.mean(acc_random) if acc_random else float("nan"):.3f} | '
          f'gap share of knobs {out["share_of_knobs"]:.2f}\n'
          f'  gains {out["knob_gain"]["names"]} -> test AP {knob_gain["test_AP"]:.4f} F1 {knob_gain["test_F1"]:.4f} '
          f'FN {knob_gain["test_FN"]} | random-feature gains test AP {r:.4f} | block gains test AP '
          f'{block_gain["test_AP"]:.4f} {block_gain["gains"]}', flush=True)
    return out


# ----------------------------------------------------------------------------- B. hyper-parameter knobs
def hyper_knobs(dataset, variant, data):
    Xtr, ytr, _, Xva, yva, Xte, yte = data
    cache = {}

    def run(seed, **over):
        key = (seed, tuple(sorted(over.items())))
        if key not in cache:
            start = time.time()
            model, trees = fit(seed, data, **over)
            logit = lambda p: np.log(np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7) / np.clip(1 - p.astype(np.float64), 1e-7, 1 - 1e-7))
            sv = logit(model.predict_proba(Xva)[:, 1])
            st = logit(np.concatenate([model.predict_proba(Xte[b:b + 500_000])[:, 1] for b in range(0, len(Xte), 500_000)]))
            cache[key] = dict(setting=dict(over), seed=seed, trees=trees, seconds=round(time.time() - start, 1),
                              **summary(yva, sv, yte, st))
        return cache[key]

    seed = SEEDS[0]
    base = run(seed)
    scan, sens = {}, {}
    for name, values in SCAN.items():
        rows = [dict(value='baseline', **base)] + [dict(value=v, **run(seed, **{name: v})) for v in values]
        scan[name] = rows
        aps = [r['val_AP'] for r in rows]
        sens[name] = dict(val_AP_range=max(aps) - min(aps), best_value=rows[int(np.argmax(aps))]['value'],
                          val_AP_gain=max(aps) - base['val_AP'],
                          test_AP_range=max(r['test_AP'] for r in rows) - min(r['test_AP'] for r in rows),
                          gap_range=max(r['val_margin_gap_sd'] for r in rows) - min(r['val_margin_gap_sd'] for r in rows))
        print(f'  {dataset} {name:18s} ' + ' | '.join(f'{r["value"]}: val {r["val_AP"]:.4f} test {r["test_AP"]:.4f}' for r in rows), flush=True)
    top = sorted(sens, key=lambda k: -sens[k]['val_AP_range'])[:2]
    default = dict(scale_pos_weight='sqrt', rounds='early', **{k: BASE[k] for k in BASE if k in SCAN})
    grid = []
    for combo in itertools.product(*[(default[n],) + SCAN[n] for n in top]):
        over = {n: v for n, v in zip(top, combo) if v != default[n]}
        grid.append(dict(values=dict(zip(top, combo)), **run(seed, **over)))
    best = max(grid, key=lambda g: (g['val_AP'], not g['setting']))
    final = dict(baseline=[run(s) for s in SEEDS], tuned=[run(s, **best['setting']) for s in SEEDS])
    out = dict(dataset=dataset, variant=variant, scan_seed=seed, baseline=base, scan=scan, sensitivity=sens,
               knob_hyperparameters=top, joint_grid=grid, chosen=best['values'], three_seeds=final)
    folder = OUT / dataset; folder.mkdir(parents=True, exist_ok=True)
    (folder / f'hyper_knobs__{variant}.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    m = lambda rows, k: float(np.mean([r[k] for r in rows]))
    print(f'  {dataset} hyper knobs {top} chosen {best["values"]}: test AP {m(final["baseline"], "test_AP"):.4f} -> '
          f'{m(final["tuned"], "test_AP"):.4f}, F1 {m(final["baseline"], "test_F1"):.4f} -> {m(final["tuned"], "test_F1"):.4f}, '
          f'FN {m(final["baseline"], "test_FN"):.1f} -> {m(final["tuned"], "test_FN"):.1f}', flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--variant', default=None)
    p.add_argument('--flat', default=None)
    p.add_argument('--part', default='both', choices=['model', 'hyper', 'both'])
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    args = p.parse_args()
    if args.dataset in K.MARKET and not args.flat:
        variant = args.variant or K.MARKET[args.dataset]
        data, names = K.market_data(args.dataset, variant)
    else:
        variant = args.variant
        data, names = K.flat_data(args.flat, variant)
    if args.part in ('model', 'both'):
        for seed in map(int, args.seeds.split(',')):
            model_knobs(args.dataset, variant, seed, data, names)
    if args.part in ('hyper', 'both'):
        hyper_knobs(args.dataset, variant, data)


if __name__ == '__main__':
    main()
