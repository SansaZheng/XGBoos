"""Tree-ensemble baselines for Sapienza 15S and the two SD-FMM point-label tasks.

Usage: python -m baselines.market --dataset sapienza|sd-stock|sd-crypto [--methods rf,ada,xgb,lgbm,position]

Splits are the Kronos_SupervisedAD research splits (coin-isolated for Sapienza,
content-grouped for SD-FMM).  Feature sets:
  raw    - the publisher's features of the current block only
  causal - raw plus changes and trailing-window statistics of the same sequence
           (current and earlier blocks only; no absolute position, no future values)
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .common import ROOT, SEEDS, record

warnings.filterwarnings("ignore")
DATA = ROOT / 'data'
SAPIENZA_FEATURES = ['std_rush_order', 'avg_rush_order', 'std_trades', 'std_volume', 'avg_volume',
                     'std_price', 'avg_price', 'avg_price_max', 'hour_sin', 'hour_cos',
                     'minute_sin', 'minute_cos']


def load(dataset, split):
    """Return a list of (X[T,F], y[T]) sequences in time order."""
    if dataset == 'sapienza':
        groups = pd.read_csv(DATA / 'sapienza/sapienza_groups_15S.csv')
        keep = set(groups[groups.split == split].pump_index)
        frame = pd.read_csv(DATA / 'sapienza/features_15S.csv.gz')
        frame = frame[frame.pump_index.isin(keep)]
        out = []
        for _, event in frame.groupby('pump_index', sort=True):
            event = event.sort_values('date', kind='stable')
            gap = np.log1p(pd.to_datetime(event.date).diff().dt.total_seconds().fillna(0).to_numpy())
            out.append((np.column_stack([event[SAPIENZA_FEATURES].to_numpy(np.float64), gap]),
                        event['gt'].astype(int).to_numpy()))
        return out
    market = dataset.removeprefix('sd-')
    folder, suffix = ('stock', '_bs') if market == 'stock' else ('crypto', '')
    manifest = pd.read_csv(DATA / 'sd_fmm/sd_fmm_sequences.csv')
    rows = manifest[(manifest.market == market) & (manifest.split == split)].sequence_index.to_numpy()
    x = np.load(DATA / f'sd_fmm/{folder}/test_data{suffix}.npy')
    y = (np.load(DATA / f'sd_fmm/{folder}/test_label{suffix}.npy') > 0).astype(int)
    return [(x[i].astype(np.float64), y[i]) for i in rows]


def causal(x):
    """Current value, changes and trailing statistics; row t reads rows <= t only."""
    t = len(x)
    prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    prefix2 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x * x, axis=0)])
    ends = np.arange(1, t + 1)
    out = [x]
    for lag in (1, 4):
        out.append(x - x[np.maximum(np.arange(t) - lag, 0)])
    for width in (4, 16, 64):
        starts = np.maximum(ends - width, 0)
        count = (ends - starts)[:, None]
        mean = (prefix[ends] - prefix[starts]) / count
        out.append(x - mean)
        if width == 16:
            out.append(np.sqrt(np.maximum((prefix2[ends] - prefix2[starts]) / count - mean ** 2, 0)))
    return np.hstack(out)


def design(sequences, variant):
    X = np.vstack([s[0] if variant == 'raw' else causal(s[0]) for s in sequences]).astype(np.float32)
    y = np.concatenate([s[1] for s in sequences])
    position = np.concatenate([np.arange(len(s[1])) for s in sequences])
    return X, y, position


def fit_predict(method, seed, Xtr, ytr, Xva, yva, Xte):
    pos_weight = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1)))
    if method == 'rf':      # La Morgia et al. 2020 classifier.py settings
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed, n_jobs=2)
        model.fit(Xtr, ytr)
    elif method == 'ada':   # La Morgia et al. 2023 AdaBoost settings (classifier.py, commented line)
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=100,
                                   learning_rate=0.5, random_state=seed)
        model.fit(Xtr, ytr)
    elif method == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=1, scale_pos_weight=pos_weight,
                                  tree_method='hist', eval_metric='aucpr', early_stopping_rounds=50,
                                  random_state=seed, n_jobs=2)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif method == 'lgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
                                   scale_pos_weight=pos_weight, random_state=seed, n_jobs=2, verbose=-1)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='average_precision',
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        raise ValueError(method)
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['sapienza', 'sd-stock', 'sd-crypto'])
    p.add_argument('--methods', default='position,rf,ada,xgb,lgbm')
    args = p.parse_args()
    splits = {s: load(args.dataset, s) for s in ('train', 'validation', 'test')}
    for method in args.methods.split(','):
        if method == 'position':
            # Control: label frequency by index inside the sequence, learnt on train only.
            _, ytr, ptr = design(splits['train'], 'raw')
            _, yva, pva = design(splits['validation'], 'raw')
            _, yte, pte = design(splits['test'], 'raw')
            size = int(max(ptr.max(), pva.max(), pte.max())) + 1
            rate = np.bincount(ptr, weights=ytr, minlength=size) / np.maximum(np.bincount(ptr, minlength=size), 1)
            record(args.dataset, 'position-only', 'control', 0, yva, rate[pva], yte, rate[pte],
                   extra=dict(note='train label frequency by index within the sequence; uses no market feature'))
            continue
        for variant in ('raw', 'causal'):
            Xtr, ytr, _ = design(splits['train'], variant)
            Xva, yva, _ = design(splits['validation'], variant)
            Xte, yte, _ = design(splits['test'], variant)
            for seed in SEEDS:
                sva, ste = fit_predict(method, seed, Xtr, ytr, Xva, yva, Xte)
                record(args.dataset, method, variant, seed, yva, sva, yte, ste,
                       extra=dict(features=int(Xtr.shape[1]), train_rows=int(len(ytr)),
                                  train_positives=int(ytr.sum())))


if __name__ == '__main__':
    main()
