"""Build flat XGBoost inputs from the Kronos_SupervisedAD splits, using that repository's own
preparation code so that train / validation / test rows are the ones its Kronos runs used.

Usage (cwd = a checkout of github.com/SansaZheng/Kronos_SupervisedAD, this repo on PYTHONPATH):
    python -m baselines.kronos_data --dataset ulb|elliptic|tstr|paysim|ibm_aml|tabformer --work SCRATCH

Variants written to SCRATCH/<dataset>/flat/
  raw      - the prepared features of the current observation only
  temporal - exactly the design of supervised/baseline.py (train-fitted standardisation, current value,
             one-step change, deviations from trailing means over 4/16/64 observations)
  summary  - (sequence task, TSTR) that file's whole-sequence summary
Rows are the prediction targets with a known label; `group` is the prepared window (or the Elliptic
time step) and is only used to build grouped folds.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--variants', default='raw,temporal')
    p.add_argument('--train-cap', type=int, default=500_000)
    p.add_argument('--validation-cap', type=int, default=100_000)
    args = p.parse_args()
    from supervised.baseline import design
    from supervised.data import Standardizer, load, prepare_builtin
    from supervised.transaction_prepare import prepare_elliptic, prepare_transactions
    work = Path(args.work) / args.dataset
    inputs, flat = work / 'inputs', work / 'flat'
    source = Path('data/supervised_financial')
    if not inputs.exists():          # same calls as scripts/run_downloaded_benchmark.py, seed 42
        work.mkdir(parents=True, exist_ok=True)
        if args.dataset in ('paysim', 'tabformer', 'ibm_aml'):
            prepare_transactions(args.dataset, source, inputs, args.train_cap, args.validation_cap, seed=42)
        elif args.dataset == 'elliptic':
            prepare_elliptic(source, inputs)
        else:
            prepare_builtin(args.dataset, source, inputs, history_length=32 if args.dataset in ('sapienza', 'ulb') else 0)
    flat.mkdir(exist_ok=True)
    train = load(inputs / 'train.npz')
    scaler = Standardizer().fit(train)
    task = train.task
    variants = ['summary'] if task == 'sequence' else args.variants.split(',')
    base = train.feature_names
    names = dict(raw=base, summary=[f'{s}({b})' for s in ('last', 'mean', 'std', 'min', 'max', 'last-first') for b in base],
                 temporal=[f'{b}{s}' for s in ('', '-lag1', '-mean4', '-mean16', '-mean64') for b in base])
    info = dict(dataset=args.dataset, task=task, variants=variants, feature_names={v: names[v] for v in variants}, splits={})
    for name in ('train', 'validation', 'test'):
        split = train if name == 'train' else load(inputs / f'{name}.npz')
        if task == 'sequence':
            x, y, _, _ = design(split, scaler)
            known = y >= 0
            np.save(flat / f'X_summary_{name}.npy', x[known].astype(np.float32))
            group = np.arange(len(y))[known]
        else:
            valid = np.arange(split.X.shape[1])[None] < split.lengths[:, None]
            if split.target_mask is not None:
                valid &= split.target_mask
            y = split.y[valid]
            known = y >= 0
            window = np.where(valid)[0]
            if args.dataset == 'elliptic':      # one node per window: group by time step instead
                window = split.X[:, 0, 0].astype(np.int64)[window]
            group = window[known]
            if 'raw' in variants:
                np.save(flat / f'X_raw_{name}.npy', split.X[valid][known].astype(np.float32))
            if 'temporal' in variants:
                x, y2, _, _ = design(split, scaler)
                assert np.array_equal(y2, y)
                np.save(flat / f'X_temporal_{name}.npy', x[known])
                del x
        np.save(flat / f'y_{name}.npy', y[known].astype(np.int8))
        np.save(flat / f'group_{name}.npy', group)
        info['splits'][name] = dict(rows=int(known.sum()), positives=int((y[known] == 1).sum()))
        print(args.dataset, name, info['splits'][name], flush=True)
        del split
    prep = inputs / 'preparation.json'
    if prep.exists():
        info['preparation'] = {k: v for k, v in json.loads(prep.read_text()).items() if k in ('protocol', 'split_policy', 'summaries', 'split_boundaries', 'excluded_features', 'limits')}
    (flat / 'info.json').write_text(json.dumps(info, indent=1) + '\n')


if __name__ == '__main__':
    main()
