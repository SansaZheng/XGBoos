"""Rewrite the task partitions of SansaZheng/Financial_unsuperviesdAD in the flat layout used by baselines.auto_xgb.

Usage: python -m baselines.finunsup_data --tabular CACHE --graphs GRAPHS --output FOLDER

CACHE  is written by that repository's  scripts/prepare_repo_data.py  (nine transaction / point tasks),
GRAPHS by its  python -m finad_zero.graph_data  (two account tasks). Nothing is re-split: train, calibration and
test are that repository's partitions; its calibration part is our validation part. Rows whose label is unknown
(-1, S-FFSD) are dropped from all three parts, so the test rows are exactly the rows that repository evaluates.

Account tasks follow its comparator definition: paying accounts only (role flag of the first snapshot), features =
the four snapshots side by side (4 x 10 values).
"""
import argparse
import json
from pathlib import Path

import numpy as np

TABULAR = ('ulb', 'banksim', 's_ffsd', 'sapienza_5S', 'sapienza_15S', 'sapienza_25S', 'sd_stock', 'sd_crypto', 'sd_synth')
GRAPH = ('banksim', 's_ffsd')
PART = {'train': 'train', 'calibration': 'validation', 'test': 'test'}


def write(out, name, parts, names, note):
    out = out / name
    out.mkdir(parents=True, exist_ok=True)
    info = dict(dataset=name, variants=['raw'], feature_names={'raw': list(names)}, note=note, splits={})
    for s, (X, y) in parts.items():
        keep = y >= 0
        X, y = np.ascontiguousarray(X[keep], dtype=np.float32), y[keep].astype(np.int8)
        np.save(out / f'X_raw_{PART[s]}.npy', X); np.save(out / f'y_{PART[s]}.npy', y)
        info['splits'][PART[s]] = dict(rows=int(len(y)), positive=int(y.sum()), unknown_dropped=int((~keep).sum()))
    np.save(out / 'group_train.npy', np.arange(info['splits']['train']['rows']))
    (out / 'info.json').write_text(json.dumps(info, indent=1) + '\n')
    print(name, info['splits'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tabular', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for name in TABULAR:
        m = json.loads((a.tabular / name / 'manifest.json').read_text())
        f = m['features']
        parts = {}
        for s in PART:
            n = m['splits'][s]['rows']
            parts[s] = (np.memmap(a.tabular / name / f'{s}.x.f32', mode='r', dtype=np.float32, shape=(n, len(f))),
                        np.fromfile(a.tabular / name / f'{s}.y.i8', dtype=np.int8))
        write(a.output, f'fu_{name}', parts, f, 'transaction / point track')
    for name in GRAPH:
        parts = {}
        for s in PART:
            with np.load(a.graphs / name / f'{s}.graph.npz', allow_pickle=False) as z:
                x = z['x']
            payer = x[0, :, 0] > .5
            flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
            parts[s] = (flat[payer], np.load(a.graphs / name / f'{s}.y.npy')[payer])
        names = [f'snap{t}_f{j}' for t in range(4) for j in range(x.shape[2])]
        write(a.output, f'fu_graph_{name}', parts, names, 'account track: paying accounts, four snapshots side by side')


if __name__ == '__main__':
    main()
