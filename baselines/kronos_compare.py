"""Controls on the Kronos_SupervisedAD splits: XGBoost per feature variant, and the gradient-boosting
control of that repository (supervised/baseline.py settings) on the identical design.

Usage: python -m baselines.kronos_compare --dataset NAME --flat SCRATCH/NAME/flat [--hgb]
(--hgb needs a Kronos_SupervisedAD checkout on PYTHONPATH for its threshold rule and metrics.)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import RESULTS, SEEDS, metrics, pick_threshold
from .knob_xgb import flat_data, xgb_model

OUT = RESULTS / 'kronos_compare'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--flat', required=True)
    p.add_argument('--hgb', action='store_true')
    args = p.parse_args()
    info = json.loads((Path(args.flat) / 'info.json').read_text())
    out = dict(dataset=args.dataset, splits=info['splits'], preparation=info.get('preparation'), xgb={}, hgb={})
    for variant in info['variants']:
        (Xtr, ytr, _, Xva, yva, Xte, yte), names = flat_data(args.flat, variant)
        pw = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1)))
        for seed in SEEDS:
            m = xgb_model(seed, pw).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            sv = m.predict_proba(Xva)[:, 1]
            st = np.concatenate([m.predict_proba(Xte[b:b + 500_000])[:, 1] for b in range(0, len(Xte), 500_000)])
            thr = pick_threshold(yva, sv)
            out['xgb'][f'{variant}__seed{seed}'] = dict(features=len(names), trees=int(m.best_iteration + 1),
                                                        validation=metrics(yva, sv, thr), test=metrics(yte, st, thr))
            print(args.dataset, 'xgb', variant, seed, 'val AP %.4f test AP %.4f F1 %.4f' % (
                out['xgb'][f'{variant}__seed{seed}']['validation']['AP'], out['xgb'][f'{variant}__seed{seed}']['test']['AP'],
                out['xgb'][f'{variant}__seed{seed}']['test']['F1']), flush=True)
    if args.hgb:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from supervised.metrics import evaluate, select_operating_point
        variant = 'summary' if info['task'] == 'sequence' else 'temporal'
        if variant in info['variants']:
            (Xtr, ytr, _, Xva, yva, Xte, yte), _ = flat_data(args.flat, variant)
            for seed in SEEDS:
                m = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1.,
                                                   class_weight='balanced', early_stopping=False, random_state=seed)
                m.fit(Xtr, ytr)
                sv = m.predict_proba(Xva)[:, 1]
                st = np.concatenate([m.predict_proba(Xte[b:b + 500_000])[:, 1] for b in range(0, len(Xte), 500_000)])
                op = select_operating_point(yva, sv, policy='f1', max_fpr=.001, min_precision=.8, beta=1.)
                out['hgb'][f'{variant}__seed{seed}'] = dict(validation=evaluate(yva, sv, op['threshold']),
                                                            test=evaluate(yte, st, op['threshold']))
                t = out['hgb'][f'{variant}__seed{seed}']['test']
                print(args.dataset, 'hgb', variant, seed, {k: round(v, 4) for k, v in t.items() if isinstance(v, float)}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{args.dataset}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
