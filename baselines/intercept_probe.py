"""Why the starting score matters: mechanism probes for baselines.intercept_xgb.

Usage: python -m baselines.intercept_probe --dataset NAME [--variant V --flat FOLDER]

For every seed and for   prior|1 (fitted prior, no class weight),  half|1 (base_score 0.5, no class weight),
prior|1|mds1 and prior|1|mds0.3 (fitted prior with the Newton step capped by max_delta_step):
  - test / validation AP after 1, 2, 5, 10, 20, 40, 80, ... trees (where along boosting the difference appears)
  - where in the ranking the difference sits: precision in the top 100 / top P (P = number of anomalies),
    recall at 0.1 % and 1 % false-positive rate
  - size of the leaf values in the first 10 trees (largest absolute leaf value)
"""
import argparse
import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from . import knob_xgb as K
from .common import RESULTS, SEEDS

OUT = RESULTS / 'intercept_xgb'
VARIANTS = dict([('prior|1', dict()), ('half|1', dict(base_score=0.5)), ('prior|1|mds1', dict(max_delta_step=1.0)),
                 ('prior|1|mds0.3', dict(max_delta_step=0.3))])
STAGES = (1, 2, 5, 10, 20, 40, 80, 160, 320, 600)


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
    dva = xgb.DMatrix(Xva)
    chunks = [xgb.DMatrix(Xte[i:i + 500_000]) for i in range(0, len(Xte), 500_000)] if len(Xte) <= 1_000_000 else None
    out = []
    for seed in SEEDS:
        for name, over in VARIANTS.items():
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=1, scale_pos_weight=1.0, tree_method='hist', eval_metric='aucpr',
                                  early_stopping_rounds=50, random_state=seed, n_jobs=2, **over)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            b, kept = m.get_booster(), int(m.best_iteration + 1)
            built = b.num_boosted_rounds()

            def test_margin(t):
                parts = chunks or (xgb.DMatrix(Xte[i:i + 500_000]) for i in range(0, len(Xte), 500_000))
                return np.concatenate([b.predict(d, output_margin=True, iteration_range=(0, t)) for d in parts]).astype(np.float64)
            staged = []
            for t in [s for s in STAGES if s <= built]:
                staged.append(dict(trees=t, val_AP=float(average_precision_score(yva, b.predict(dva, output_margin=True, iteration_range=(0, t)))),
                                   test_AP=float(average_precision_score(yte, test_margin(t)))))
            s = test_margin(kept)
            order = np.argsort(-s, kind='stable'); P = int(yte.sum()); neg = np.sort(s[yte == 0])
            frame = b.trees_to_dataframe(); leaf = frame[(frame.Feature == 'Leaf') & (frame.Tree < 10)]
            r = dict(dataset=args.dataset, seed=seed, name=name, trees_kept=kept, test_AP=float(average_precision_score(yte, s)),
                     precision_top100=float(yte[order[:100]].mean()), precision_topP=float(yte[order[:P]].mean()),
                     recall_fpr_0p1pct=float((s[yte == 1] > neg[int(len(neg) * 0.999) - 1]).mean()),
                     recall_fpr_1pct=float((s[yte == 1] > neg[int(len(neg) * 0.99) - 1]).mean()),
                     max_abs_leaf_first10=float(leaf.Gain.abs().max()), mean_abs_leaf_first10=float(leaf.Gain.abs().mean()), staged=staged)
            out.append(r)
            print(f'[{args.dataset} seed={seed}] {name}: kept {kept} AP {r["test_AP"]:.4f} P@100 {r["precision_top100"]:.2f} P@P '
                  f'{r["precision_topP"]:.3f} R@0.1% {r["recall_fpr_0p1pct"]:.3f} R@1% {r["recall_fpr_1pct"]:.3f} max leaf '
                  f'{r["max_abs_leaf_first10"]:.3f} | staged test AP ' + ' '.join(f'{x["trees"]}:{x["test_AP"]:.3f}' for x in staged), flush=True)
    (OUT / args.dataset).mkdir(parents=True, exist_ok=True)
    (OUT / args.dataset / f'probe__{variant}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
