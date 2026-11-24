"""Tables and figures for baselines.knob_hyper_focus.  Usage: python -m baselines.knob_hyper_aggregate"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import RESULTS
from .knob_hyper_focus import BASE, GRID

OUT = RESULTS / 'knob_hyper'
ORDER = ['sapienza', 'sd-stock', 'sd-crypto', 'tstr', 'ulb', 'elliptic', 'paysim', 'ibm_aml', 'tabformer']
COLS = ['val_AP', 'test_AP', 'test_ROC_AUC', 'test_F1', 'test_precision', 'test_recall', 'test_FN', 'test_FP', 'trees',
        'd_prime', 'recall_at_fpr_0p1pct', 'recall_at_k_positives', 'anomaly_margin_mean', 'normal_margin_mean',
        'top_gap_share', 'effective_features', 'first_half_tree_share', 'scale_pos_weight']


def main():
    frames, meta = [], {}
    for p in sorted(OUT.glob('*__*.json')):
        d = json.loads(p.read_text())
        meta[d['dataset']] = {k: d[k] for k in ('large', 'train_rows', 'train_positives', 'validation_positives', 'test_positives')}
        f = pd.DataFrame([{k: v for k, v in r.items() if k != 'setting'} for r in d['records']])
        f['dataset'] = d['dataset']; frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / 'runs.csv', index=False)
    datasets = [d for d in ORDER if d in set(df.dataset)]
    g = df.groupby(['dataset', 'knob', 'value'])
    curves = g[COLS].mean().join(g[['val_AP', 'test_AP', 'test_F1', 'test_FN', 'test_FP', 'trees']].std().add_suffix('_sd')).join(
        g.size().rename('seeds')).reset_index()
    curves.to_csv(OUT / 'curves.csv', index=False)

    rows = []
    for ds in datasets:
        for knob in GRID:
            c = curves[(curves.dataset == ds) & (curves.knob == knob)].sort_values('value')
            base = c[c.value == BASE[knob]].iloc[0]
            vbest, tbest = c.loc[c.val_AP.idxmax()], c.loc[c.test_AP.idxmax()]
            runs = df[(df.dataset == ds) & (df.knob == knob)]
            pair = runs[runs.value == vbest.value].set_index('seed').test_AP - runs[runs.value == BASE[knob]].set_index('seed').test_AP
            pair = pair.dropna()
            rho = spearmanr(c.val_AP, c.test_AP).statistic if c.test_AP.nunique() > 1 and c.val_AP.nunique() > 1 else np.nan
            rows.append(dict(
                dataset=ds, knob=knob, seeds=int(c.seeds.min()), baseline=BASE[knob],
                val_AP_range=c.val_AP.max() - c.val_AP.min(), test_AP_range=c.test_AP.max() - c.test_AP.min(),
                seed_sd_at_baseline=base.test_AP_sd, val_best=vbest.value, test_best=tbest.value,
                base_test_AP=base.test_AP, val_best_test_AP=vbest.test_AP, test_best_test_AP=tbest.test_AP,
                gain_val_chosen=vbest.test_AP - base.test_AP, paired_sd=pair.std(ddof=1) if len(pair) > 1 else np.nan,
                seeds_improved=int((pair > 0).sum()), seeds_compared=len(pair),
                regret=tbest.test_AP - vbest.test_AP, best_possible_gain=tbest.test_AP - base.test_AP,
                val_test_rank_corr=rho,
                base_F1=base.test_F1, val_best_F1=vbest.test_F1, base_FN=base.test_FN, val_best_FN=vbest.test_FN,
                base_FP=base.test_FP, val_best_FP=vbest.test_FP, base_trees=base.trees, val_best_trees=vbest.trees))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'summary.csv', index=False)

    spw = curves[curves.knob == 'spw_alpha']
    at = lambda ds, a, col: float(spw[(spw.dataset == ds) & (spw.value == a)][col].iloc[0])
    trend = pd.DataFrame([dict(dataset=ds, **meta[ds], **{f'{col}@{a:g}': at(ds, a, col) for col in
                               ('scale_pos_weight', 'trees', 'test_AP', 'test_precision', 'test_recall', 'test_FN', 'test_FP',
                                'd_prime', 'recall_at_fpr_0p1pct', 'top_gap_share', 'effective_features', 'first_half_tree_share',
                                'anomaly_margin_mean', 'normal_margin_mean')
                               for a in (0.0, 0.5, 1.0, 1.25)}) for ds in datasets])
    trend.to_csv(OUT / 'spw_trend.csv', index=False)

    pd.set_option('display.width', 280); pd.set_option('display.max_columns', 80)
    show = ['dataset', 'knob', 'seeds', 'val_AP_range', 'test_AP_range', 'seed_sd_at_baseline', 'val_best', 'test_best', 'base_test_AP',
            'gain_val_chosen', 'paired_sd', 'seeds_improved', 'best_possible_gain', 'regret', 'val_test_rank_corr',
            'base_FN', 'val_best_FN', 'base_FP', 'val_best_FP', 'base_trees', 'val_best_trees']
    print(summary[show].round(4).to_string())
    for col in ('test_AP', 'test_recall', 'test_precision', 'test_FN', 'test_FP', 'trees', 'd_prime', 'recall_at_fpr_0p1pct',
                'top_gap_share', 'effective_features', 'first_half_tree_share', 'scale_pos_weight'):
        print(col); print(spw.pivot(index='dataset', columns='value', values=col).reindex(datasets).round(4).to_string())
    for knob in ('max_depth', 'min_child_weight', 'reg_lambda'):
        c = curves[curves.knob == knob]
        for col in ('test_AP', 'trees', 'test_FN', 'test_FP', 'top_gap_share'):
            print(knob, col); print(c.pivot(index='dataset', columns='value', values=col).reindex(datasets).round(4).to_string())

    ncols = 5; nrows = -(-len(datasets) // ncols)
    labels = dict(spw_alpha='alpha  (scale_pos_weight = (neg/pos)^alpha)', max_depth='max_depth',
                  min_child_weight='min_child_weight', reg_lambda='reg_lambda')
    for knob in GRID:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), squeeze=False)
        for ax in axes.ravel()[len(datasets):]: ax.axis('off')
        for i, ds in enumerate(datasets):
            ax = axes.ravel()[i]
            c = curves[(curves.dataset == ds) & (curves.knob == knob)].sort_values('value')
            x = np.arange(len(c)); b = float(c[c.value == BASE[knob]].test_AP.iloc[0]); bv = float(c[c.value == BASE[knob]].val_AP.iloc[0])
            ax.errorbar(x, c.test_AP - b, yerr=c.test_AP_sd.fillna(0), marker='o', ms=4, lw=1.5, capsize=2, color='#c0392b', label='test AP')
            ax.plot(x, c.val_AP - bv, marker='s', ms=3, lw=1.1, ls='--', color='#2c7fb8', label='validation AP')
            ax.axhline(0, color='k', lw=0.6); ax.axvline(list(c.value).index(BASE[knob]), color='grey', lw=0.8, ls=':')
            ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in c.value], fontsize=8)
            ax.set_title(f'{ds}  (seeds: {int(c.seeds.min())})', fontsize=10)
            if i == 0: ax.legend(fontsize=7)
        fig.suptitle(f'Hyper-parameter knob {labels[knob]}: change in AP against the baseline value (dotted line); mean and sd over seeds', fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / f'dose_{knob}.png', dpi=130); plt.close(fig)

    for stem, left, right, ll, rl in (('spw_tradeoff', 'test_recall', 'test_precision', 'recall', 'precision'),
                                      ('spw_internal', 'trees', 'top_gap_share', 'trees kept by early stopping', 'gap share of the largest feature')):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), squeeze=False)
        for ax in axes.ravel()[len(datasets):]: ax.axis('off')
        for i, ds in enumerate(datasets):
            ax = axes.ravel()[i]; c = spw[spw.dataset == ds].sort_values('value'); x = np.arange(len(c))
            ax.plot(x, c[left], marker='o', ms=4, color='#c0392b'); ax.set_ylabel(ll, color='#c0392b', fontsize=8)
            ax2 = ax.twinx(); ax2.plot(x, c[right], marker='s', ms=3, ls='--', color='#2c7fb8'); ax2.set_ylabel(rl, color='#2c7fb8', fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in c.value], fontsize=8); ax.set_title(ds, fontsize=10)
            ax.axvline(list(c.value).index(0.5), color='grey', lw=0.8, ls=':')
        fig.suptitle(f'Turning scale_pos_weight (alpha): {ll} (red) and {rl} (blue, dashed); mean over three seeds', fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / f'{stem}.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
