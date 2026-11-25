"""Tables and figures for baselines.intercept_xgb / intercept_probe.  Usage: python -m baselines.intercept_aggregate"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import RESULTS

OUT = RESULTS / 'intercept_xgb'
ORDER = ['sapienza', 'sd-stock', 'sd-crypto', 'tstr', 'ulb', 'elliptic', 'paysim', 'ibm_aml', 'tabformer']
V = ['prior|sqrt', 'prior|1', 'half|sqrt', 'half|1', 'prior|1|mcw0', 'prior|sqrt|mcw0', 'half|1|ens']


def main():
    rows, prev = [], {}
    for p in sorted(OUT.glob('*/intercept__*.json')):
        r = json.loads(p.read_text()); b = r['results']['prior|sqrt']; prev[r['dataset']] = r['prevalence']
        for k, v in r['results'].items():
            rows.append(dict(dataset=r['dataset'], seed=r['seed'], variant=k, val_AP=v['val_AP'], test_AP=v['test_AP'], F1=v['test_F1'],
                             precision=v['test_precision'], recall=v['test_recall'], FN=v['test_FN'], FP=v['test_FP'], trees=v['trees'],
                             dAP=v['test_AP'] - b['test_AP'], dvalAP=v['val_AP'] - b['val_AP'], dF1=v['test_F1'] - b['test_F1'],
                             recovered=v.get('missed_by_base_recovered', 0), lost=v.get('caught_by_base_lost', 0)))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'runs.csv', index=False)
    g = df.groupby(['dataset', 'variant'])
    s = g[['val_AP', 'test_AP', 'F1', 'precision', 'recall', 'FN', 'FP', 'trees', 'dAP', 'dvalAP', 'dF1', 'recovered', 'lost']].mean().join(
        g[['test_AP', 'dAP']].std().add_suffix('_sd')).join(g.dAP.apply(lambda x: int((x > 0).sum())).rename('seeds_up')).reset_index()
    s['train_prevalence'] = s.dataset.map(prev)
    s.to_csv(OUT / 'summary.csv', index=False)
    piv = lambda c: s.pivot(index='dataset', columns='variant', values=c).reindex(ORDER)[V]
    overall = pd.DataFrame(dict(mean_dAP=piv('dAP').mean(), median_dAP=piv('dAP').median(), worst_dAP=piv('dAP').min(),
                                tasks_up_all_seeds=(piv('seeds_up') == 3).sum(), mean_seed_sd=piv('test_AP_sd').mean(), mean_dF1=piv('dF1').mean()))
    overall.to_csv(OUT / 'overall.csv')
    probe = pd.DataFrame([dict({k: v for k, v in r.items() if k != 'staged'}) for p in sorted(OUT.glob('*/probe__*.json')) for r in json.loads(p.read_text())])
    pm = probe.groupby(['dataset', 'name'])[['trees_kept', 'test_AP', 'precision_top100', 'precision_topP', 'recall_fpr_0p1pct', 'recall_fpr_1pct',
                                             'max_abs_leaf_first10']].mean().reset_index()
    pm.to_csv(OUT / 'probe_summary.csv', index=False)
    staged = pd.DataFrame([dict(dataset=r['dataset'], seed=r['seed'], name=r['name'], **x) for p in sorted(OUT.glob('*/probe__*.json'))
                           for r in json.loads(p.read_text()) for x in r['staged']])
    sm = staged.groupby(['dataset', 'name', 'trees'])[['val_AP', 'test_AP']].agg(['mean', 'count'])
    sm.columns = [f'{a}_{b}' for a, b in sm.columns]; sm = sm.reset_index(); sm.to_csv(OUT / 'staged.csv', index=False)

    pd.set_option('display.width', 230); pd.set_option('display.max_columns', 40)
    print(overall.round(4).to_string())
    for c in ('test_AP', 'dAP', 'dAP_sd', 'seeds_up', 'dvalAP', 'recovered', 'lost'):
        print(c); print(piv(c).round(4).to_string())
    for c in ('test_AP', 'recall_fpr_0p1pct', 'recall_fpr_1pct', 'precision_topP', 'precision_top100', 'trees_kept', 'max_abs_leaf_first10'):
        print(c); print(pm.pivot(index='dataset', columns='name', values=c).reindex(ORDER).round(4).to_string())
    print(sm[sm.dataset.isin(['ibm_aml', 'paysim', 'tabformer'])].pivot_table(index=['dataset', 'name'], columns='trees', values='test_AP_mean').round(3).to_string())

    shown = ['prior|1', 'half|sqrt', 'half|1', 'half|1|ens']
    colours = ['#2c7fb8', '#e6a23c', '#c0392b', '#55a868']
    fig, axes = plt.subplots(2, 5, figsize=(21, 7), squeeze=False)
    for ax in axes.ravel()[len(ORDER):]: ax.axis('off')
    for i, ds in enumerate(ORDER):
        ax = axes.ravel()[i]; part = s[s.dataset == ds].set_index('variant')
        ax.bar(shown, [part.loc[v, 'dAP'] for v in shown], yerr=[part.loc[v, 'dAP_sd'] for v in shown], capsize=3, color=colours)
        ax.axhline(0, color='k', lw=0.6); ax.tick_params(axis='x', labelsize=8)
        ax.set_title(f'{ds}  (baseline AP {part.loc["prior|sqrt", "test_AP"]:.3f}, anomalies {100 * prev[ds]:.2f} %)', fontsize=9)
    fig.suptitle('Removing the class weight ("|1") and the fitted prior ("half"): change in test AP against baseline XGBoost\n'
                 '(mean and sd of the per-seed difference, three seeds)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'delta_ap.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, ds in zip(axes, ('ibm_aml', 'paysim', 'tabformer')):
        for name, c in (('prior|1', '#2c7fb8'), ('half|1', '#c0392b'), ('prior|1|mds1', '#7f7f7f')):
            part = sm[(sm.dataset == ds) & (sm.name == name) & (sm.test_AP_count == 3)].sort_values('trees')
            ax.plot(part.trees, part.test_AP_mean, marker='o', ms=4, color=c, label=name)
        ax.set_xscale('log'); ax.set_xlabel('boosting rounds used'); ax.set_ylabel('test AP'); ax.set_title(ds, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle('Test AP along boosting (mean of three seeds; only rounds every seed reached)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'staged_ap.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
