"""Tables and figures for baselines.dual_xgb.  Usage: python -m baselines.dual_aggregate"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import RESULTS

OUT = RESULTS / 'dual_xgb'
ORDER = ['sapienza', 'sd-stock', 'sd-crypto', 'tstr', 'ulb', 'elliptic', 'paysim', 'ibm_aml', 'tabformer']
VARIANTS = ['base', 'ens', 'bag', 'focal', 'base+N', 'bag+N', 'nov']


def main():
    runs = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/dual__*.json'))]
    rows = []
    for r in runs:
        b = r['results']['base']
        for v in VARIANTS:
            x = r['results'].get(v)
            if x is None: continue
            rows.append(dict(dataset=r['dataset'], seed=r['seed'], variant=v, val_AP=x['val_AP'], test_AP=x['test_AP'],
                             test_ROC_AUC=x['test_ROC_AUC'], F1=x['test_F1'], precision=x['test_precision'], recall=x['test_recall'],
                             FN=x['test_FN'], FP=x['test_FP'], dAP=x['test_AP'] - b['test_AP'], dF1=x['test_F1'] - b['test_F1'],
                             dvalAP=x['val_AP'] - b['val_AP'], dFN=x['test_FN'] - b['test_FN'], dFP=x['test_FP'] - b['test_FP'],
                             recovered=x.get('missed_by_base_recovered', 0), lost=x.get('caught_by_base_lost', 0),
                             recall_at_budget=x.get('recall_at_base_alert_budget'),
                             trees=np.mean(x['trees']) if 'trees' in x else np.nan, seconds=r['seconds'].get(v, r['seconds'].get('novelty'))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'runs.csv', index=False)
    g = df.groupby(['dataset', 'variant'])
    s = g[['val_AP', 'test_AP', 'F1', 'precision', 'recall', 'FN', 'FP', 'dAP', 'dF1', 'dvalAP', 'dFN', 'dFP', 'recovered', 'lost',
           'recall_at_budget', 'trees', 'seconds', 'test_ROC_AUC']].mean()
    s = s.join(g[['test_AP', 'dAP', 'F1']].std().add_suffix('_sd')).join(g.dAP.apply(lambda x: int((x > 0).sum())).rename('seeds_up')).join(
        g.dAP.apply(lambda x: int((x < 0).sum())).rename('seeds_down')).reset_index()
    datasets = [d for d in ORDER if d in set(s.dataset)]
    s['dataset'] = pd.Categorical(s.dataset, datasets); s['variant'] = pd.Categorical(s.variant, VARIANTS)
    s = s.sort_values(['dataset', 'variant']).reset_index(drop=True)
    s.to_csv(OUT / 'summary.csv', index=False)
    sup = s[s.variant != 'nov']
    piv = lambda col: sup.pivot(index='dataset', columns='variant', values=col).reindex(datasets)[[v for v in VARIANTS if v != 'nov']]
    rank = piv('test_AP').rank(axis=1, ascending=False)
    overall = pd.DataFrame(dict(mean_dAP=piv('dAP').mean(), median_dAP=piv('dAP').median(), mean_rank=rank.mean(),
                                tasks_best=(rank == 1).sum(), tasks_up_all_seeds=(piv('seeds_up') == 3).sum(),
                                tasks_down_all_seeds=(piv('seeds_down') == 3).sum(), mean_seed_sd=piv('test_AP_sd').mean(),
                                mean_dF1=piv('dF1').mean(), mean_seconds=piv('seconds').mean()))
    overall.to_csv(OUT / 'overall.csv')
    nov = [dict(dataset=r['dataset'], seed=r['seed'], groups=r['feature_groups'], features=r['features'],
                test_ROC_AUC=r['results']['nov']['test_ROC_AUC'], test_AP=r['results']['nov']['test_AP'],
                prevalence=(r['results']['base']['test_TP'] + r['results']['base']['test_FN'])) for r in runs]
    nov = pd.DataFrame(nov).groupby('dataset')[['groups', 'features', 'test_ROC_AUC', 'test_AP']].mean().reindex(datasets)
    nov.to_csv(OUT / 'novelty_alone.csv')

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
    for col in ('test_AP', 'test_AP_sd', 'dAP', 'dAP_sd', 'seeds_up', 'seeds_down', 'F1', 'FN', 'FP', 'recovered', 'lost', 'recall_at_budget', 'trees', 'seconds'):
        print(col); print(piv(col).round(4).to_string())
    print(overall.round(4).to_string()); print(nov.round(4).to_string())

    shown = [v for v in VARIANTS if v not in ('base', 'nov')]
    colours = dict(zip(shown, ['#7f7f7f', '#2c7fb8', '#e6a23c', '#55a868', '#c0392b']))
    fig, axes = plt.subplots(2, 5, figsize=(21, 7), squeeze=False)
    for ax in axes.ravel()[len(datasets):]: ax.axis('off')
    for i, ds in enumerate(datasets):
        ax = axes.ravel()[i]; part = s[s.dataset == ds].set_index('variant')
        ax.bar(shown, [part.loc[v, 'dAP'] for v in shown], yerr=[part.loc[v, 'dAP_sd'] for v in shown], capsize=3,
               color=[colours[v] for v in shown])
        ax.axhline(0, color='k', lw=0.6); ax.set_title(f'{ds}  (baseline AP {part.loc["base", "test_AP"]:.3f})', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Change in test AP against baseline XGBoost (mean and sd of the per-seed difference, three seeds)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'delta_ap.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
