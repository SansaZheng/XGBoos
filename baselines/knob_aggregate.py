"""Aggregate results/knob/*/knob_xgb__*.json into tables and figures; re-derive headline metrics
from the saved test scores as an independent check.

Usage: python -m baselines.knob_aggregate
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .common import RESULTS

OUT = RESULTS / 'knob'
FAMILIES = ['steer-normal', 'steer-tangent', 'steer-random', 'gate-knob', 'gate-random-knob',
            'reweight-knob', 'reweight-random-knob', 'reweight-margin']


def main():
    runs = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/knob_xgb__*.json'))]
    rows, fam, probe, sweep, audit = [], [], [], [], []
    for r in runs:
        key = dict(dataset=r['dataset'], variant=r['variant'], seed=r['seed'])
        b, k, s = r['baseline']['test'], r['knob_xgb']['test'], r['subtle_anomalies']
        rows.append(dict(**key, chosen=r['chosen'], val_AP_base=r['baseline']['validation']['AP'],
                         val_AP_knob=r['knob_xgb']['validation']['AP'], AP_base=b['AP'], AP_knob=k['AP'],
                         F1_base=b['F1'], F1_knob=k['F1'], recall_base=b['recall'], recall_knob=k['recall'],
                         FN_base=b['FN'], FN_knob=k['FN'], FP_base=b['FP'], FP_knob=k['FP'],
                         missed_recovered=s['missed_recovered_by_knob'], caught_lost=s['caught_lost_by_knob'],
                         recall_at_budget_base=s['recall_at_budget_baseline'],
                         recall_at_budget_knob=s['recall_at_budget_knob']))
        for name in FAMILIES:
            f = r['family_best_on_validation'].get(name)
            if f:
                fam.append(dict(**key, family=name, config=f['config'],
                                d_val_AP=f['validation']['AP'] - r['baseline']['validation']['AP'],
                                d_test_AP=f['test']['AP'] - b['AP'], d_test_F1=f['test']['F1'] - b['F1'],
                                d_test_FN=f['test']['FN'] - b['FN'], d_test_FP=f['test']['FP'] - b['FP']))
        acc, kn, ho = r['boundary_cv_balanced_accuracy'], r['knobs'], r['kan']['held_out']['test']
        probe.append(dict(**key, trees=r['trees'], d=r['features'], **{f'n_{c}': v for c, v in r['probe_set'].items()},
                          acc_full_cnn=acc['full']['cnn'], acc_full_mlp=acc['full']['mlp'], acc_full_lr=acc['full']['lr'],
                          acc_margin_only=max(acc['margin_only'].values()), k_knob=kn['k'],
                          knobs=' + '.join(kn['names']), acc_knob=kn['acc_top_k'][str(kn['k'])] if isinstance(
                              next(iter(kn['acc_top_k'])), str) else kn['acc_top_k'][kn['k']],
                          acc_random_mean=kn['acc_random_mean'], acc_random_max=kn['acc_random_max'],
                          late_ratio=kn['late_ratio'], test_errors=ho['errors'],
                          test_bal_acc_knob=ho['knob_bal_acc'], test_bal_acc_random_knob=ho['random_knob_bal_acc'],
                          test_bal_acc_margin_only=ho['margin_only_bal_acc']))
        for e in r['intervention_sweep']:
            sweep.append(dict(**key, **e, d_test_AP=e['test_AP'] - b['AP'],
                              d_val_AP=e['val_AP'] - r['baseline']['validation']['AP']))
        z = np.load(OUT / r['dataset'] / f"knob_xgb__{r['variant']}__seed{r['seed']}.npz")
        y = z['y_test']
        for tag, score, thr, ref in (('baseline', z['baseline'], z['threshold_baseline'], b),
                                     ('knob', z['knob'], z['threshold_knob'], k)):
            alert = score >= thr
            audit.append(dict(**key, score=tag, AP_saved=float(average_precision_score(y, score)), AP_json=ref['AP'],
                              TP_saved=int((alert & (y == 1)).sum()), TP_json=ref['TP'],
                              FP_saved=int((alert & (y == 0)).sum()), FP_json=ref['FP']))
    rows, fam, probe, sweep, audit = map(pd.DataFrame, (rows, fam, probe, sweep, audit))
    rows.to_csv(OUT / 'runs.csv', index=False); fam.to_csv(OUT / 'families.csv', index=False)
    probe.to_csv(OUT / 'boundary_and_knobs.csv', index=False); sweep.to_csv(OUT / 'intervention_sweep.csv', index=False)
    audit.to_csv(OUT / 'audit.csv', index=False)
    num = [c for c in rows.columns if c not in ('dataset', 'variant', 'seed', 'chosen')]
    summary = rows.groupby(['dataset', 'variant'])[num].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary.to_csv(OUT / 'summary.csv')
    fam_summary = fam.groupby(['dataset', 'variant', 'family'])[['d_val_AP', 'd_test_AP', 'd_test_F1', 'd_test_FN', 'd_test_FP']].agg(
        ['mean', 'std']).round(4)
    fam_summary.columns = [f'{a}_{b}' for a, b in fam_summary.columns]
    fam_summary.to_csv(OUT / 'families_summary.csv')
    sweep_mean = sweep.groupby(['dataset', 'variant', 'direction', 'tau'])[
        ['d_val_AP', 'd_test_AP', 'dlogp_errors', 'dlogp_anomalies', 'dlogp_normals', 'test_FN', 'test_FP']].mean()
    sweep_mean.round(4).to_csv(OUT / 'intervention_sweep_mean.csv')

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(rows.round(4).to_string()); print(fam_summary.to_string()); print(probe.round(3).to_string())
    bad = audit[(np.abs(audit.AP_saved - audit.AP_json) > 1e-6) | (audit.TP_saved != audit.TP_json)
                | (audit.FP_saved != audit.FP_json)]
    print(f'audit: {len(audit)} score sets re-derived, {len(bad)} mismatches')

    rows['dataset'] = rows.dataset + ' (' + rows.variant + ')'
    for t in (sweep, probe): t['dataset'] = t.dataset + ' (' + t.variant + ')'
    datasets = list(rows.dataset.unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 7.5), squeeze=False)
    colours = {'normal': '#c0392b', 'tangent': '#2c7fb8', 'random': '#7f7f7f'}
    for j, ds in enumerate(datasets):
        for i, (col, label) in enumerate((('d_test_AP', 'change in test AP'),
                                          ('dlogp_errors', 'change in log p(true label), baseline errors'))):
            ax = axes[i, j]
            for name, colour in colours.items():
                part = sweep[(sweep.dataset == ds) & (sweep.direction == name)]
                if part.empty: continue
                g = part.groupby('tau')[col].agg(['mean', 'std']).reset_index()
                x = np.arange(len(g))
                ax.errorbar(x, g['mean'], yerr=g['std'].fillna(0), marker='o', ms=4, lw=1.5, capsize=2,
                            color=colour, label=name)
                ax.set_xticks(x); ax.set_xticklabels([f'{t:+g}' for t in g.tau], rotation=60, fontsize=8)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_title(f'{ds}: {label}', fontsize=10); ax.set_xlabel('step size tau (+ = towards the "correct" side)')
            if i == 0 and j == 0: ax.legend(title='direction')
    fig.suptitle('Knob steering in XGBoost: normal direction vs tangent and random controls (mean and sd over 3 seeds)')
    fig.tight_layout(); fig.savefig(OUT / 'steering_dose_response.png', dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.8), squeeze=False)
    for j, ds in enumerate(datasets):
        part = probe[probe.dataset == ds]
        cols = ['acc_full_cnn', 'acc_full_mlp', 'acc_full_lr', 'acc_knob', 'acc_random_mean', 'acc_margin_only']
        labels = ['full H\nCNN', 'full H\nMLP', 'full H\nLR', 'top-k\nknobs', 'random-k\ndims', 'margin\nonly']
        ax = axes[0, j]
        ax.bar(labels, part[cols].mean(), yerr=part[cols].std().fillna(0), capsize=3,
               color=['#4c72b0'] * 3 + ['#c0392b', '#7f7f7f', '#55a868'])
        ax.axhline(0.5, color='k', lw=0.6, ls='--'); ax.set_ylim(0.4, 1.0)
        ax.set_title(f'{ds}: correct-vs-incorrect boundary\n(grouped-CV balanced accuracy)', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'boundary_accuracy.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
