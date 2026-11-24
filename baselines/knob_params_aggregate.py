"""Tables and figures for baselines.knob_params.  Usage: python -m baselines.knob_params_aggregate"""
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import RESULTS
from .knob_params import GAINS, SCAN

OUT = RESULTS / 'knob_params'
ORDER = ['sapienza', 'sd-stock', 'sd-crypto', 'tstr', 'ulb', 'elliptic', 'paysim', 'ibm_aml', 'tabformer']


def main():
    runs = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/model_knobs__*.json'))]
    rows = []
    for r in runs:
        b = r['baseline']
        d = lambda x, k: x[k] - b[k]
        rnd = r['random_gain']
        rows.append(dict(
            dataset=r['dataset'], variant=r['variant'], seed=r['seed'], trees=r['trees'], d=r['features'],
            acc_full=max(r['boundary_cv_balanced_accuracy'].values()), k_knob=r['k_knob'],
            knobs=' + '.join(r['knob_features']), acc_knob=r['acc_knob'],
            acc_random=np.mean(r['acc_random']) if r['acc_random'] else np.nan,
            gap_share_of_knobs=r['share_of_knobs'], top_share_feature=next(iter(r['share_by_feature'])),
            top_share=next(iter(r['share_by_feature'].values())), best_rank_of_knob_by_share=min(r['rank_of_knobs_by_share']),
            late_ratio=r['late_ratio'], share_first_half_blocks=float(np.sum(r['share_by_block'][:r['layers'] // 2])),
            base_AP=b['test_AP'], base_F1=b['test_F1'], base_FN=b['test_FN'], base_FP=b['test_FP'],
            knob_gains=json.dumps(r['knob_gain']['names']),
            knob_dvalAP=d(r['knob_gain'], 'val_AP'), knob_dAP=d(r['knob_gain'], 'test_AP'), knob_dF1=d(r['knob_gain'], 'test_F1'),
            knob_dFN=d(r['knob_gain'], 'test_FN'), knob_dFP=d(r['knob_gain'], 'test_FP'),
            random_dAP=np.mean([d(x, 'test_AP') for x in rnd]) if rnd else np.nan,
            random_dvalAP=np.mean([d(x, 'val_AP') for x in rnd]) if rnd else np.nan,
            share_dAP=d(r['share_gain'], 'test_AP'), block_dvalAP=d(r['block_gain'], 'val_AP'),
            block_dAP=d(r['block_gain'], 'test_AP'), block_dF1=d(r['block_gain'], 'test_F1'), block_dFN=d(r['block_gain'], 'test_FN')))
    model = pd.DataFrame(rows)
    model.to_csv(OUT / 'model_knobs.csv', index=False)
    num = [c for c in model.columns if model[c].dtype != object and c not in ('seed',)]
    ms = model.groupby('dataset')[num].agg(['mean', 'std'])
    ms.columns = [f'{a}_{b}' for a, b in ms.columns]
    ms = ms.reindex([d for d in ORDER if d in ms.index])
    ms.to_csv(OUT / 'model_knobs_summary.csv')
    knobs = {ds: Counter(f for r in runs if r['dataset'] == ds for f in r['knob_features']) for ds in ms.index}

    hyper = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/hyper_knobs__*.json'))]
    sens, tuned = [], []
    for h in hyper:
        for name, s in h['sensitivity'].items():
            sens.append(dict(dataset=h['dataset'], hyperparameter=name, **s))
        m = lambda rows_, k: float(np.mean([x[k] for x in rows_]))
        sd = lambda rows_, k: float(np.std([x[k] for x in rows_], ddof=1))
        t = h['three_seeds']
        diff = [a['test_AP'] - b['test_AP'] for a, b in zip(t['tuned'], t['baseline'])]
        tuned.append(dict(dataset=h['dataset'], knobs=' + '.join(h['knob_hyperparameters']), chosen=json.dumps(h['chosen']),
                          changed=any(x['setting'] for x in t['tuned']),
                          base_valAP=m(t['baseline'], 'val_AP'), tuned_valAP=m(t['tuned'], 'val_AP'),
                          base_AP=m(t['baseline'], 'test_AP'), base_AP_sd=sd(t['baseline'], 'test_AP'),
                          tuned_AP=m(t['tuned'], 'test_AP'), tuned_AP_sd=sd(t['tuned'], 'test_AP'),
                          dAP=float(np.mean(diff)), dAP_sd=float(np.std(diff, ddof=1)), seeds_improved=int(sum(x > 0 for x in diff)),
                          base_F1=m(t['baseline'], 'test_F1'), tuned_F1=m(t['tuned'], 'test_F1'),
                          base_FN=m(t['baseline'], 'test_FN'), tuned_FN=m(t['tuned'], 'test_FN'),
                          base_FP=m(t['baseline'], 'test_FP'), tuned_FP=m(t['tuned'], 'test_FP'),
                          base_gap=m(t['baseline'], 'test_margin_gap_sd'), tuned_gap=m(t['tuned'], 'test_margin_gap_sd')))
    sens, tuned = pd.DataFrame(sens), pd.DataFrame(tuned)
    if len(sens):
        sens.to_csv(OUT / 'hyper_sensitivity.csv', index=False)
        tuned = tuned.set_index('dataset').reindex([d for d in ORDER if d in set(tuned.dataset)]).reset_index()
        tuned.to_csv(OUT / 'hyper_tuned.csv', index=False)

    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 80)
    cols = ['acc_full', 'k_knob', 'acc_knob', 'acc_random', 'gap_share_of_knobs', 'share_first_half_blocks', 'late_ratio',
            'knob_dvalAP', 'knob_dAP', 'random_dAP', 'share_dAP', 'block_dvalAP', 'block_dAP', 'knob_dF1', 'knob_dFN', 'block_dFN']
    print(ms[[f'{c}_mean' for c in cols]].round(4).to_string())
    print(ms[[f'{c}_std' for c in ('knob_dAP', 'random_dAP', 'block_dAP')]].round(4).to_string())
    for ds, c in knobs.items(): print(ds, dict(c.most_common(6)))
    print(model[['dataset', 'seed', 'k_knob', 'knobs', 'knob_gains', 'top_share_feature', 'top_share', 'best_rank_of_knob_by_share']]
          .assign(knobs=model.knobs.str[:60], knob_gains=model.knob_gains.str[:70]).round(3).to_string())
    if len(sens):
        piv = sens.pivot(index='dataset', columns='hyperparameter', values='val_AP_range').reindex(
            [d for d in ORDER if d in set(sens.dataset)])[list(SCAN)]
        print(piv.round(4).to_string()); print(tuned.round(4).to_string())
        fig, ax = plt.subplots(figsize=(10, 4.6))
        rel = piv.div(piv.max(1), axis=0)
        im = ax.imshow(rel.values, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=35, ha='right')
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f'{piv.values[i, j]:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if rel.values[i, j] > 0.6 else 'black')
        ax.set_title('Hyper-parameter knobs: range of validation AP when one hyper-parameter is changed\n'
                     '(numbers: AP range; colour: relative to the most sensitive one in the same task)', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.025); fig.tight_layout()
        fig.savefig(OUT / 'hyper_sensitivity.png', dpi=130); plt.close(fig)

    datasets = list(ms.index)
    ncols = min(5, len(datasets)); nrows = -(-len(datasets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel()[len(datasets):]: ax.axis('off')
    for i, ds in enumerate(datasets):
        ax = axes.ravel()[i]
        for r in [x for x in runs if x['dataset'] == ds]:
            c = r['dose_top_knob']['curve']
            ax.plot(range(len(GAINS)), [p['test_AP'] - r['baseline']['test_AP'] for p in c], marker='o', ms=3, lw=1.2,
                    label=f"{r['seed']}: {r['dose_top_knob']['feature'][:22]}")
        ax.axhline(0, color='k', lw=0.6); ax.set_xticks(range(len(GAINS))); ax.set_xticklabels([f'{g:g}' for g in GAINS], fontsize=7)
        ax.set_title(ds, fontsize=10); ax.legend(fontsize=6, title='seed: top knob', title_fontsize=6)
    fig.suptitle('Turning the strongest anomaly/normal knob feature: change in test AP against its gain (1 = untouched)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'gain_dose_response.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
