"""One table per task: recorded Kronos-mini (Kronos_SupervisedAD), that repository's gradient-boosting
control, XGBoost and Knob-XGBoost, all on the same splits and test rows.

Usage: python -m baselines.kronos_table
"""
import json

import numpy as np
import pandas as pd

from .common import RESULTS

VARIANT = {'sapienza': 'raw', 'sd-stock': 'raw', 'sd-crypto': 'causal', 'tstr': 'summary', 'ulb': 'raw',
           'elliptic': 'raw', 'paysim': 'temporal', 'ibm_aml': 'temporal', 'tabformer': 'raw'}
OUT = RESULTS / 'kronos_compare'


def agg(tests, method, note=''):
    f1 = [t.get('F1', t.get('f1')) for t in tests]; ap = [t.get('AP', t.get('average_precision')) for t in tests]
    return dict(method=method, runs=len(tests), AP=np.mean(ap), AP_sd=np.std(ap, ddof=1) if len(ap) > 1 else np.nan,
                F1=np.mean(f1), F1_sd=np.std(f1, ddof=1) if len(f1) > 1 else np.nan,
                precision=np.mean([t['precision'] for t in tests]), recall=np.mean([t['recall'] for t in tests]), note=note)


def main():
    rec = pd.read_csv(RESULTS / 'kronos_reference/repair_rounds_20261103_comparison.csv').set_index('dataset')
    conf = pd.read_csv(RESULTS / 'kronos_reference/repair_rounds_20261103_seed_confirmation.csv')
    rows = []
    for ds, variant in VARIANT.items():
        r = rec.loc[ds]
        info = dict(dataset=ds, unit=r.unit, test_n=int(r.n), test_positives=int(r.positives))
        rows.append(dict(**info, method='Kronos-mini R0 (recorded)', runs=1, AP=r.old_AP, F1=r.old_F1,
                         precision=r.old_precision, recall=r.old_recall, note='seed 42, not rerun'))
        rows.append(dict(**info, method='Kronos-mini repaired (recorded)', runs=1, AP=r.new_AP, F1=r.new_F1,
                         precision=r.new_precision, recall=r.new_recall, note=f'{r.selected_round}, seed 42, not rerun'))
        c = conf[(conf.dataset == ds) & (conf.method == 'selected')]
        if len(c):
            rows.append(dict(**info, method='Kronos-mini repaired, 3 seeds (recorded)', runs=3, AP=c.AP_mean.iloc[0],
                             AP_sd=c.AP_sample_sd.iloc[0], F1=c.F1_mean.iloc[0], F1_sd=c.F1_sample_sd.iloc[0],
                             note='not rerun'))
        path = OUT / f'{ds}.json'
        if path.exists():
            cmp_ = json.loads(path.read_text())
            if cmp_['hgb']:
                tests = [v['test'] for v in cmp_['hgb'].values()]
                same = len({round(t['average_precision'], 12) for t in tests}) == 1
                rows.append(dict(**info, **agg(tests[:1] if same else tests, 'HistGradientBoosting control',
                                               'repo settings; identical for all seeds' if same else 'repo settings')))
            for v in sorted({k.split('__')[0] for k in cmp_['xgb']}):
                tests = [x['test'] for k, x in cmp_['xgb'].items() if k.startswith(v + '__')]
                val = np.mean([x['validation']['AP'] for k, x in cmp_['xgb'].items() if k.startswith(v + '__')])
                if v != variant:
                    rows.append(dict(**info, **agg(tests, f'XGBoost ({v})', f'not chosen; validation AP {val:.4f}')))
        runs = [json.loads(p.read_text()) for p in sorted((RESULTS / 'knob' / ds).glob(f'knob_xgb__{variant}__seed*.json'))]
        if runs:
            val = np.mean([x['baseline']['validation']['AP'] for x in runs])
            rows.append(dict(**info, **agg([x['baseline']['test'] for x in runs], f'XGBoost ({variant})', f'validation AP {val:.4f}')))
            rows.append(dict(**info, **agg([x['knob_xgb']['test'] for x in runs], 'Knob-XGBoost',
                                           'chosen: ' + ', '.join(x['chosen'] for x in runs))))
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / 'comparison.csv', index=False)
    pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 70)
    print(table.drop(columns=['unit']).round(4).to_string())


if __name__ == '__main__':
    main()
