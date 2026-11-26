"""Final table: three selection schemes of the combined XGBoost against every reference on the same splits.
Usage: python -m baselines.auto_aggregate"""
import json

import numpy as np
import pandas as pd

from .common import RESULTS

ORDER = ['sapienza', 'sd-stock', 'sd-crypto', 'tstr', 'ulb', 'elliptic', 'paysim', 'ibm_aml', 'tabformer']


def load(folder, ds):
    f = list((RESULTS / folder).glob(f'{ds}__*.json'))
    if not f: return None
    r = json.loads(f[0].read_text())
    ap = [x['test_AP'] for x in r['final']]
    c = r['chosen']
    label = f"{c['start']}/{c['reg']}" + ('/novelty' if c['novelty'] else '') + ('/ensemble' if c['ensemble'] else '')
    return dict(AP=float(np.mean(ap)), sd=float(np.std(ap, ddof=1)), F1=float(np.mean([x['test_F1'] for x in r['final']])),
                FN=float(np.mean([x['test_FN'] for x in r['final']])), FP=float(np.mean([x['test_FP'] for x in r['final']])), chosen=label,
                best_candidate=r.get('best_test_candidate', {}).get('test_AP'))


def main():
    cmp_ = pd.read_csv(RESULTS / 'kronos_compare/comparison.csv')
    rows = []
    for ds in ORDER:
        c = cmp_[cmp_.dataset == ds]
        ref = lambda pat: c[c.method.str.contains(pat, regex=False)]
        k3, k1 = ref('repaired, 3 seeds'), ref('repaired (recorded)')
        kronos = float(k3.AP.iloc[0]) if len(k3) else float(k1.AP.iloc[0])
        h = ref('HistGradientBoosting'); hgb = float(h.AP.iloc[0]) if len(h) else np.nan
        base = json.loads(next((RESULTS / 'intercept_xgb' / ds).glob('intercept__*seed42.json')).read_text())
        b = [json.loads(p.read_text())['results']['prior|sqrt'] for p in (RESULTS / 'intercept_xgb' / ds).glob('intercept__*.json')]
        row = dict(dataset=ds, kronos=kronos, hgb=hgb, best_reference=np.nanmax([kronos, hgb]),
                   baseline_AP=float(np.mean([x['test_AP'] for x in b])), baseline_F1=float(np.mean([x['test_F1'] for x in b])),
                   baseline_FN=float(np.mean([x['test_FN'] for x in b])), baseline_FP=float(np.mean([x['test_FP'] for x in b])))
        for tag, folder in (('s1', 'auto_xgb_first'), ('s2', 'auto_xgb'), ('s3', 'auto_xgb_cv')):
            r = load(folder, ds)
            if r: row.update({f'{tag}_{k}': v for k, v in r.items()})
        v = RESULTS / 'auto_xgb' / f'vs_hgb__{ds}.json'
        if v.exists():
            j = json.loads(v.read_text()); row.update(vs_hgb_diff=j['diff'], vs_hgb_lo=j['ci95'][0], vs_hgb_hi=j['ci95'][1])
        row['s1_minus_best_reference'] = row.get('s1_AP', np.nan) - row['best_reference']
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / 'auto_xgb' / 'final_table.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    print(t[['dataset', 'kronos', 'hgb', 'baseline_AP', 's1_AP', 's1_sd', 's2_AP', 's3_AP', 's1_minus_best_reference', 'vs_hgb_diff', 'vs_hgb_lo', 'vs_hgb_hi']].round(4).to_string())
    print(t[['dataset', 's1_chosen', 's2_chosen', 's3_chosen']].to_string())
    print(t[['dataset', 'baseline_F1', 's1_F1', 'baseline_FN', 's1_FN', 'baseline_FP', 's1_FP', 's1_best_candidate']].round(4).to_string())
    print('mean AP  baseline %.4f  s1 %.4f  s2 %.4f  s3 %.4f' % (t.baseline_AP.mean(), t.s1_AP.mean(), t.s2_AP.mean(), t.get('s3_AP', pd.Series([np.nan])).mean()))


if __name__ == '__main__':
    main()
