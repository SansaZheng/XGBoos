"""Table: our supervised XGBoost against the recorded label-free methods of SansaZheng/Financial_unsuperviesdAD.

Usage: python -m baselines.finunsup_aggregate --repo PATH_TO_Financial_unsuperviesdAD
Recorded numbers are read from that repository's results/roleweight_20261117 summaries (never re-run here).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import RESULTS

RECENT = {  # method id in the recorded tables -> (name, source, year); variants of one method are merged (best per task)
    'cae_if_core': 'CAE-IF (ICMLA 2025)', 'rgld2026': 'RGLD-ID (arXiv 2026)', 'rttad2026': 'RTTAD-TTA (arXiv 2026)',
    'kronos_potratio_matched': 'KronosAD-M (Kronos, 2025)', 'kronos_stable_nc': 'Kronos-S (Kronos, 2025)',
    'himvh_repair_cm': 'HIMVH-Zero (WWW 2026)', 'himvh_repair_legacy': 'HIMVH-Zero (WWW 2026)', 'himvh_roleweight_cm_tail': 'HIMVH-Zero (WWW 2026)',
    'sdfmm_repair_ge': 'SD-FMM-Zero (IPM 2026)', 'sdfmm_repair_legacy': 'SD-FMM-Zero (IPM 2026)',
    'huge_roleweight_dual': 'HUGE (AAAI 2025)', 'huge_roleweight_legacy': 'HUGE (AAAI 2025)', 'huge_roleweight_role': 'HUGE (AAAI 2025)',
    'ebgad_release_zero': 'EB-GAD (arXiv 2026)', 'audit_hcl_zero_adapted': 'Audit-HCL (Sci. Rep. 2026)'}
ORDER = ['ulb', 'banksim', 's_ffsd', 'sapienza_5S', 'sapienza_15S', 'sapienza_25S', 'sd_stock', 'sd_crypto', 'sd_synth', 'graph_banksim', 'graph_s_ffsd']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True); a = p.parse_args()
    rec = []
    for f, prefix in (('combined_transaction_summary.csv', ''), ('combined_account_graph_summary.csv', 'graph_')):
        d = pd.read_csv(a.repo / 'results/roleweight_20261117' / f)
        d['task'] = prefix + d.dataset
        rec.append(d)
    rec = pd.concat(rec)
    ap = rec[rec.operating_point == 'pot'].set_index(['task', 'method']).average_precision_mean
    f1 = rec.groupby(['task', 'method']).f1_mean.max()              # better of the two label-free thresholds
    rows, long = [], []
    for task in ORDER:
        ours = json.loads((RESULTS / 'finunsup_compare' / f'fu_{task}.json').read_text())
        o = pd.DataFrame(ours['rows']).groupby('variant')[['test_AP', 'test_F1', 'test_F1_q99', 'train_pos']].agg(['mean', 'std'])
        auto = json.loads((RESULTS / 'finunsup_auto' / f'fu_{task}__raw.json').read_text())
        fa = pd.DataFrame(auto['final'])
        t_ap, t_f1 = ap[task], f1[task]
        recent = t_ap[[m for m in t_ap.index if m in RECENT]]
        r = dict(task=task, test_rows=ours['test_rows'], test_pos=ours['test_pos'],
                 best_recorded=t_ap.idxmax(), best_recorded_AP=t_ap.max(), best_recent=RECENT[recent.idxmax()], best_recent_AP=recent.max(),
                 best_recorded_F1=t_f1.max(),
                 plain_AP=o.loc['plain', ('test_AP', 'mean')], noweight_AP=o.loc['noweight', ('test_AP', 'mean')],
                 combined_AP=fa.test_AP.mean(), combined_AP_sd=fa.test_AP.std(ddof=1), combined_F1=fa.test_F1.mean(),
                 noweight_F1_q99=o.loc['noweight', ('test_F1_q99', 'mean')],
                 rows8192_AP=o.loc['rows8192', ('test_AP', 'mean')], rows8192_pos=o.loc['rows8192', ('train_pos', 'mean')],
                 pos10_AP=o.loc['pos10', ('test_AP', 'mean')], pos50_AP=o.loc['pos50', ('test_AP', 'mean')],
                 chosen=json.dumps(auto['chosen']))
        r['combined_minus_best_recorded'] = r['combined_AP'] - r['best_recorded_AP']
        rows.append(r)
        for m in recent.index:
            long.append(dict(task=task, method=RECENT[m], AP=float(recent[m])))
    t = pd.DataFrame(rows)
    out = RESULTS / 'finunsup_compare'
    t.to_csv(out / 'table.csv', index=False)
    wide = pd.DataFrame(long).groupby(['method', 'task']).AP.max().unstack()[[c for c in ORDER if c in set(pd.DataFrame(long).task)]]
    wide.loc['Combined XGBoost (ours, uses labels)'] = t.set_index('task').combined_AP
    wide.to_csv(out / 'recent_methods_AP.csv')
    pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(t.drop(columns=['chosen']).T.to_string()); print(wide.to_string()); print(t[['task', 'chosen']].to_string())


if __name__ == '__main__':
    main()
