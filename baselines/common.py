"""Shared metrics, validation-only threshold choice and result records."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SEEDS = (42, 13, 2026)


def pick_threshold(y, score):
    """Threshold with the best F1 on validation labels; never called on test labels."""
    if y.sum() == 0:
        return float(np.max(score)) + 1.0
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def metrics(y, score, threshold):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    alert = score >= threshold
    tp = int((alert & (y == 1)).sum()); fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum()); tn = int((~alert & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(n=int(len(y)), positives=int(y.sum()), prevalence=float(y.mean()),
                AP=float(average_precision_score(y, score)), ROC_AUC=float(roc_auc_score(y, score)),
                precision=precision, recall=recall, F1=f1, TP=tp, FP=fp, FN=fn, TN=tn,
                threshold=float(threshold))


def record(dataset, method, variant, seed, y_val, s_val, y_test, s_test, extra=None, save_scores=True):
    """Choose the threshold on validation, then score the held-out test split once."""
    threshold = pick_threshold(y_val, s_val)
    out = dict(dataset=dataset, method=method, variant=variant, seed=seed,
               validation=metrics(y_val, s_val, threshold), test=metrics(y_test, s_test, threshold),
               threshold_rule='max validation F1', **(extra or {}))
    folder = RESULTS / 'runs' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{method}__{variant}__seed{seed}'
    (folder / f'{stem}.json').write_text(json.dumps(out, indent=2) + '\n')
    if save_scores:
        scores = RESULTS / 'test_scores' / dataset
        scores.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scores / f'{stem}.npz', score=np.asarray(s_test),   # native precision
                            threshold=np.float64(threshold))
    print(f"{dataset:10s} {method:22s} {variant:12s} seed={seed:<5d} "
          f"val AP={out['validation']['AP']:.4f} | test AP={out['test']['AP']:.4f} "
          f"F1={out['test']['F1']:.4f} P={out['test']['precision']:.3f} R={out['test']['recall']:.3f} "
          f"TP/FP={out['test']['TP']}/{out['test']['FP']}", flush=True)
    return out
