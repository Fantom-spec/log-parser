"""Threshold calibration and evaluation metrics.

Convention everywhere: predict anomaly when score >= threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def calibrate_threshold(y: np.ndarray, score: np.ndarray) -> dict:
    """Threshold maximising F1 on the VALIDATION split (ties -> highest threshold = fewer alerts)."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("validation split needs both normal and attack windows to calibrate a threshold")
    p, r, t = precision_recall_curve(y, score)
    p, r = p[:-1], r[:-1]
    f1 = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
    best = f1.max()
    thr = float(t[np.flatnonzero(f1 >= best - 1e-12)[-1]])
    return {"threshold": thr, "strategy": "max_f1_on_validation", "val_f1_at_threshold": float(best)}


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {"tp": int((y & pred).sum()), "fp": int((~y & pred).sum()),
            "fn": int((y & ~pred).sum()), "tn": int((~y & ~pred).sum())}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def threshold_metrics(y, score, thr, span_days=None) -> dict:
    """Metrics at one operating point, plus threshold-free PR-AUC / ROC-AUC and class balance."""
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    pred = score >= thr
    cm = confusion(y, pred)
    p, r, f1 = _prf(cm["tp"], cm["fp"], cm["fn"])
    neg = cm["fp"] + cm["tn"]
    out = {
        "threshold": float(thr), "n_windows": int(len(y)), "n_attack": int(y.sum()),
        "prevalence": float(y.mean()), "precision": p, "recall": r, "f1": f1,
        "false_positive_rate": cm["fp"] / neg if neg else 0.0,
        "n_alerts": int(pred.sum()), **cm,
        "pr_auc": float(average_precision_score(y, score)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None,
    }
    if span_days:
        out["alerts_per_day"] = out["n_alerts"] / span_days
    return out


def per_scenario_recall(scenarios, y, pred) -> dict:
    """Recall at the operating point for each attack scenario (shows what each model misses)."""
    scenarios, y, pred = np.asarray(scenarios, dtype=object), np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    out = {}
    for s in sorted({x for x, yy in zip(scenarios, y) if yy and isinstance(x, str) and x}):
        m = y & (scenarios == s)
        out[s] = {"n_windows": int(m.sum()), "recall": float(pred[m].mean())}
    return out


def bootstrap_ci(y, score, thr, n_boot=1000, seed=0, level=0.95) -> dict:
    """Percentile bootstrap over test windows at a FIXED threshold.

    Caveat: windows are resampled independently although an attack incident spans several
    windows, so intervals are optimistic (too narrow). Treat as indicative.
    """
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    pred_all = score >= thr
    vals = {"precision": [], "recall": [], "f1": [], "pr_auc": []}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() == 0:
            continue
        cm = confusion(yb, pred_all[idx])
        p, r, f = _prf(cm["tp"], cm["fp"], cm["fn"])
        vals["precision"].append(p); vals["recall"].append(r); vals["f1"].append(f)
        vals["pr_auc"].append(average_precision_score(yb, score[idx]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: {"low": float(np.percentile(v, lo)), "high": float(np.percentile(v, hi))}
            for k, v in vals.items() if v} | {"n_resamples_used": len(vals["f1"]), "level": level}


def aggregate(metric_dicts: list[dict], keys=("precision", "recall", "f1", "pr_auc", "false_positive_rate",
                                              "alerts_per_day", "n_alerts")) -> dict:
    out = {"n_seeds": len(metric_dicts)}
    for k in keys:
        v = np.array([m[k] for m in metric_dicts if m.get(k) is not None], dtype=float)
        if len(v):
            out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def binary_metrics(y, pred, span_days=None) -> dict:
    """Confusion-based metrics for a boolean alert decision (no scores needed)."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(bool)
    cm = confusion(y, pred)
    p, r, f1 = _prf(cm["tp"], cm["fp"], cm["fn"])
    neg = cm["fp"] + cm["tn"]
    out = {"n_windows": int(len(y)), "n_attack": int(y.sum()), "precision": p, "recall": r, "f1": f1,
           "false_positive_rate": cm["fp"] / neg if neg else 0.0, "n_alerts": int(pred.sum()), **cm}
    if span_days:
        out["alerts_per_day"] = out["n_alerts"] / span_days
    return out
