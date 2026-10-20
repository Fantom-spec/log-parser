"""Risk tiers calibrated on VALIDATION data.

NORMAL < LOW <= MEDIUM <= HIGH <= CRITICAL, all on one model's anomaly score.
LOW is the Phase 2 operating threshold (max F1 on validation). Each higher tier is the lowest
validation threshold whose precision reaches its target (with a minimum number of true positives
flagged so a lucky top-scored window cannot define a tier). A tier whose target is never reached
gets an infinite threshold, i.e. it is unreachable by score, and is reported as such.
"""
from __future__ import annotations

import numpy as np

from .rules import TIERS


def _precision_at(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, int]:
    m = s >= thr
    return (float(y[m].mean()) if m.any() else 0.0), int(y[m].sum())


def calibrate_tiers(y_val, s_val, t_op: float, targets: dict, min_tp: int = 3) -> dict:
    y, s = np.asarray(y_val).astype(int), np.asarray(s_val, dtype=float)
    thr = {"LOW": float(t_op)}
    info = {"LOW": {"validation_precision": _precision_at(y, s, t_op)[0], "validation_true_positives": _precision_at(y, s, t_op)[1]}}
    prev, candidates = float(t_op), np.unique(s)
    for tier in ("MEDIUM", "HIGH", "CRITICAL"):
        target, found = float(targets[tier]), None
        for c in candidates[candidates >= prev]:
            p, tp = _precision_at(y, s, c)
            if p >= target and tp >= min_tp:
                found = float(c)
                break
        prev_tier = {"MEDIUM": "LOW", "HIGH": "MEDIUM", "CRITICAL": "HIGH"}[tier]
        thr[tier] = found if found is not None else float("inf")
        merged = found is not None and found == thr[prev_tier]
        prev = thr[tier] if found is not None else prev
        info[tier] = {"target_precision": target, "reachable": found is not None,
                      # same cut-point as the tier below: the lower tier is empty by score (higher tier wins)
                      **({"merged_with": prev_tier} if merged else {}),
                      **({"validation_precision": _precision_at(y, s, found)[0],
                          "validation_true_positives": _precision_at(y, s, found)[1]} if found is not None else {})}
    return {"thresholds": thr, "info": info, "min_true_positives": int(min_tp)}


def thresholds_from_json(d: dict) -> dict:
    """JSON stores unreachable tiers as null; restore +inf."""
    return {k: (float("inf") if v is None else float(v)) for k, v in d.items()}


def assign_tier_index(scores, thresholds: dict) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    idx = np.zeros(len(s), dtype=int)
    for i, tier in enumerate(TIERS[1:], 1):
        idx[s >= thresholds[tier]] = i
    return idx
