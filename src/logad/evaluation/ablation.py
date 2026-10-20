"""Pure helpers for ablation / paired comparisons (no I/O)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def make_variants(families: dict[str, list[str]], feature_order: list[str]) -> list[dict]:
    """Reference ('all'), leave-one-family-out ('drop_*') and single-family ('only_*') variants.

    Both views are needed: correlated features hide each other in leave-one-out, and a single
    family alone shows what it can do without help.
    """
    out = [{"name": "all", "mode": "all", "family": None, "features": list(feature_order)}]
    for fam, fs in families.items():
        keep = [f for f in feature_order if f not in fs]
        if keep:
            out.append({"name": f"drop_{fam}", "mode": "drop", "family": fam, "features": keep})
    for fam, fs in families.items():
        out.append({"name": f"only_{fam}", "mode": "only", "family": fam,
                    "features": [f for f in feature_order if f in fs]})
    return out


def resample_indices(y: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Shared bootstrap resamples (only those containing attacks), so every comparison is paired."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    out = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() > 0:
            out.append(idx)
    return out


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int((y.astype(bool) & pred).sum())
    fp = int((~y.astype(bool) & pred).sum())
    fn = int((y.astype(bool) & ~pred).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def paired_bootstrap(y, s_var, t_var, s_ref, t_ref, idx_list, level=0.95) -> dict:
    """Delta = variant - reference for PR-AUC and F1 (at each model's own fixed threshold).

    Windows are resampled independently although an incident spans several windows, so the
    intervals are optimistic. A delta is called 'distinguishable' only if its CI excludes 0.
    """
    y = np.asarray(y).astype(int)
    p_var, p_ref = s_var >= t_var, s_ref >= t_ref
    point = {"delta_pr_auc": float(average_precision_score(y, s_var) - average_precision_score(y, s_ref)),
             "delta_f1": float(_f1(y, p_var) - _f1(y, p_ref))}
    d_ap, d_f1 = [], []
    for idx in idx_list:
        yb = y[idx]
        d_ap.append(average_precision_score(yb, s_var[idx]) - average_precision_score(yb, s_ref[idx]))
        d_f1.append(_f1(yb, p_var[idx]) - _f1(yb, p_ref[idx]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = dict(point)
    for name, v in (("pr_auc", d_ap), ("f1", d_f1)):
        low, high = (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v else (None, None)
        out[f"delta_{name}_ci"] = [low, high]
        out[f"delta_{name}_distinguishable"] = bool(v) and (low > 0 or high < 0)
    out["n_resamples_used"] = len(d_ap)
    return out


def verdict(ci: list, pos: str = "supported", neg: str = "reversed") -> str:
    """Label a delta CI (variant minus reference): `pos` if wholly above 0, `neg` if wholly below,
    'no_change' if the delta is exactly 0, else 'inconclusive'."""
    if ci[0] is None:
        return "undefined"
    if ci[0] == 0 and ci[1] == 0:
        return "no_change"
    if ci[0] > 0:
        return pos
    if ci[1] < 0:
        return neg
    return "inconclusive"
