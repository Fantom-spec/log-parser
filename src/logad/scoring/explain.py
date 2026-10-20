"""Evidence for flagged windows: triggered rules plus the features that deviate most from normal training windows.

This is feature-deviation evidence, NOT a causal attribution. A window is explained only when it is flagged;
the deviation is a robust z-score against the median/scale of normal training windows (`feature_profile`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from logad.scoring.rules import rule_explanation

Z_MIN = 3.0   # a feature is listed only when it is at least this many robust deviations from typical


def feature_profile(fit_df: pd.DataFrame, features: list[str]) -> dict:
    """Typical value and spread of each feature over normal training windows (for evidence text)."""
    prof = {}
    for f in features:
        x = fit_df[f].to_numpy(float)
        med = float(np.median(x))
        prof[f] = {"median": med, "scale": float(max(1.4826 * np.median(np.abs(x - med)), x.std(), 0.1))}   # 0.1 floor: no absurd z for constant features
    return prof


def explain_window(row, rule_ids: list[str], features: list[str], profile: dict, rules_cfg: dict, top_k: int) -> list[str]:
    """Evidence lines: one per triggered rule, then up to `top_k` most deviating features with |z| >= Z_MIN."""
    out = [rule_explanation(r, row, rules_cfg) for r in rule_ids]
    dev = []
    for f in features:
        p = profile[f]
        z = (float(row[f]) - p["median"]) / p["scale"]
        dev.append((abs(z), f, float(row[f]), p["median"], z))
    for _, f, v, med, z in sorted(dev, reverse=True)[:top_k]:
        if abs(z) >= Z_MIN:
            out.append(f"{f}={v:g} (typical {med:g}, {z:+.1f} robust deviations)")
    return out
