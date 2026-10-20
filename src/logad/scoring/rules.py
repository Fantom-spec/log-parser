"""Transparent detection rules that run beside the model.

R1 distributed brute force : many failures from many source IPs in one window
R2 success after failures  : a login succeeded after repeated failures (possible compromise)
R3 known-bad IP + failures : a blocklisted source IP together with failed logins (optional)

Thresholds come from config (never tuned on test). A rule whose input features are not in the
feature set (e.g. R1 when grouping by src_ip, R3 without a blocklist) is reported unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}
REQUIRES = {"R1": ["failed_count", "distinct_src_ips"], "R2": ["success_after_failed"],
            "R3": ["blocklisted_ip_count", "failed_count"]}
DESCRIPTION = {"R1": "distributed brute force", "R2": "success after repeated failures",
               "R3": "known-bad IP with failed logins"}


def apply_rules(df: pd.DataFrame, rules_cfg: dict):
    """-> (DataFrame of bool columns R1..R3 aligned with df, list of unavailable/disabled rules)."""
    hits = pd.DataFrame(False, index=df.index, columns=list(REQUIRES))
    off = []
    for rid, key in (("R1", "r1"), ("R2", "r2"), ("R3", "r3")):
        c = rules_cfg.get(key, {})
        if not c.get("enabled", True):
            off.append(f"{rid} disabled")
            continue
        missing = [f for f in REQUIRES[rid] if f not in df.columns]
        if missing:
            off.append(f"{rid} unavailable (missing {missing})")
            continue
        if rid == "R1":
            hits[rid] = (df["failed_count"] >= c["min_failed"]) & (df["distinct_src_ips"] >= c["min_distinct_ips"])
        elif rid == "R2":
            hits[rid] = df["success_after_failed"] >= 1
        else:
            hits[rid] = (df["blocklisted_ip_count"] >= 1) & (df["failed_count"] >= c["min_failed"])
    return hits, off


def rule_floor_index(hits: pd.DataFrame, floors: dict) -> np.ndarray:
    """Highest tier index demanded by any triggered rule (0 = none)."""
    out = np.zeros(len(hits), dtype=int)
    for rid in hits.columns:
        out = np.maximum(out, np.where(hits[rid].to_numpy(), TIER_INDEX[floors[rid]], 0))
    return out


def rule_explanation(rid: str, row, rules_cfg: dict) -> str:
    if rid == "R1":
        return f"R1 distributed brute force: {int(row['failed_count'])} failed logins from {int(row['distinct_src_ips'])} source IPs"
    if rid == "R2":
        return "R2 success after failures: a login succeeded after repeated failed attempts in the same window"
    return f"R3 known-bad IP: {int(row['blocklisted_ip_count'])} blocklisted source IP(s) with {int(row['failed_count'])} failed login(s)"
