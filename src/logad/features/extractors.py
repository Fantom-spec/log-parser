"""Per-window behavioural features that need no learned baseline.

Baseline-dependent features (event_rate_z, new_ip_count, hour_deviation) are added in
baselines.py so that every learned quantity is visibly fitted on training data only.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

BURST_SECONDS = 60  # fixed: the feature name max_failed_burst_60s encodes it

FEATURE_FAMILIES: dict[str, list[str]] = {
    "failed_logins": ["failed_count", "failed_ratio", "max_failed_burst_60s",
                      "consecutive_failed_max", "distinct_hosts_failed", "success_after_failed"],
    "event_frequency": ["event_count", "event_rate_z", "distinct_event_types"],
    "ip_activity": ["distinct_src_ips", "new_ip_count", "max_events_per_ip",
                    "geo_country_change", "distinct_users"],
    "time_patterns": ["hour_sin", "hour_cos", "is_off_hours", "is_weekend", "hour_deviation"],
    "threat_intel": ["blocklisted_ip_count", "blocklisted_event_count"],
}
ALL_FEATURES = [f for fam in FEATURE_FAMILIES.values() for f in fam]
FAMILY_OF = {f: fam for fam, fs in FEATURE_FAMILIES.items() for f in fs}


def active_features(group_by: str, geoip_enabled: bool, threat_intel_enabled: bool):
    """Features that carry information for this setup -> (ordered list, {dropped: reason}).

    Features that are constant *by construction* are dropped, never emitted as filler.
    """
    dropped: dict[str, str] = {}
    if not geoip_enabled:
        dropped["geo_country_change"] = "GeoIP data unavailable (deferred)"
    if not threat_intel_enabled:
        for f in FEATURE_FAMILIES["threat_intel"]:
            dropped[f] = "threat intel disabled or no blocklist snapshot available"
    if group_by == "user":
        dropped["distinct_users"] = "constant 1 when grouping by user"
    else:
        dropped["distinct_src_ips"] = "constant 1 when grouping by src_ip"
        dropped["max_events_per_ip"] = "equals event_count when grouping by src_ip"
    return [f for f in ALL_FEATURES if f not in dropped], dropped


def max_weighted_burst(times: np.ndarray, weights: np.ndarray, span: float) -> int:
    """Largest weighted event count within any half-open `span`-second interval."""
    best = cur = j = 0
    for i in range(len(times)):
        cur += int(weights[i])
        while times[i] - times[j] >= span:
            cur -= int(weights[j])
            j += 1
        best = max(best, cur)
    return best


def compute_raw_windows(ev: pd.DataFrame, *, group_by: str, tz_offset_hours: float,
                        success_after_failed_min: int, off_hours_start: float,
                        off_hours_end: float, blocklist_ips=frozenset()) -> pd.DataFrame:
    """One row per (entity, window_start) with all baseline-free features plus metadata.

    Underscore columns (_ips) and hour_local are internal and removed before export.
    """
    if ev.empty:
        raise ValueError("no attributed events to window")
    tz = pd.Timedelta(hours=tz_offset_hours)
    rows = []
    for (ws, ent), g in ev.groupby(["window_start", "entity"], sort=True):
        et = g["event_type"].to_numpy()
        rep = g["repeat"].to_numpy(dtype=np.int64)
        t = g["ts_sec"].to_numpy(dtype=float)
        hosts = g["host"].to_numpy()
        ips = g["src_ip"].to_numpy()
        users = g["username"].to_numpy()
        lab = g["label"].to_numpy(dtype=float)
        scen = g["scenario"].to_numpy()

        is_f, is_s = et == "LOGIN_FAILED", et == "LOGIN_SUCCESS"
        failed, succ = int(rep[is_f].sum()), int(rep[is_s].sum())

        run = best_run = fails_before = saf = 0
        for e, w in zip(et[is_f | is_s], rep[is_f | is_s]):
            if e == "LOGIN_FAILED":
                run += int(w)
                fails_before += int(w)
                best_run = max(best_run, run)
            else:
                if fails_before >= success_after_failed_min:
                    saf = 1
                run = 0

        ip_events: Counter = Counter()
        for ip, w in zip(ips, rep):
            if isinstance(ip, str):
                ip_events[ip] += int(w)
        ip_set = frozenset(ip_events)
        bl_ips = ip_set & blocklist_ips
        lt = ws + tz
        hour_local = lt.hour + lt.minute / 60.0 + lt.second / 3600.0

        if (lab == 1).any():
            label = 1.0
            sc = Counter(s for s, l in zip(scen, lab) if l == 1 and isinstance(s, str))
            scenario = sc.most_common(1)[0][0] if sc else ""
        elif not np.isnan(lab).any():
            label, scenario = 0.0, ""
        else:
            label, scenario = np.nan, ""      # unlabeled or partially labeled -> unknown

        rows.append({
            "entity": ent, "window_start": ws,
            "failed_count": failed,
            "failed_ratio": failed / (failed + succ) if (failed + succ) else 0.0,
            "max_failed_burst_60s": max_weighted_burst(t[is_f], rep[is_f], BURST_SECONDS),
            "consecutive_failed_max": best_run,
            "distinct_hosts_failed": len({h for h in hosts[is_f] if isinstance(h, str)}),
            "success_after_failed": saf,
            "event_count": int(rep.sum()),
            "distinct_event_types": len(set(et)),
            "distinct_src_ips": len(ip_set),
            "max_events_per_ip": max(ip_events.values()) if ip_events else 0,
            "distinct_users": len({u for u in users if isinstance(u, str)}),
            "hour_sin": math.sin(2 * math.pi * hour_local / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour_local / 24.0),
            "is_off_hours": int(hour_local < off_hours_start or hour_local >= off_hours_end),
            "is_weekend": int(lt.weekday() >= 5),
            "blocklisted_ip_count": len(bl_ips),
            "blocklisted_event_count": int(sum(ip_events[i] for i in bl_ips)),
            "hour_local": hour_local, "label": label, "attack_scenario": scenario,
            "_ips": ip_set,
        })
    return pd.DataFrame(rows)
