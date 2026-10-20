"""Service: scoring latency and detection delay, measured (not assumed).

* latency: wall-clock for parse+features and for model scoring over a full log (median of repeats).
* detection delay: for each labelled incident, seconds from its first attack event to the CLOSE of the first
  flagged window containing incident events. Assumes a window is evaluated when it closes; late events would
  need a grace period on top. Incidents never flagged are counted as missed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from logad.config import PROJECT_ROOT
from logad.evaluation.latency import time_call
from logad.io import dump_json, load_json
from logad.parsing.normalizer import attach_labels
from logad.parsing.ssh_parser import parse_file
from logad.scoring.scorer import Scorer


def measure(dataset_dir, cfg: dict, tag: str = "all", repeats: int = 3, reports_root=None) -> dict:
    dataset_dir = Path(dataset_dir)
    sc = Scorer(dataset_dir, cfg, tag)
    meta, c = sc.meta, sc.meta["config"]
    log = meta["inputs"]["log"]
    ws = int(c["windowing"]["window_seconds"])
    t_feat = time_call(lambda: sc.feature_table_from_log(log), repeats)
    table, info = sc.feature_table_from_log(log)
    t_score = time_call(lambda: sc.score_features(table), repeats)
    n_ev, n_win = info["parse_report"]["event_lines"], len(table)
    out = {"dataset": meta["dataset"], "model_id": sc.model_id, "repeats": repeats,
           "latency": {"events": n_ev, "windows": n_win, "parse_and_features_s": t_feat, "model_scoring_s": t_score,
                       "events_per_second_end_to_end": n_ev / (t_feat + t_score),
                       "ms_per_window_end_to_end": (t_feat + t_score) * 1000 / n_win,
                       "ms_per_window_model_only": t_score * 1000 / n_win,
                       "note": "wall-clock on this machine, single process, whole log in memory"}}
    labels = meta["inputs"]["labels"]
    if labels:
        scored = sc.score_features(table)
        flagged = set(map(tuple, scored.loc[scored["tier"] != "NORMAL", ["entity", "window_start"]].itertuples(index=False, name=None)))
        ev, _ = parse_file(log, c["dataset"]["log_year"], c["dataset"]["tz_offset_hours"], c["parser"]["failure_methods"],
                           c["parser"].get("dedupe_exact_lines", False))
        lab = pd.read_csv(labels)
        ev = ev.merge(lab[["line_no", "label", "incident_id"]], on="line_no", how="left")
        ev = ev[(ev["label"] == 1) & ev["incident_id"].notna()]
        col = "username" if sc.group_by == "user" else "src_ip"
        ev = ev[ev[col].notna()].copy()
        ev["ws"] = ev["ts"].dt.floor(f"{ws}s")
        delays, missed = {}, []
        for inc, g in ev.groupby("incident_id"):
            first = g["ts"].min()
            ends = [w + pd.Timedelta(seconds=ws) for e, w in zip(g[col], g["ws"])
                    if (e, w.strftime("%Y-%m-%dT%H:%M:%SZ")) in flagged]
            if ends:
                delays[inc] = (min(ends) - first).total_seconds()
            else:
                missed.append(inc)
        d = np.array(list(delays.values()), dtype=float)
        out["detection_delay"] = {
            "incidents": len(delays) + len(missed), "detected": len(delays), "missed": len(missed), "missed_ids": missed,
            "delay_seconds": ({"min": float(d.min()), "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
                              if len(d) else None),
            "window_seconds": ws,
            "note": "delay = close of first flagged window minus first attack event; late events would add a grace period",
            "scope": "all splits (includes training-period incidents); a detection-timing demo, not an accuracy estimate"}
    root = Path(reports_root) if reports_root else PROJECT_ROOT / "reports"
    dump_json(out, root / "metrics" / meta["dataset"] / tag / "robustness.json")
    return out
