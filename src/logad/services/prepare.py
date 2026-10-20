"""Service: raw log (+ optional labels) -> window_features.csv and companion artefacts.

Outputs in <out_dir>/ : window_features.csv, feature_names.json (source of truth for every
later stage), baselines.json, parse_report.json, meta.json. Re-running on the same inputs and
config produces byte-identical files (no wall-clock values are written).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from logad import __version__
from logad.config import resolve_path
from logad.features.baselines import add_baseline_features
from logad.features.extractors import (
    FAMILY_OF, FEATURE_FAMILIES, active_features, compute_raw_windows)
from logad.features.split import chronological_split
from logad.features.windowing import assign_windows
from logad.parsing.normalizer import add_empty_labels, attach_labels
from logad.parsing.ssh_parser import parse_file
from logad.threatintel.blocklist import BlocklistError, load_blocklist, sha256_file

META_COLUMNS = ["entity", "window_start", "split", "label", "attack_scenario",
                "history_windows", "low_history"]


def _dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def prepare_dataset(log_path, out_dir, cfg: dict, labels_path=None, dataset_name=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fcfg, wcfg = cfg["features"], cfg["windowing"]
    group_by = wcfg["group_by"]
    warnings: list[str] = []
    if fcfg.get("geoip_enabled"):
        raise NotImplementedError("GeoIP enrichment is deferred; set features.geoip_enabled: false")

    # 1. parse (nothing dropped silently) -----------------------------------------------
    events, report = parse_file(log_path, cfg["dataset"]["log_year"], cfg["dataset"]["tz_offset_hours"],
                                cfg["parser"]["failure_methods"], cfg["parser"].get("dedupe_exact_lines", False))
    if events.empty:
        raise ValueError("parser produced no events; check log format, year and parse_report")
    has_labels = labels_path is not None
    events = attach_labels(events, pd.read_csv(labels_path)) if has_labels else add_empty_labels(events)

    # 2. threat intel (static snapshot, offline) ------------------------------------------
    ti_enabled, blocklist, ti_meta = bool(fcfg["threat_intel_enabled"]), None, {"enabled": False}
    if ti_enabled:
        try:
            blocklist = load_blocklist(resolve_path(cfg["threat_intel"]["blocklist_path"]),
                                       resolve_path(cfg["threat_intel"]["manifest_path"]))
            ti_meta = {"enabled": True, "sha256": blocklist.sha256, "n_ips": len(blocklist.ips),
                       "verified_against_manifest": blocklist.verified, "source": blocklist.path}
            warnings += blocklist.warnings
        except BlocklistError as exc:
            if "not found" in str(exc):
                ti_enabled = False
                warnings.append(f"threat_intel disabled: {exc}")
            else:
                raise                                    # hash mismatch must never pass silently

    # 3. windows + baseline-free features ------------------------------------------------
    ev, win_stats = assign_windows(events, wcfg["window_seconds"], group_by)
    raw = compute_raw_windows(
        ev, group_by=group_by, tz_offset_hours=cfg["dataset"]["tz_offset_hours"],
        success_after_failed_min=fcfg["success_after_failed_min"],
        off_hours_start=fcfg["off_hours_start"], off_hours_end=fcfg["off_hours_end"],
        blocklist_ips=blocklist.ips if blocklist else frozenset())

    # 4. chronological split BEFORE anything is learned ------------------------------------
    s = cfg["split"]
    split, cutoffs = chronological_split(raw["window_start"], s["train"], s["val"], s["test"])
    raw["split"] = split

    # 5. baselines fitted on training windows only (normal-only when labels exist) ----------
    fit_mask = (raw["split"] == "train").to_numpy() & ~(raw["label"] == 1).to_numpy()
    table, baselines = add_baseline_features(
        raw, raw["split"].to_numpy(), fit_mask, group_by, fcfg["min_history_windows"], fcfg["z_clip"])

    # 6. feature selection: drop constant-by-construction features, never emit filler -----------
    features, dropped = active_features(group_by, False, ti_enabled)
    train_rows = table[table["split"] == "train"]
    const_train = [f for f in features if train_rows[f].nunique(dropna=False) <= 1]
    if const_train:
        warnings.append(f"constant on the training split (kept, review): {const_train}")
    if has_labels:
        for sp in ("train", "val", "test"):
            if (table.loc[table["split"] == sp, "label"] == 1).sum() == 0:
                warnings.append(f"no attack windows in the {sp} split")

    # 7. write artefacts -----------------------------------------------------------------
    out = table[META_COLUMNS + features].copy()
    out["window_start"] = out["window_start"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["label"] = out["label"].astype("Int64")
    out["low_history"] = out["low_history"].astype(int)
    out.to_csv(out_dir / "window_features.csv", index=False, float_format="%.10g")
    _dump({"schema_version": 1, "feature_names": features,
           "families": {fam: [f for f in fs if f in features] for fam, fs in FEATURE_FAMILIES.items()
                        if any(f in features for f in fs)},
           "dropped": dropped, "group_by": group_by, "window_seconds": int(wcfg["window_seconds"])},
          out_dir / "feature_names.json")
    baselines.to_json(out_dir / "baselines.json")
    _dump(report.to_dict(), out_dir / "parse_report.json")

    per_split = {}
    for sp in ("train", "val", "test"):
        d = table[table["split"] == sp]
        per_split[sp] = {"windows": int(len(d)), "attack_windows": int((d["label"] == 1).sum()),
                         "normal_windows": int((d["label"] == 0).sum()),
                         "unlabeled_windows": int(d["label"].isna().sum()),
                         "low_history_windows": int(d["low_history"].sum()),
                         "start": str(d["window_start"].min()), "end": str(d["window_start"].max())}
    meta = {
        "dataset": dataset_name or cfg["dataset"]["name"], "logad_version": __version__,
        "inputs": {"log": str(log_path), "log_sha256": sha256_file(log_path),
                   "labels": str(labels_path) if has_labels else None,
                   "labels_sha256": sha256_file(labels_path) if has_labels else None},
        "has_labels": has_labels, "window_stats": win_stats, "n_windows": int(len(table)),
        "split": {"cutoffs": cutoffs, "per_split": per_split}, "features": features, "dropped_features": dropped,
        "baseline_fit": {"windows_used": int(fit_mask.sum()), "entities": len(baselines.entities),
                         "policy": "fitted on training split only (normal-only when labels exist), frozen afterwards"},
        "threat_intel": ti_meta, "warnings": warnings, "config": cfg,
    }
    _dump(meta, out_dir / "meta.json")
    return meta
