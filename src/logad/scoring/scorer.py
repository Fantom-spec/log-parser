"""Scoring: feature table or raw log -> score, tier, rules, evidence.

The same frozen preprocessing, feature order, baselines and blocklist snapshot as training are used,
so a replayed log produces the same features as the training pipeline for validation/test windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from logad.config import resolve_path
from logad.features.baselines import Baselines, apply_frozen_baselines
from logad.features.extractors import compute_raw_windows
from logad.features.windowing import assign_windows
from logad.io import load_json
from logad.models.registry import ModelRegistry
from logad.parsing.normalizer import add_empty_labels
from logad.parsing.ssh_parser import parse_file
from logad.scoring.explain import explain_window, feature_profile  # noqa: F401  (feature_profile re-exported)
from logad.scoring.rules import TIERS, apply_rules, rule_floor_index
from logad.scoring.tiers import assign_tier_index, thresholds_from_json
from logad.threatintel.blocklist import load_blocklist


class Scorer:
    def __init__(self, dataset_dir, cfg: dict, tag: str = "all"):
        d = Path(dataset_dir)
        self.cfg, self.dir = cfg, d
        self.meta = load_json(d / "meta.json")
        fn = load_json(d / "feature_names.json")
        self.features, self.group_by = fn["feature_names"], fn["group_by"]
        mdir = d / "models" / tag
        sc = load_json(mdir / "scoring.json")
        self.model_id = sc["model_id"]
        reg = ModelRegistry.load(mdir)
        entry = reg.entries[self.model_id]
        if entry["features"] != self.features:                      # train/serve skew guard
            raise ValueError("registry feature list differs from feature_names.json")
        self.version, self.model = entry["version"], reg.load_model(self.model_id)
        self.pre = joblib.load(mdir / "preprocessor.joblib")
        self.tier_thr = thresholds_from_json(sc["tier_thresholds"])
        self.profile, self.rules_cfg, self.floors = sc["profile"], sc["rules"], sc["rule_floors"]
        self.top_k = int(sc["evidence_top_k"])
        self.baselines = Baselines.from_dict(load_json(d / "baselines.json"))

    # ---- evidence ------------------------------------------------------------------
    def _evidence(self, row, rule_ids: list[str]) -> list[str]:
        return explain_window(row, rule_ids, self.features, self.profile, self.rules_cfg, self.top_k)

    # ---- scoring ---------------------------------------------------------------------
    def score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """df needs entity, window_start and the feature columns; other columns pass through."""
        df = df.reset_index(drop=True)
        X = self.pre.transform_raw(df) if self.model.input_kind == "raw" else self.pre.transform_scaled(df)
        score = self.model.score(X)
        model_idx = assign_tier_index(score, self.tier_thr)
        hits, _ = apply_rules(df, self.rules_cfg)
        final_idx = np.maximum(model_idx, rule_floor_index(hits, self.floors))
        out = df.copy()
        out["score"], out["model_tier_idx"], out["tier_idx"] = score, model_idx, final_idx
        out["model_tier"] = [TIERS[i] for i in model_idx]
        out["tier"] = [TIERS[i] for i in final_idx]
        out["rules"] = [[r for r in hits.columns if hits.at[i, r]] for i in range(len(df))]
        out["evidence"] = [self._evidence(df.iloc[i], out.at[i, "rules"]) if final_idx[i] else []
                           for i in range(len(df))]
        return out

    def feature_table_from_log(self, log_path, year: int | None = None) -> tuple[pd.DataFrame, dict]:
        """Raw log -> feature table with the training pipeline's windows and frozen baselines."""
        c = self.meta["config"]
        events, report = parse_file(log_path, year or c["dataset"]["log_year"], c["dataset"]["tz_offset_hours"],
                                    c["parser"]["failure_methods"], c["parser"].get("dedupe_exact_lines", False))
        if events.empty:
            raise ValueError("no parsable SSH events in the log")
        ev, win_stats = assign_windows(add_empty_labels(events), c["windowing"]["window_seconds"], self.group_by)
        bl = frozenset()
        if any(f.startswith("blocklisted_") for f in self.features):
            b = load_blocklist(resolve_path(c["threat_intel"]["blocklist_path"]),
                               resolve_path(c["threat_intel"]["manifest_path"]))
            if b.sha256 != self.meta["threat_intel"]["sha256"]:
                raise ValueError("blocklist snapshot differs from the one used for training")
            bl = b.ips
        f = c["features"]
        raw = compute_raw_windows(ev, group_by=self.group_by, tz_offset_hours=c["dataset"]["tz_offset_hours"],
                                  success_after_failed_min=f["success_after_failed_min"],
                                  off_hours_start=f["off_hours_start"], off_hours_end=f["off_hours_end"],
                                  blocklist_ips=bl)
        table = apply_frozen_baselines(raw, self.baselines, f["z_clip"])
        table["split"] = "scored"
        table["window_start"] = table["window_start"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return table, {"parse_report": report.to_dict(), "window_stats": win_stats}

    def score_log(self, log_path, year: int | None = None) -> tuple[pd.DataFrame, dict]:
        table, info = self.feature_table_from_log(log_path, year)
        return self.score_features(table), info
