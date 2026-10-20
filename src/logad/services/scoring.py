"""Service: calibrate risk tiers on validation, then evaluate model-only vs model+rules.

calibrate_scoring reads train and validation rows only. evaluate_scoring reports validation
(calibration data, labelled as such) and held-out test side by side.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from logad.config import PROJECT_ROOT
from logad.evaluation import plots
from logad.evaluation.metrics import binary_metrics, per_scenario_recall
from logad.io import dump_json, load_json
from logad.models.registry import ModelRegistry
from logad.preprocessing.pipeline import load_dataset, training_mask
from logad.scoring.rules import TIERS, apply_rules
from logad.scoring.explain import feature_profile
from logad.scoring.scorer import Scorer
from logad.scoring.tiers import calibrate_tiers
from logad.services.train import _span_days


def calibrate_scoring(dataset_dir, cfg: dict, tag: str = "all") -> dict:
    dataset_dir = Path(dataset_dir)
    df, names, meta = load_dataset(dataset_dir)
    if not meta["has_labels"]:
        raise ValueError("tier calibration needs a labelled validation split")
    sc = cfg["scoring"]
    mdir = dataset_dir / "models" / tag
    reg = ModelRegistry.load(mdir)
    mid = sc["model_id"]
    if mid not in reg.entries:
        raise ValueError(f"model {mid} not found in registry (have {sorted(reg.entries)})")
    thresholds = load_json(mdir / "thresholds.json")
    pre = joblib.load(mdir / "preprocessor.joblib")
    model = reg.load_model(mid)
    val = df[(df["split"] == "val") & df["label"].notna()]            # test rows are never read here
    X = pre.transform_raw(val) if model.input_kind == "raw" else pre.transform_scaled(val)
    cal = calibrate_tiers(val["label"].astype(int).to_numpy(), model.score(X), thresholds[mid]["threshold"],
                          sc["tier_precision_targets"], sc["tier_min_true_positives"])
    out = {"model_id": mid, "model_version": reg.entries[mid]["version"], "tier_thresholds": cal["thresholds"],
           "tier_calibration": cal["info"], "rules": sc["rules"], "rule_floors": sc["rule_floors"],
           "evidence_top_k": sc["evidence_top_k"], "profile": feature_profile(df[training_mask(df)], names),
           "note": "tiers are validation-precision bands; evidence lists feature deviations, not causal attribution"}
    dump_json(out, mdir / "scoring.json")
    return out


def _config_metrics(scored, y, span):
    """model_only / rules_only / model_plus_rules alert decisions on one labelled frame."""
    any_rule = scored["rules"].map(len).to_numpy() > 0
    decisions = {"model_only": scored["model_tier_idx"].to_numpy() >= 1,
                 "rules_only": any_rule,
                 "model_plus_rules": scored["tier_idx"].to_numpy() >= 1}
    out = {}
    for name, pred in decisions.items():
        out[name] = {**binary_metrics(y, pred, span),
                     "per_scenario_recall": per_scenario_recall(scored["attack_scenario"].to_numpy(), y, pred)}
    return out


def _tier_table(scored, y, col="tier"):
    t = scored[col].to_numpy()
    return {tier: {"windows": int((t == tier).sum()), "attack_windows": int(y[t == tier].sum()),
                   "attack_rate": float(y[t == tier].mean()) if (t == tier).any() else None} for tier in TIERS}


def _rule_table(scored, y):
    out = {}
    for r in ("R1", "R2", "R3"):
        m = scored["rules"].map(lambda x: r in x).to_numpy()
        out[r] = {"hits": int(m.sum()), "attack_hits": int(y[m].sum()),
                  "precision": float(y[m].mean()) if m.any() else None,
                  "recall": float(y[m].sum() / y.sum()) if y.sum() else None}
    return out


def evaluate_scoring(dataset_dir, cfg: dict, tag: str = "all", reports_root=None) -> dict:
    dataset_dir = Path(dataset_dir)
    df, names, meta = load_dataset(dataset_dir)
    scorer = Scorer(dataset_dir, cfg, tag)
    ws = meta["window_stats"]["window_seconds"]
    result = {"dataset": meta["dataset"], "model_id": scorer.model_id, "model_version": scorer.version,
              "tier_thresholds": scorer.tier_thr, "splits": {}}
    for split in ("val", "test"):
        d = df[(df["split"] == split) & df["label"].notna()].reset_index(drop=True)
        scored = scorer.score_features(d)
        y = d["label"].astype(int).to_numpy()
        result["splits"][split] = {
            "role": "calibration data (optimistic)" if split == "val" else "held-out",
            "n_windows": int(len(y)), "n_attack": int(y.sum()),
            "configs": _config_metrics(scored, y, _span_days(d, ws)),
            "tiers": _tier_table(scored, y), "model_tiers": _tier_table(scored, y, "model_tier"), "rules": _rule_table(scored, y)}
    _, off = apply_rules(df.head(1), scorer.rules_cfg)
    result["rules_unavailable"] = off
    result["tier_calibration"] = load_json(dataset_dir / "models" / tag / "scoring.json")["tier_calibration"]
    result["caveats"] = [
        "Rule thresholds and tier floors are config choices, not tuned on test; validation numbers are optimistic because tiers were calibrated there.",
        "Tiers are validation precision bands; with few validation attacks the HIGH/CRITICAL cut-points are noisy.",
        "Unreachable tiers (infinite threshold) can still be reached through a rule floor.",
    ]
    if meta["dataset"] == "synthetic":
        result["caveats"].append("Synthetic data: rule hits and the threat_intel feature are partly circular with how attacks were generated.")
    root = Path(reports_root) if reports_root else PROJECT_ROOT / "reports"
    mdir, fdir = root / "metrics" / meta["dataset"] / tag, root / "figures" / meta["dataset"] / tag
    dump_json(result, mdir / "scoring.json")
    plots.plot_scoring(result["splits"]["test"], fdir / "scoring.png")
    return {"metrics_dir": str(mdir), "figures_dir": str(fdir), "result": result}
