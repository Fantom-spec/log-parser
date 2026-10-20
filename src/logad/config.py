"""Configuration loading: YAML file deep-merged over built-in defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict = {
    "dataset": {"name": "synthetic", "log_year": 2025, "tz_offset_hours": 0},
    "windowing": {"window_seconds": 300, "group_by": "user"},
    "parser": {"failure_methods": ["password", "keyboard-interactive/pam", "keyboard-interactive"],
               "dedupe_exact_lines": False},
    "features": {
        "success_after_failed_min": 5,
        "off_hours_start": 8,
        "off_hours_end": 19,
        "min_history_windows": 5,
        "z_clip": 20.0,
        "geoip_enabled": False,
        "threat_intel_enabled": True,
    },
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "threat_intel": {
        "blocklist_path": "data/threat_intel/blocklist_synthetic.txt",
        "manifest_path": "data/threat_intel/manifest.json",
    },
    "synthetic": {
        "seed": 42,
        "start_month_day": "03-03",
        "days": 14,
        "n_users": 20,
        "incidents_per_type": 12,
        "typo_prob": 0.05,
        "benign_new_ip_prob": 0.02,
        "host": "bastion01",
        "n_blocklist_decoys": 200,
        "attacker_blocklisted_fraction": 0.5,
    },
    "preprocessing": {
        "log1p_features": [
            "failed_count", "max_failed_burst_60s", "consecutive_failed_max", "distinct_hosts_failed",
            "event_count", "distinct_event_types", "distinct_src_ips", "new_ip_count",
            "max_events_per_ip", "distinct_users", "blocklisted_ip_count", "blocklisted_event_count"],
    },
    "models": {
        "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        "iforest": {"n_estimators": 200, "max_samples": 256},
        "ocsvm": {"kernel": "rbf", "nu": 0.05, "gamma": "scale"},
        "lof": {"n_neighbors": 20},
        "rule": {"feature": "failed_count"},
    },
    "scoring": {
        "model_id": "iforest_seed42",
        "tier_precision_targets": {"MEDIUM": 0.75, "HIGH": 0.90, "CRITICAL": 0.97},
        "tier_min_true_positives": 3,
        "rules": {"r1": {"enabled": True, "min_failed": 10, "min_distinct_ips": 5},
                  "r2": {"enabled": True},
                  "r3": {"enabled": True, "min_failed": 1}},
        "rule_floors": {"R1": "HIGH", "R2": "CRITICAL", "R3": "MEDIUM"},
        "evidence_top_k": 3,
    },
    "unlabeled": {"tier_quantiles": {"LOW": 0.95, "MEDIUM": 0.98, "HIGH": 0.99, "CRITICAL": 0.995},
                  "proxy_min_failed": 5},
    "real": {"dataset": "loghub", "log_path": "data/raw/OpenSSH.log", "year": 2015, "group_by": "src_ip",
             "blocklist_path": "data/threat_intel/ipsum.txt", "manifest_path": "data/threat_intel/manifest.json",
             "cross_domain_dataset": "synthetic"},
    "fetch": {"loghub_url": "https://raw.githubusercontent.com/logpai/loghub/master/OpenSSH/OpenSSH_2k.log",
              "blocklist_url": "https://raw.githubusercontent.com/stamparm/ipsum/master/ipsum.txt",
              "max_bytes": 200_000_000, "timeout_seconds": 30},
    "evaluation": {"threshold_strategy": "max_f1", "bootstrap_samples": 1000,
                   "bootstrap_seed": 0, "ci_level": 0.95},
    "paths": {
        "synthetic_dir": "data/synthetic",
        "processed_dir": "data/processed",
        "threat_intel_dir": "data/threat_intel",
        "db_path": "data/logad.db",
        "dashboard_dir": "reports/dashboard",
    },
}


def merge_config(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load config.yaml, then thresholds.yaml next to it (if present), over DEFAULTS, then apply `overrides`."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    for f in (p, p.parent / "thresholds.yaml"):
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                cfg = merge_config(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = merge_config(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["windowing"]["group_by"] not in ("user", "src_ip"):
        raise ValueError("windowing.group_by must be 'user' or 'src_ip'")
    if int(cfg["windowing"]["window_seconds"]) <= 0:
        raise ValueError("windowing.window_seconds must be positive")
    s = cfg["split"]
    if abs(s["train"] + s["val"] + s["test"] - 1.0) > 1e-9 or min(s.values()) <= 0:
        raise ValueError("split fractions must be positive and sum to 1")


def resolve_path(p: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
