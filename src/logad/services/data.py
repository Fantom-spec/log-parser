"""Service: generate synthetic data. Shared by scripts, tests and (later) the API."""
from __future__ import annotations

from logad.config import resolve_path
from logad.synthetic import generate_synthetic


def generate_synthetic_dataset(cfg: dict, out_dir=None, threat_intel_dir=None, seed=None) -> dict:
    return generate_synthetic(
        cfg,
        out_dir or resolve_path(cfg["paths"]["synthetic_dir"]),
        threat_intel_dir or resolve_path(cfg["paths"]["threat_intel_dir"]),
        seed=seed,
    )
