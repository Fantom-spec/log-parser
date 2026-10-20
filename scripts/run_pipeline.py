#!/usr/bin/env python3
"""Synthetic end-to-end: generate -> prepare -> train -> evaluate -> tiers/rules -> SQLite -> dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.services.data import generate_synthetic_dataset
from logad.dashboard import build_dashboard
from logad.services.evaluate import evaluate_models
from logad.services.ingest import ingest_dataset
from logad.services.scoring import calibrate_scoring, evaluate_scoring
from logad.services.prepare import prepare_dataset
from logad.services.train import train_models


def main():
    cfg = load_config()
    gen = generate_synthetic_dataset(cfg)
    out = resolve_path(cfg["paths"]["processed_dir"]) / "synthetic"
    prepare_dataset(gen["files"]["log"], out, cfg, gen["files"]["labels"], "synthetic")
    train_models(out, cfg)
    print(evaluate_models(out, cfg)["comparison"].round(4).to_string(index=False))
    calibrate_scoring(out, cfg)
    evaluate_scoring(out, cfg)
    ingest_dataset(out, cfg)
    print("dashboard:", build_dashboard(cfg, "synthetic"))


if __name__ == "__main__":
    main()
