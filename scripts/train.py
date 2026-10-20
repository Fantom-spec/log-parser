#!/usr/bin/env python3
"""Fit preprocessing + models on train rows; calibrate thresholds on validation rows."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.services.train import train_models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--dataset-dir"); ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = a.dataset_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    print(json.dumps(train_models(d, cfg, tag=a.tag), indent=2))


if __name__ == "__main__":
    main()
