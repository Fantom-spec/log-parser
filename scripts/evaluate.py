#!/usr/bin/env python3
"""Held-out test evaluation with thresholds frozen at training time; writes metrics + figures."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from logad.config import load_config, resolve_path
from logad.services.evaluate import evaluate_models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--dataset-dir"); ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = a.dataset_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    res = evaluate_models(d, cfg, tag=a.tag)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res["comparison"][["model", "threshold", "precision", "recall", "f1", "pr_auc",
                                 "false_positive_rate", "alerts_per_day"]].round(4).to_string(index=False))
    print("\nmetrics:", res["metrics_dir"], "\nfigures:", res["figures_dir"])


if __name__ == "__main__":
    main()
