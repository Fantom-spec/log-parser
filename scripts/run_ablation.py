#!/usr/bin/env python3
"""Feature-family ablation (drop-one and only-one) with paired bootstrap deltas."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from logad.config import load_config, resolve_path
from logad.services.experiments import run_ablation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic"); ap.add_argument("--dataset-dir")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = a.dataset_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    res = run_ablation(d, cfg)
    rows = [{"variant": r["variant"], "k": r["n_features"],
             **{f"{m}_prauc": r["models"][m]["pr_auc"] for m in ("iforest", "ocsvm", "lof")},
             **{f"d_{m}": (r["delta_vs_all"].get(m) or {}).get("delta_pr_auc") for m in ("iforest", "ocsvm", "lof")}}
            for r in res["results"]]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nmetrics:", res["metrics_dir"], "\nfigures:", res["figures_dir"])


if __name__ == "__main__":
    main()
