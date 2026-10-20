#!/usr/bin/env python3
"""H1: per-user vs global baseline, paired on identical windows/splits/seeds."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.services.experiments import run_h1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic"); ap.add_argument("--dataset-dir")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = a.dataset_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    res = run_h1(d, cfg)
    print("features that differ between arms:", res["features_that_differ"])
    for m, e in res["models"].items():
        dl = e["delta"]
        print(f"{m:8s} PR-AUC per-user={e['per_user']['pr_auc']:.3f} global={e['global']['pr_auc']:.3f} "
              f"delta={dl['delta_pr_auc']:+.3f} CI={[round(x,3) for x in dl['delta_pr_auc_ci']]} -> {dl['verdict_pr_auc']}")
    print("\nmetrics:", res["metrics_dir"], "\nfigures:", res["figures_dir"])


if __name__ == "__main__":
    main()
