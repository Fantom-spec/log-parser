#!/usr/bin/env python3
"""Ingest a scored dataset into SQLite, then render the static dashboard (open the HTML file locally)."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.dashboard import build_dashboard
from logad.services.ingest import ingest_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--dataset-dir"); ap.add_argument("--tag", default="all"); ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = a.dataset_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    print("ingested:", ingest_dataset(d, cfg, a.tag))
    print("dashboard:", build_dashboard(cfg, a.dataset, a.tag, a.out))


if __name__ == "__main__":
    main()
