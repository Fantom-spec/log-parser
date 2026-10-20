#!/usr/bin/env python3
"""Raw log (+ optional labels) -> data/processed/<dataset>/ feature table and metadata."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.services.prepare import prepare_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--log"); ap.add_argument("--labels")
    ap.add_argument("--group-by", choices=["user", "src_ip"]); ap.add_argument("--year", type=int)
    ap.add_argument("--no-threat-intel", action="store_true"); ap.add_argument("--out-dir")
    a = ap.parse_args()
    over = {"dataset": {"name": a.dataset}, "windowing": {}, "features": {}}
    if a.group_by: over["windowing"]["group_by"] = a.group_by
    if a.year: over["dataset"]["log_year"] = a.year
    if a.no_threat_intel: over["features"]["threat_intel_enabled"] = False
    cfg = load_config(a.config, over)
    syn = resolve_path(cfg["paths"]["synthetic_dir"])
    log = a.log or (syn / "auth.log" if a.dataset == "synthetic" else None)
    labels = a.labels or (syn / "labels.csv" if a.dataset == "synthetic" and not a.log else None)
    if log is None:
        ap.error("--log is required for non-synthetic datasets")
    out = a.out_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    meta = prepare_dataset(log, out, cfg, labels, a.dataset)
    print(json.dumps({k: meta[k] for k in ("dataset", "n_windows", "features", "warnings")}, indent=2))
    print(json.dumps(meta["split"]["per_split"], indent=2))


if __name__ == "__main__":
    main()
