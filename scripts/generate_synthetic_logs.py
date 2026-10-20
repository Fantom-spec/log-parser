#!/usr/bin/env python3
"""Generate synthetic SSH logs + labels + planted blocklist snapshot."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config
from logad.services.data import generate_synthetic_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--seed", type=int)
    ap.add_argument("--days", type=int); ap.add_argument("--year", type=int)
    ap.add_argument("--start", help="MM-DD start date (default from config)")
    ap.add_argument("--out-dir"); ap.add_argument("--threat-intel-dir")
    a = ap.parse_args()
    over = {"synthetic": {}, "dataset": {}}
    if a.days: over["synthetic"]["days"] = a.days
    if a.start: over["synthetic"]["start_month_day"] = a.start
    if a.year: over["dataset"]["log_year"] = a.year
    cfg = load_config(a.config, over)
    print(json.dumps(generate_synthetic_dataset(cfg, a.out_dir, a.threat_intel_dir, a.seed), indent=2))


if __name__ == "__main__":
    main()
