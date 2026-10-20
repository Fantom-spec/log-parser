#!/usr/bin/env python3
"""Score a raw SSH log with a trained model: writes detections (and parsed events) to SQLite, prints top windows."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.services.ingest import score_log_to_db


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log"); ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic",
                    help="trained dataset whose model, baselines and feature list are used")
    ap.add_argument("--dataset-dir"); ap.add_argument("--tag", default="all"); ap.add_argument("--year", type=int)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = a.dataset_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    r = score_log_to_db(d, cfg, a.log, a.tag, year=a.year)
    s = r["scored"]
    print(f"run {r['run_id']} -> {r['db_path']}; windows={len(s)}; tiers={s['tier'].value_counts().to_dict()}")
    for _, x in s.sort_values('score', ascending=False).head(a.top).iterrows():
        print(f"{x['tier']:8s} {x['score']:.3f} {x['entity']:>14s} {x['window_start']}  {'; '.join(x['rules'])}")


if __name__ == "__main__":
    main()
