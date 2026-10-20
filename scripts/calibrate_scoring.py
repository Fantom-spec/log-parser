#!/usr/bin/env python3
"""Calibrate risk tiers on validation, then evaluate model-only vs model+rules (val and test)."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.services.scoring import calibrate_scoring, evaluate_scoring


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--dataset-dir"); ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = a.dataset_dir or resolve_path(cfg["paths"]["processed_dir"]) / a.dataset
    cal = calibrate_scoring(d, cfg, a.tag)
    print("model:", cal["model_id"])
    for t, v in cal["tier_thresholds"].items():
        print(f"  {t:9s} threshold={v}  {cal['tier_calibration'][t]}")
    res = evaluate_scoring(d, cfg, a.tag)["result"]
    for split, s in res["splits"].items():
        print(f"\n[{split}] {s['role']}  windows={s['n_windows']} attacks={s['n_attack']}")
        for k, m in s["configs"].items():
            print(f"  {k:17s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['false_positive_rate']:.3f} alerts={m['n_alerts']}")
        print("  tiers:", {t: f"{v['attack_windows']}/{v['windows']}" for t, v in s["tiers"].items()})
        print("  rules:", {r: (v['hits'], v['attack_hits']) for r, v in s["rules"].items()})


if __name__ == "__main__":
    main()
