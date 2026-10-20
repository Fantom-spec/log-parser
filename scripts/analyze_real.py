#!/usr/bin/env python3
"""Real, unlabeled log (e.g. Loghub OpenSSH): prepare -> train unlabeled -> analysis report. No P/R/F1."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config, resolve_path
from logad.services.realdata import run_real_analysis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--log"); ap.add_argument("--dataset"); ap.add_argument("--year", type=int)
    ap.add_argument("--group-by", choices=["user", "src_ip"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = a.log or resolve_path(cfg["real"]["log_path"])
    if not Path(log).exists():
        sys.exit(f"log not found: {log}  (put the file there or run scripts/fetch_loghub.py)")
    r = run_real_analysis(log, cfg, a.dataset, a.year, a.group_by)
    print(f"{r['dataset']}: {r['n_windows']} windows, grouped by {r['group_by']}; parse coverage (sshd lines) {r['parse'].get('coverage_sshd_lines')}")
    print("tiers (all windows):", r["tiers"]["all"])
    print("rule hits:", r["rules"]["hits_total"], "| unavailable:", r["rules"]["unavailable"])
    print("blocklist overlap:", json.dumps(r["blocklist_overlap"])[:200])
    cd = r["cross_domain"]
    print("cross-domain:", {k: cd[k] for k in ("windows", "tiers", "share_flagged")} if cd.get("available") else cd)
    print("top windows:")
    for w in r["top_windows"][:5]:
        print(f"  {w['tier']:8s} {w['score']:.3f} {w['entity']:>18s} {w['window_start']} {w['rules']} {w['evidence'][:1]}")


if __name__ == "__main__":
    main()
