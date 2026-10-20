#!/usr/bin/env python3
"""Download the Loghub OpenSSH log (explicit, one-time; training/evaluation never use the network)."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config
from logad.services.fetch import fetch_loghub


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--url"); ap.add_argument("--dest")
    a = ap.parse_args()
    try:
        print(json.dumps(fetch_loghub(load_config(a.config), a.url, a.dest), indent=2))
    except Exception as e:                                           # noqa: BLE001
        sys.exit(f"fetch failed: {e}")


if __name__ == "__main__":
    main()
