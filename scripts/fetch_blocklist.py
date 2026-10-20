#!/usr/bin/env python3
"""Download a static blocklist snapshot (e.g. ipsum) and record url + sha256 + date in the manifest."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logad.config import load_config
from logad.services.fetch import fetch_blocklist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--url"); ap.add_argument("--dest"); ap.add_argument("--source", default="ipsum")
    a = ap.parse_args()
    try:
        print(json.dumps(fetch_blocklist(load_config(a.config), a.url, a.dest, source=a.source), indent=2))
    except Exception as e:                                           # noqa: BLE001
        sys.exit(f"fetch failed: {e}")


if __name__ == "__main__":
    main()
