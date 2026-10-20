#!/usr/bin/env python3
"""Start the local app: http://127.0.0.1:8000  (stdlib only; no extra packages needed)."""
import argparse, sys, threading, webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from logad.app.server import make_server
from logad.app.service import AppState
from logad.config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--tag", default="all"); ap.add_argument("--config"); ap.add_argument("--open", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    state = AppState(load_config(a.config), a.dataset, a.tag)
    try:
        srv = make_server(state, port=a.port, verbose=a.verbose)
    except OSError as e:
        sys.exit(f"cannot bind 127.0.0.1:{a.port} ({e}); try --port 8001")
    url = f"http://127.0.0.1:{a.port}"
    print(f"logad running at {url}   (Ctrl+C to stop)")
    print("model ready" if state.ready() else "no trained model found: open the Pipeline tab and run 'full_demo'")
    if a.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
