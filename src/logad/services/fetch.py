"""Service: download a Loghub log or a blocklist snapshot, record url + sha256 + date in the manifest.

This is the ONLY place that downloads data, and only when you run it explicitly. Training and evaluation
never touch the network. http(s) and file:// are accepted (file:// is what the tests use).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from logad.config import resolve_path


def download(url: str, dest: Path, max_bytes: int, timeout: float = 30.0) -> dict:
    if urlparse(url).scheme not in ("http", "https", "file"):
        raise ValueError("only http, https and file URLs are allowed")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h, n = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as fh:
            for chunk in iter(lambda: r.read(1 << 16), b""):
                n += len(chunk)
                if n > max_bytes:
                    raise ValueError(f"download exceeds {max_bytes} bytes; aborted")
                h.update(chunk)
                fh.write(chunk)
        if n == 0:
            raise ValueError("downloaded file is empty")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(dest), "bytes": n, "sha256": h.hexdigest()}


def fetch_loghub(cfg: dict, url: str | None = None, dest=None) -> dict:
    f = cfg["fetch"]
    dest = Path(dest) if dest else resolve_path(cfg["real"]["log_path"])
    info = download(url or f["loghub_url"], dest, f["max_bytes"], f["timeout_seconds"])
    info["url"] = url or f["loghub_url"]
    return info


def fetch_blocklist(cfg: dict, url: str | None = None, dest=None, manifest=None, source: str = "ipsum") -> dict:
    """Download a snapshot and write/update its manifest entry (url, sha256, date) so results reproduce."""
    f = cfg["fetch"]
    dest = Path(dest) if dest else resolve_path(cfg["real"]["blocklist_path"])
    manifest = Path(manifest) if manifest else resolve_path(cfg["real"]["manifest_path"])
    info = download(url or f["blocklist_url"], dest, f["max_bytes"], f["timeout_seconds"])
    m = json.loads(manifest.read_text()) if manifest.exists() else {"files": []}
    m["files"] = [e for e in m["files"] if e.get("file") != dest.name]
    m["files"].append({"file": dest.name, "source": source, "url": url or f["blocklist_url"],
                       "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "sha256": info["sha256"],
                       "note": "Snapshot taken on the date shown; old logs vs a current blocklist can match poorly."})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    info["manifest"] = str(manifest)
    return info
