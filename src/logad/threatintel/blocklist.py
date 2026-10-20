"""Static IP blocklist snapshots. Offline only; hash-verified against a manifest."""
from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import dataclass, field
from pathlib import Path


class BlocklistError(Exception):
    pass


@dataclass
class Blocklist:
    ips: frozenset
    sha256: str
    path: str
    verified: bool
    manifest_entry: dict | None = None
    warnings: list = field(default_factory=list)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_blocklist(path, manifest_path=None) -> Blocklist:
    """Read ipsum/plain-text snapshot (first token per line is the IP; '#' = comment).

    If a manifest exists it must list this file with a matching sha256, else BlocklistError.
    """
    path = Path(path)
    if not path.exists():
        raise BlocklistError(f"blocklist snapshot not found: {path}")
    digest = sha256_file(path)
    entry, verified, warns = None, False, []
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        entry = next((e for e in manifest.get("files", []) if e.get("file") == path.name), None)
        if entry is None:
            raise BlocklistError(f"{path.name} is not listed in the manifest")
        if entry.get("sha256") != digest:
            raise BlocklistError(f"sha256 mismatch for {path.name}: file changed since snapshot")
        verified = True
    else:
        warns.append("no manifest found; snapshot hash not verified")
    ips = set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ips.add(str(ipaddress.ip_address(line.split()[0])))
            except ValueError:
                continue
    return Blocklist(frozenset(ips), digest, str(path), verified, entry, warns)
