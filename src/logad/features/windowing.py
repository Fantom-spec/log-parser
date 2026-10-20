"""Tumbling-window assignment. One sample = one (entity, window_start)."""
from __future__ import annotations

import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def assign_windows(events: pd.DataFrame, window_seconds: int, group_by: str):
    """Attach `entity`, `window_start`, `ts_sec`. Events with no entity are counted, not hidden.

    group_by='user'   -> entity = username
    group_by='src_ip' -> entity = src_ip
    Windows exist only where at least one attributed event exists (groupby semantics).
    """
    if group_by not in ("user", "src_ip"):
        raise ValueError("group_by must be 'user' or 'src_ip'")
    col = "username" if group_by == "user" else "src_ip"
    ev = events.copy()
    ev["entity"] = ev[col]
    attributed = ev["entity"].notna() & (ev["entity"].astype(str) != "")
    stats = {
        "group_by": group_by,
        "window_seconds": int(window_seconds),
        "events_total": int(len(ev)),
        "unattributed_events": int((~attributed).sum()),
        "unattributed_weighted": int(ev.loc[~attributed, "repeat"].sum()),
    }
    ev = ev[attributed].copy()
    ev["entity"] = ev["entity"].astype(str)
    ev["window_start"] = ev["ts"].dt.floor(f"{int(window_seconds)}s")
    ev["ts_sec"] = (ev["ts"] - _EPOCH).dt.total_seconds()
    ev = ev.sort_values(["ts", "line_no"], kind="stable").reset_index(drop=True)
    return ev, stats
