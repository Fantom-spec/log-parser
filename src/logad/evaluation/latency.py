"""Scoring latency measurement (wall-clock; stored separately from reproducible metrics)."""
from __future__ import annotations

import time


def time_call(fn, repeats: int = 5) -> float:
    """Median wall-clock seconds over `repeats` calls."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def ms_per_1000(seconds: float, n: int) -> float:
    return seconds * 1000.0 / n * 1000.0 if n else float("nan")
