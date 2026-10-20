"""Chronological train/val/test split (never shuffled)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(window_starts: pd.Series, train: float, val: float, test: float):
    """Return (split array of 'train'|'val'|'test', cutoffs dict).

    Cut points are placed on window_start boundaries, so windows that share a start time
    (different entities) always land in the same split. Tumbling windows never overlap in
    time, so no near-duplicate window can straddle a boundary.
    """
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    n = len(window_starts)
    if n < 3:
        raise ValueError("need at least 3 windows to split")
    s = np.sort(window_starts.to_numpy())
    val_start = s[int(round(n * train))] if int(round(n * train)) < n else None
    test_start = s[int(round(n * (train + val)))] if int(round(n * (train + val))) < n else None
    ws = window_starts.to_numpy()
    split = np.full(n, "train", dtype=object)
    if val_start is not None:
        split[ws >= val_start] = "val"
    if test_start is not None:
        split[ws >= test_start] = "test"
    if (split == "train").sum() == 0:
        raise ValueError("training split is empty")
    cut = {
        "val_starts_at": str(pd.Timestamp(val_start)) if val_start is not None else None,
        "test_starts_at": str(pd.Timestamp(test_start)) if test_start is not None else None,
        "fractions": {"train": train, "val": val, "test": test},
    }
    return split, cut
