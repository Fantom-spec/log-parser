"""Common model interface: every model returns scores where HIGHER = MORE ANOMALOUS.

Scales differ between algorithms, so scores are never compared across models directly;
each model gets its own validation-calibrated threshold.
"""
from __future__ import annotations

import numpy as np


class AnomalyModel:
    family = "base"
    input_kind = "raw"          # "raw" or "scaled": which matrix the model consumes

    def fit(self, X: np.ndarray) -> "AnomalyModel":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    @staticmethod
    def _check(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if not np.isfinite(s).all():
            raise ValueError("model produced non-finite scores")
        return s
