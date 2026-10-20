"""Isolation Forest (primary model)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AnomalyModel


class IsolationForestModel(AnomalyModel):
    family = "iforest"
    input_kind = "raw"

    def __init__(self, seed: int = 42, n_estimators: int = 200, max_samples: int = 256):
        self.seed, self.n_estimators, self.max_samples = int(seed), int(n_estimators), int(max_samples)
        self.est = None

    def fit(self, X):
        self.est = IsolationForest(
            n_estimators=self.n_estimators, max_samples=min(self.max_samples, len(X)),
            random_state=self.seed, n_jobs=-1).fit(X)
        return self

    def score(self, X):
        return self._check(-self.est.score_samples(X))

    def params(self):
        return {"n_estimators": self.n_estimators, "max_samples": self.max_samples,
                "random_state": self.seed}
