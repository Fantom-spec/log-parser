"""Comparison models: One-Class SVM, Local Outlier Factor (novelty mode), rule baseline."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .base import AnomalyModel


class OneClassSVMModel(AnomalyModel):
    family = "ocsvm"
    input_kind = "scaled"

    def __init__(self, kernel="rbf", nu=0.05, gamma="scale"):
        self.kernel, self.nu, self.gamma = kernel, float(nu), gamma
        self.est = None

    def fit(self, X):
        self.est = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma).fit(X)
        return self

    def score(self, X):
        return self._check(-self.est.decision_function(X))

    def params(self):
        return {"kernel": self.kernel, "nu": self.nu, "gamma": self.gamma}


class LOFModel(AnomalyModel):
    """novelty=True so unseen validation/test samples can be scored against the training set."""
    family = "lof"
    input_kind = "scaled"

    def __init__(self, n_neighbors=20):
        self.n_neighbors = int(n_neighbors)
        self.est = None

    def fit(self, X):
        self.est = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(X) - 1), novelty=True).fit(X)
        return self

    def score(self, X):
        return self._check(-self.est.score_samples(X))

    def params(self):
        return {"n_neighbors": self.n_neighbors, "novelty": True}


class RuleBaseline(AnomalyModel):
    """Score = one raw feature (default failed_count). The alert threshold is tuned on validation."""
    family = "rule"
    input_kind = "raw"

    def __init__(self, feature="failed_count", feature_index=None):
        self.feature, self.feature_index = feature, feature_index

    def fit(self, X):
        if self.feature_index is None:
            raise ValueError("RuleBaseline needs the column index of its feature")
        return self

    def score(self, X):
        return self._check(X[:, self.feature_index])

    def params(self):
        return {"feature": self.feature}
