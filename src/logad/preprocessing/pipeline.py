"""Dataset loading and the training-only preprocessing pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_dir):
    """-> (windows DataFrame, feature_names, meta). feature_names.json is the source of truth."""
    d = Path(dataset_dir)
    # round_trip: the default C parser is not exact, which would make results depend on how the CSV was rewritten
    df = pd.read_csv(d / "window_features.csv", float_precision="round_trip")
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    names = json.loads((d / "feature_names.json").read_text())["feature_names"]
    meta = json.loads((d / "meta.json").read_text())
    missing = [f for f in names if f not in df.columns]
    if missing:
        raise ValueError(f"features listed in feature_names.json but missing from the table: {missing}")
    if df[names].isna().any().any():
        raise ValueError("feature table contains NaN; Phase 1 should never emit them")
    return df, names, meta


def training_mask(df: pd.DataFrame) -> pd.Series:
    """Training rows: the train split, normal windows only when labels exist (unlabeled kept)."""
    return (df["split"] == "train") & ~(df["label"] == 1)


class Preprocessor:
    """log1p on count features -> median imputation -> standardisation. Fitted on train rows only.

    transform_raw  : untouched features (Isolation Forest: trees are invariant to monotone scaling,
                     and the rule baseline needs the original counts).
    transform_scaled: for distance/kernel models (One-Class SVM, LOF).
    """

    def __init__(self, features, log1p_features=()):
        self.features = list(features)
        logs = set(log1p_features)
        self.log_idx = [i for i, f in enumerate(self.features) if f in logs]
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.fitted = False

    def _log(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        if self.log_idx:
            if (X[:, self.log_idx] < 0).any():
                raise ValueError("negative value in a log1p feature")
            X[:, self.log_idx] = np.log1p(X[:, self.log_idx])
        return X

    def transform_raw(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.features].to_numpy(dtype=float)

    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        if df_train.empty:
            raise ValueError("no training rows to fit the preprocessor")
        X = self._log(self.transform_raw(df_train))
        self.scaler.fit(self.imputer.fit_transform(X))
        self.fitted = True
        return self

    def transform_scaled(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")
        return self.scaler.transform(self.imputer.transform(self._log(self.transform_raw(df))))
