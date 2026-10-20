"""Service: train on an UNLABELED dataset (real logs). No labels, so no calibrated precision.

Isolation Forest is fitted on all training windows. Tier cut-points are quantiles of the VALIDATION
score distribution, so tiers mean "unusual relative to this log" and flag a fixed share of windows by
construction. The artefact layout matches the labelled path, so the Scorer works unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from logad.io import dump_json
from logad.models.iforest import IsolationForestModel
from logad.models.registry import ModelRegistry
from logad.preprocessing.pipeline import Preprocessor, load_dataset
from logad.scoring.explain import feature_profile


def train_unlabeled(dataset_dir, cfg: dict, tag: str = "unlabeled") -> dict:
    dataset_dir = Path(dataset_dir)
    df, names, meta = load_dataset(dataset_dir)
    fit_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    if len(fit_df) < 10 or len(val_df) < 5:
        raise ValueError(f"too few windows to train/calibrate (train={len(fit_df)}, val={len(val_df)})")
    seed = cfg["models"]["seeds"][0]
    pre = Preprocessor(names, cfg["preprocessing"]["log1p_features"]).fit(fit_df)
    model = IsolationForestModel(seed, **cfg["models"]["iforest"]).fit(pre.transform_raw(fit_df))
    s_val = model.score(pre.transform_raw(val_df))
    q = cfg["unlabeled"]["tier_quantiles"]
    thr = {t: float(np.quantile(s_val, q[t])) for t in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}

    mdir = dataset_dir / "models" / tag
    mdir.mkdir(parents=True, exist_ok=True)
    for old in mdir.glob("*.joblib"):
        old.unlink()
    joblib.dump(pre, mdir / "preprocessor.joblib")
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(pd.concat([fit_df, val_df])[names].astype(float), index=False).to_numpy().tobytes())
    mid = f"iforest_seed{seed}"
    reg = ModelRegistry(mdir)
    entry = reg.register(mid, model, {
        "family": "iforest", "params": model.params(), "seed": seed, "features": names, "fit_data_sha256": h.hexdigest(),
        "input_kind": "raw", "tag": tag, "train_rows": int(len(fit_df)), "val_rows": int(len(val_df)),
        "log1p_features": [], "sklearn_version": sklearn.__version__, "threshold": thr["LOW"],
        "threshold_strategy": "validation_score_quantile", "val_metrics": None})
    reg.save()
    dump_json({mid: {"threshold": thr["LOW"], "strategy": "validation_score_quantile", "quantile": q["LOW"]}}, mdir / "thresholds.json")
    sc = cfg["scoring"]
    dump_json({
        "model_id": mid, "model_version": entry["version"], "tier_thresholds": thr,
        "tier_calibration": {t: {"quantile": q[t], "basis": "validation score distribution (unlabeled)"} for t in thr},
        "rules": sc["rules"], "rule_floors": sc["rule_floors"], "evidence_top_k": sc["evidence_top_k"],
        "profile": feature_profile(fit_df, names),
        "note": "UNLABELED: tiers are score quantiles, meaning unusual for this log, not attack probability"}, mdir / "scoring.json")
    return {"models_dir": str(mdir), "model_id": mid, "tier_thresholds": thr, "train_rows": int(len(fit_df)),
            "val_rows": int(len(val_df))}
