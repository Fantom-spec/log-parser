"""Service: fit preprocessing + all models on TRAIN rows, calibrate thresholds on VALIDATION rows.

Test rows are never read for fitting or calibration: only train and validation rows are passed on.
Artefacts go to <dataset_dir>/models/<tag>/ (joblib models, registry.json, thresholds.json).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from logad.evaluation.metrics import calibrate_threshold, threshold_metrics
from logad.io import dump_json
from logad.models.baselines import LOFModel, OneClassSVMModel, RuleBaseline
from logad.models.iforest import IsolationForestModel
from logad.models.registry import ModelRegistry
from logad.preprocessing.pipeline import Preprocessor, load_dataset, training_mask


def _span_days(df: pd.DataFrame, window_seconds: int) -> float:
    return ((df["window_start"].max() - df["window_start"].min()).total_seconds() + window_seconds) / 86400.0


def build_models(cfg: dict, features: list[str]) -> list[tuple[str, object]]:
    m, out = cfg["models"], []
    rf = m["rule"]["feature"]
    if rf in features:                                   # rule baseline needs its feature (ablations may remove it)
        out.append(("rule", RuleBaseline(rf, features.index(rf))))
    for seed in m["seeds"]:
        out.append((f"iforest_seed{seed}", IsolationForestModel(seed, **m["iforest"])))
    out.append(("ocsvm", OneClassSVMModel(**m["ocsvm"])))
    out.append(("lof", LOFModel(**m["lof"])))
    return out


def train_models(dataset_dir, cfg: dict, features=None, tag: str = "all") -> dict:
    dataset_dir = Path(dataset_dir)
    df, names, meta = load_dataset(dataset_dir)
    if not meta["has_labels"]:
        raise ValueError("this dataset has no labels: threshold calibration and metrics need a labelled "
                         "validation split. Unlabeled real logs get score-distribution analysis instead.")
    feats = list(features) if features else list(names)
    unknown = [f for f in feats if f not in names]
    if unknown:
        raise ValueError(f"unknown features requested: {unknown}")

    fit_df = df[training_mask(df)]
    val_df = df[(df["split"] == "val") & df["label"].notna()]           # test rows are never touched here
    if len(val_df) != int((df["split"] == "val").sum()):
        pass  # unlabeled validation windows (if any) are excluded from calibration
    window_seconds = meta["window_stats"]["window_seconds"]
    val_span = _span_days(val_df, window_seconds)

    pre = Preprocessor(feats, cfg["preprocessing"]["log1p_features"]).fit(fit_df)
    X = {"train": {"raw": pre.transform_raw(fit_df), "scaled": pre.transform_scaled(fit_df)},
         "val": {"raw": pre.transform_raw(val_df), "scaled": pre.transform_scaled(val_df)}}
    y_val = val_df["label"].astype(int).to_numpy()

    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(pd.concat([fit_df, val_df])[feats + ["label"]].astype(float), index=False).to_numpy().tobytes())
    h.update(",".join(feats).encode())
    fit_sha = h.hexdigest()

    models_dir = dataset_dir / "models" / tag
    models_dir.mkdir(parents=True, exist_ok=True)
    for old in models_dir.glob("*.joblib"):
        old.unlink()
    joblib.dump(pre, models_dir / "preprocessor.joblib")
    reg = ModelRegistry(models_dir)
    thresholds, fit_times = {}, {}
    for model_id, model in build_models(cfg, feats):
        t0 = time.perf_counter()
        model.fit(X["train"][model.input_kind])
        fit_times[model_id] = time.perf_counter() - t0
        s_val = model.score(X["val"][model.input_kind])
        cal = calibrate_threshold(y_val, s_val)
        thresholds[model_id] = cal
        reg.register(model_id, model, {
            "family": model.family, "params": model.params(), "seed": getattr(model, "seed", None),
            "features": feats, "fit_data_sha256": fit_sha, "input_kind": model.input_kind, "tag": tag,
            "train_rows": int(len(fit_df)), "val_rows": int(len(val_df)),
            "log1p_features": [feats[i] for i in pre.log_idx], "sklearn_version": sklearn.__version__,
            "threshold": cal["threshold"], "threshold_strategy": cal["strategy"],
            "val_metrics": threshold_metrics(y_val, s_val, cal["threshold"], val_span),
        })
    reg.save()
    dump_json(thresholds, models_dir / "thresholds.json")
    dump_json(fit_times, models_dir / "fit_times.json")        # wall-clock: kept out of reproducible files
    return {"models_dir": str(models_dir), "model_ids": list(reg.entries), "features": feats,
            "train_rows": int(len(fit_df)), "val_rows": int(len(val_df)), "tag": tag}
