"""Service: score a prepared dataset and persist everything to SQLite."""
from __future__ import annotations

from pathlib import Path

from logad.config import resolve_path
from logad.io import load_json
from logad.models.registry import ModelRegistry
from logad.preprocessing.pipeline import load_dataset
from logad.scoring.scorer import Scorer
from logad.storage.db import Database


def ingest_dataset(dataset_dir, cfg: dict, tag: str = "all", db_path=None) -> dict:
    """Window features, model registry and detections (all windows, with split + label) -> SQLite."""
    dataset_dir = Path(dataset_dir)
    df, names, meta = load_dataset(dataset_dir)
    scorer = Scorer(dataset_dir, cfg, tag)
    scored = scorer.score_features(df)
    reg = ModelRegistry.load(dataset_dir / "models" / tag)
    path = db_path or resolve_path(cfg["paths"]["db_path"])
    with Database(path) as db:
        ds = meta["dataset"]
        db.put_dataset(ds, names, meta["window_stats"]["group_by"], meta["window_stats"]["window_seconds"],
                       {"n_windows": meta["n_windows"], "split": meta["split"]["per_split"]})
        db.put_window_features(ds, df, names)
        db.put_models(ds, tag, reg.entries)
        run_id = db.start_run("replay_dataset", ds, scorer.model_id, scorer.version, {"tag": tag})
        db.put_detections(run_id, scored)
        counts = db.counts()
    return {"db_path": str(path), "run_id": run_id, "counts": counts}


def score_log_to_db(dataset_dir, cfg: dict, log_path, tag: str = "all", db_path=None, year=None,
                    name: str | None = None, scorer: Scorer | None = None) -> dict:
    """Score an uploaded/raw log with a trained model; store detections and the parsed events."""
    from logad.parsing.ssh_parser import parse_file
    dataset_dir = Path(dataset_dir)
    scorer = scorer or Scorer(dataset_dir, cfg, tag)
    scored, info = scorer.score_log(log_path, year)
    c = scorer.meta["config"]
    events, _ = parse_file(log_path, year or c["dataset"]["log_year"], c["dataset"]["tz_offset_hours"],
                           c["parser"]["failure_methods"], c["parser"].get("dedupe_exact_lines", False))
    path = db_path or resolve_path(cfg["paths"]["db_path"])
    label = name or f"{scorer.meta['dataset']}:scored:{Path(log_path).name}"
    with Database(path) as db:
        db.put_dataset(label, scorer.features, scorer.group_by, c["windowing"]["window_seconds"],
                       {"scored_with": scorer.model_id, "parse_report": info["parse_report"]})
        run_id = db.start_run("score_log", label, scorer.model_id, scorer.version, {"log": str(log_path)})
        db.put_detections(run_id, scored)
        db.put_events(run_id, events)
    return {"db_path": str(path), "run_id": run_id, "scored": scored, "info": info}
