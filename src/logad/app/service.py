"""Application logic behind the HTTP API (no HTTP types here, so it is directly testable)."""
from __future__ import annotations

import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from logad.app.samples import DESCRIPTIONS, build_sample
from logad.config import PROJECT_ROOT, resolve_path
from logad.io import load_json
from logad.scoring.scorer import Scorer
from logad.services.ingest import score_log_to_db
from logad.storage.db import Database
from logad.threatintel.blocklist import BlocklistError, load_blocklist

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 300
FIGS = [("pr_curves.png", "{tag}", "Precision-recall, held-out test"),
        ("scoring.png", "{tag}", "Risk tiers and model + rules (test)"),
        ("ablation.png", "ablation", "Feature-family ablation"),
        ("h1.png", "h1", "H1: per-user vs global baseline")]
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


class ApiError(Exception):
    def __init__(self, msg, status=400, extra=None):
        super().__init__(msg)
        self.status, self.extra = status, extra or {}


@dataclass
class AppState:
    cfg: dict
    dataset: str = "synthetic"
    tag: str = "all"
    dataset_dir: Path | None = None
    db_path: Path | None = None
    reports_root: Path | None = None
    upload_dir: Path | None = None
    _scorer: Scorer | None = field(default=None, repr=False)
    _scorer_key: tuple = field(default=(), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir or resolve_path(self.cfg["paths"]["processed_dir"]) / self.dataset)
        self.db_path = Path(self.db_path or resolve_path(self.cfg["paths"]["db_path"]))
        self.reports_root = Path(self.reports_root or PROJECT_ROOT / "reports")
        self.upload_dir = Path(self.upload_dir or PROJECT_ROOT / "data" / "uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        Database(self.db_path).close()                                # create schema early

    @property
    def models_dir(self) -> Path:
        return self.dataset_dir / "models" / self.tag

    def ready(self) -> bool:
        need = [self.dataset_dir / "meta.json", self.dataset_dir / "feature_names.json", self.dataset_dir / "baselines.json",
                self.models_dir / "registry.json", self.models_dir / "scoring.json", self.models_dir / "preprocessor.joblib"]
        return all(p.exists() for p in need)

    def scorer(self) -> Scorer:
        if not self.ready():
            raise ApiError("No trained model yet. Open the Pipeline tab and run 'full_demo'.", 409)
        key = (self.models_dir / "scoring.json").stat().st_mtime_ns, (self.models_dir / "registry.json").stat().st_mtime_ns
        with self._lock:
            if self._scorer is None or key != self._scorer_key:
                self._scorer, self._scorer_key = Scorer(self.dataset_dir, self.cfg, self.tag), key
            return self._scorer


def _meta(state: AppState) -> dict:
    return load_json(state.dataset_dir / "meta.json")


def _blocklist(state: AppState):
    c = _meta(state)["config"]["threat_intel"]
    return load_blocklist(resolve_path(c["blocklist_path"]), resolve_path(c["manifest_path"]))


def status(state: AppState, jobs_busy: bool) -> dict:
    out = {"ready": state.ready(), "dataset": state.dataset, "tag": state.tag, "jobs_busy": jobs_busy}
    with Database(state.db_path) as db:
        out["counts"] = db.counts()
        out["has_replay_run"] = db.latest_run("replay_dataset", state.dataset) is not None
    if not out["ready"]:
        return out
    sc = load_json(state.models_dir / "scoring.json")
    meta = _meta(state)
    fn = load_json(state.dataset_dir / "feature_names.json")
    out.update({
        "model_id": sc["model_id"], "model_version": sc["model_version"], "tier_thresholds": sc["tier_thresholds"],
        "tier_calibration": sc["tier_calibration"], "group_by": fn["group_by"], "window_seconds": fn["window_seconds"],
        "n_features": len(fn["feature_names"]), "families": fn["families"], "n_windows": meta["n_windows"],
        "split": meta["split"]["per_split"], "has_labels": meta["has_labels"], "threat_intel": meta["threat_intel"],
        "rules": sc["rules"], "rule_floors": sc["rule_floors"]})
    return out


def results(state: AppState) -> dict:
    m = state.reports_root / "metrics" / state.dataset
    out = {"dataset": state.dataset}
    for key, p in (("models", m / state.tag / "metrics.json"), ("scoring", m / state.tag / "scoring.json"),
                   ("ablation", m / "ablation" / "ablation.json"), ("h1", m / "h1" / "h1.json")):
        out[key] = load_json(p) if p.exists() else None
    f = state.reports_root / "figures" / state.dataset
    out["figures"] = []
    for name, sub, title in FIGS:
        sub = sub.format(tag=state.tag)
        if (f / sub / name).exists():
            out["figures"].append({"title": title, "url": f"/figures/{state.dataset}/{sub}/{name}"})
    return out


def samples(state: AppState) -> dict:
    return {"samples": [{"name": n, "description": d} for n, d in DESCRIPTIONS.items()]}


def sample_text(state: AppState, name: str) -> str:
    sc = state.scorer()
    meta = _meta(state)
    try:
        bl = _blocklist(state).ips if sc.meta["threat_intel"]["enabled"] else frozenset()
    except BlocklistError:
        bl = frozenset()
    try:
        return build_sample(name, state.dataset_dir / "baselines.json", bl, meta["config"]["dataset"]["log_year"])
    except KeyError:
        raise ApiError(f"unknown sample {name!r}", 404)


def _row(r: dict, features: list[str]) -> dict:
    return {"entity": r["entity"], "window_start": r["window_start"], "score": round(float(r["score"]), 4),
            "model_tier": r["model_tier"], "tier": r["tier"], "rules": list(r["rules"]), "evidence": list(r["evidence"]),
            "features": {f: (round(float(r[f]), 4)) for f in features}}


def score_text(state: AppState, text: str, name: str | None = None, year: int | None = None) -> dict:
    if len(text.encode("utf-8", "replace")) > MAX_UPLOAD_BYTES:
        raise ApiError("log too large (limit 5 MB)", 413)
    if not text.strip():
        raise ApiError("empty log", 400)
    if year is not None and not (1970 <= int(year) <= 2100):
        raise ApiError("year out of range", 400)
    scorer = state.scorer()
    label = "upload:" + (_SAFE_NAME.sub("_", name or "pasted.log")[:60] or "pasted.log")
    fd, tmp = tempfile.mkstemp(suffix=".log", dir=state.upload_dir)          # server-chosen name, never client paths
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(text)
        try:
            r = score_log_to_db(state.dataset_dir, state.cfg, tmp, state.tag, state.db_path, year, label, scorer)
        except ValueError as exc:
            raise ApiError(str(exc), 422, {"hint": "Check the log is OpenSSH syslog text and the year is right."})
    finally:
        Path(tmp).unlink(missing_ok=True)
    s = r["scored"]
    flagged = s[s["tier"] != "NORMAL"].sort_values("score", ascending=False)
    return {"run_id": r["run_id"], "label": label, "windows": int(len(s)),
            "by_tier": {t: int((s["tier"] == t).sum()) for t in ("NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL")},
            "by_model_tier": {t: int((s["model_tier"] == t).sum()) for t in ("NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL")},
            "rules_hit": {k: int(sum(k in x for x in s["rules"])) for k in ("R1", "R2", "R3")},
            "parse": {k: r["info"]["parse_report"][k] for k in (
                "total_lines", "event_lines", "malformed_lines", "not_sshd_lines", "unsupported_sshd_lines",
                "coverage_sshd_lines", "year_rollovers")},
            "shown": int(min(len(flagged), MAX_ROWS)),
            "rows": [_row(x, scorer.features) for x in flagged.head(MAX_ROWS).to_dict("records")]}


def detections(state: AppState, params: dict) -> dict:
    run = params.get("run")
    with Database(state.db_path) as db:
        run_id = int(run) if run else db.latest_run("replay_dataset", state.dataset)
        if run_id is None:
            raise ApiError("no detections stored yet; run 'dashboard' in the Pipeline tab", 404)
        limit = max(1, min(int(params.get("limit", 100)), 500))
        rows, total = db.detections_page(run_id, params.get("tier", "FLAGGED"), params.get("split") or None,
                                         (params.get("q") or "")[:80] or None, params.get("missed") == "1",
                                         limit, max(0, int(params.get("offset", 0))))
        feats = load_json(state.dataset_dir / "feature_names.json")["feature_names"] if state.ready() else []
        wf = db.window_features_for(state.dataset, [(r["entity"], r["window_start"]) for r in rows], feats) if feats else {}
        runs = db.list_runs(limit=30)
    import json as _j
    return {"run_id": run_id, "total": total, "runs": runs, "rows": [{
        **{k: r[k] for k in ("entity", "window_start", "split", "label", "attack_scenario", "model_tier", "tier")},
        "score": round(r["score"], 4), "rules": _j.loads(r["rules"] or "[]"), "evidence": _j.loads(r["evidence"] or "[]"),
        "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in wf.get((r["entity"], r["window_start"]), {}).items()}}
        for r in rows]}


def window_events(state: AppState, run_id: int, entity: str, start: str) -> dict:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", start or ""):
        raise ApiError("start must look like 2025-03-11T10:20:00Z")
    from datetime import datetime, timedelta, timezone
    ws = int(load_json(state.dataset_dir / "feature_names.json")["window_seconds"])
    t0 = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end = (t0 + timedelta(seconds=ws)).strftime("%Y-%m-%dT%H:%M:%SZ")
    gb = load_json(state.dataset_dir / "feature_names.json")["group_by"]
    with Database(state.db_path) as db:
        ev = db.events_in_window(run_id, gb, entity, start, end)
    return {"entity": entity, "start": start, "end": end, "events": ev}


_NAME = re.compile(r"^[A-Za-z0-9_-]{1,60}$")


def realdata_list(state: AppState) -> dict:
    root = state.reports_root / "metrics"
    names = sorted(p.parent.name for p in root.glob("*/realdata.json")) if root.exists() else []
    return {"datasets": names}


def realdata_get(state: AppState, name: str) -> dict:
    if not _NAME.match(name or ""):
        raise ApiError("bad dataset name", 400)
    p = state.reports_root / "metrics" / name / "realdata.json"
    if not p.exists():
        raise ApiError("no real-log analysis for that dataset; run 'analyze_real' in the Pipeline tab", 404)
    d = load_json(p)
    fig = state.reports_root / "figures" / name / "realdata" / "realdata.png"
    d["figure"] = f"/figures/{name}/realdata/realdata.png" if fig.exists() else None
    return d


def maintenance(state: AppState) -> dict:
    with Database(state.db_path) as db:
        return {"db_path": str(state.db_path), "counts": db.counts(), "runs": db.list_runs(limit=20),
                "upload_dir": str(state.upload_dir)}


def purge_uploads(state: AppState) -> dict:
    with Database(state.db_path) as db:
        n = db.delete_runs("score_log")
    return {"deleted_runs": n}


def settings(state: AppState) -> dict:
    sc = state.cfg["scoring"]
    return {"scoring": {"model_id": sc["model_id"], "tier_precision_targets": sc["tier_precision_targets"],
                        "rules": sc["rules"], "rule_floors": sc["rule_floors"]},
            "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_rows_returned": MAX_ROWS},
            "security": ["binds 127.0.0.1 only", "no CORS headers; POST needs a custom header",
                         "Host header checked (DNS-rebinding guard)", "uploads are content only: no client file paths",
                         "log text rendered with textContent only", "models loaded with joblib from local artefacts"]}
