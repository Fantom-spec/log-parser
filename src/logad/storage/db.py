"""SQLite persistence. Feature columns follow the dataset's feature list (no fixed 18).

All values are bound parameters. Identifiers (feature column names) are validated against a strict
pattern and quoted, so a malformed feature list cannot inject SQL.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
FIXED_WF = ["dataset", "entity", "window_start", "split", "label", "attack_scenario"]


def _ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe column name: {name!r}")
    return f'"{name}"'


def _py(v):
    """numpy/pandas scalars -> plain Python for sqlite."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%dT%H:%M:%SZ")
    if v is pd.NA or v is pd.NaT:
        return None
    return v


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.conn.commit()
        self.close()

    def _init_schema(self):
        ver = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if ver not in (0, SCHEMA_VERSION):
            raise RuntimeError(f"database schema version {ver} is not supported (expected {SCHEMA_VERSION})")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets(
            dataset TEXT PRIMARY KEY, group_by TEXT, window_seconds INTEGER,
            feature_names TEXT NOT NULL, meta TEXT);
        CREATE TABLE IF NOT EXISTS window_features(
            dataset TEXT NOT NULL, entity TEXT NOT NULL, window_start TEXT NOT NULL,
            split TEXT, label INTEGER, attack_scenario TEXT,
            PRIMARY KEY(dataset, entity, window_start));
        CREATE TABLE IF NOT EXISTS model_registry(
            dataset TEXT NOT NULL, tag TEXT NOT NULL, model_id TEXT NOT NULL, family TEXT, version TEXT,
            threshold REAL, features TEXT, params TEXT, val_metrics TEXT,
            PRIMARY KEY(dataset, tag, model_id));
        CREATE TABLE IF NOT EXISTS runs(
            run_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, dataset TEXT NOT NULL,
            model_id TEXT, model_version TEXT, created_at TEXT NOT NULL, params TEXT);
        CREATE TABLE IF NOT EXISTS detections(
            run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            entity TEXT NOT NULL, window_start TEXT NOT NULL, split TEXT, label INTEGER, attack_scenario TEXT,
            score REAL NOT NULL, model_tier TEXT NOT NULL, tier TEXT NOT NULL, rules TEXT, evidence TEXT,
            PRIMARY KEY(run_id, entity, window_start));
        CREATE INDEX IF NOT EXISTS idx_det_tier ON detections(run_id, tier, score DESC);
        CREATE TABLE IF NOT EXISTS jobs(
            job_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, status TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, exit_code INTEGER, log TEXT);
        CREATE TABLE IF NOT EXISTS events(
            run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            line_no INTEGER, ts TEXT, host TEXT, username TEXT, src_ip TEXT, event_type TEXT,
            status TEXT, repeat INTEGER, raw_line TEXT);
        """)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    # ---- datasets and features -----------------------------------------------------------
    def ensure_feature_columns(self, features: list[str]) -> None:
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(window_features)")}
        for f in features:
            if f not in have:
                self.conn.execute(f"ALTER TABLE window_features ADD COLUMN {_ident(f)} REAL")

    def put_dataset(self, dataset, features, group_by, window_seconds, meta: dict | None = None):
        self.conn.execute("INSERT OR REPLACE INTO datasets VALUES(?,?,?,?,?)",
                          (dataset, group_by, int(window_seconds), json.dumps(features),
                           json.dumps(meta or {}, default=str)))
        self.ensure_feature_columns(features)

    def put_window_features(self, dataset, df: pd.DataFrame, features: list[str]):
        self.ensure_feature_columns(features)
        self.conn.execute("DELETE FROM window_features WHERE dataset = ?", (dataset,))
        cols = FIXED_WF + features
        sql = f"INSERT INTO window_features({','.join(_ident(c) for c in cols)}) VALUES({','.join('?' * len(cols))})"
        d = df.assign(dataset=dataset)
        rows = [tuple(_py(v) for v in r) for r in d[cols].itertuples(index=False, name=None)]
        self.conn.executemany(sql, rows)

    def put_models(self, dataset, tag, entries: dict):
        self.conn.execute("DELETE FROM model_registry WHERE dataset = ? AND tag = ?", (dataset, tag))
        for mid, e in entries.items():
            self.conn.execute("INSERT INTO model_registry VALUES(?,?,?,?,?,?,?,?,?)", (
                dataset, tag, mid, e.get("family"), e.get("version"), e.get("threshold"),
                json.dumps(e.get("features")), json.dumps(e.get("params")), json.dumps(e.get("val_metrics"))))

    # ---- runs, detections, events ---------------------------------------------------------
    def start_run(self, kind, dataset, model_id, model_version, params=None, replace_previous=True) -> int:
        if replace_previous:
            self.conn.execute("DELETE FROM runs WHERE kind = ? AND dataset = ?", (kind, dataset))
        cur = self.conn.execute(
            "INSERT INTO runs(kind,dataset,model_id,model_version,created_at,params) VALUES(?,?,?,?,?,?)",
            (kind, dataset, model_id, model_version,
             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), json.dumps(params or {}, default=str)))
        return int(cur.lastrowid)

    def put_detections(self, run_id: int, scored: pd.DataFrame):
        rows = []
        for r in scored.itertuples(index=False):
            d = r._asdict()
            rows.append((run_id, d["entity"], _py(d["window_start"]), d.get("split"),
                         _py(d.get("label")) if d.get("label") is not None else None,
                         d.get("attack_scenario") or None, float(d["score"]), d["model_tier"], d["tier"],
                         json.dumps(list(d["rules"])), json.dumps(list(d["evidence"]))))
        self.conn.executemany("INSERT INTO detections VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)

    def put_events(self, run_id: int, events: pd.DataFrame, limit: int = 200_000):
        if len(events) > limit:
            raise ValueError(f"refusing to store more than {limit} events in one run")
        cols = ["line_no", "ts", "host", "username", "src_ip", "event_type", "status", "repeat", "raw_line"]
        rows = [(run_id, *[_py(v) for v in r]) for r in events[cols].itertuples(index=False, name=None)]
        self.conn.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", rows)

    # ---- queries (parameterised) ------------------------------------------------------------
    def latest_run(self, kind, dataset) -> int | None:
        r = self.conn.execute("SELECT MAX(run_id) FROM runs WHERE kind=? AND dataset=?", (kind, dataset)).fetchone()
        return r[0]

    def detections(self, run_id: int, min_tier: str | None = None, limit: int = 500, split: str | None = None):
        order = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        tiers = order[order.index(min_tier):] if min_tier else order
        q = f"SELECT * FROM detections WHERE run_id=? AND tier IN ({','.join('?' * len(tiers))})"
        args: list = [run_id, *tiers]
        if split:
            q += " AND split=?"; args.append(split)
        q += " ORDER BY score DESC LIMIT ?"; args.append(int(limit))
        return [dict(r) for r in self.conn.execute(q, args)]

    def detections_page(self, run_id: int, tier: str = "ANY", split: str | None = None, q: str | None = None,
                        missed: bool = False, limit: int = 200, offset: int = 0):
        """Filtered, paged detections -> (rows, total). tier: ANY | FLAGGED | one tier name."""
        order = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        where, args = ["run_id=?"], [run_id]
        if tier == "FLAGGED":
            where.append("tier != 'NORMAL'")
        elif tier in order:
            where.append("tier = ?"); args.append(tier)
        if split:
            where.append("split = ?"); args.append(split)
        if q:
            esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("entity LIKE ? ESCAPE '\\'"); args.append(f"%{esc}%")
        if missed:
            where.append("label = 1 AND tier = 'NORMAL'")
        w = " AND ".join(where)
        total = self.conn.execute(f"SELECT COUNT(*) FROM detections WHERE {w}", args).fetchone()[0]
        rows = [dict(r) for r in self.conn.execute(
            f"SELECT * FROM detections WHERE {w} ORDER BY score DESC LIMIT ? OFFSET ?", [*args, int(limit), int(offset)])]
        return rows, total

    def events_in_window(self, run_id: int, group_by: str, entity: str, start: str, end: str, limit: int = 500):
        col = {"user": "username", "src_ip": "src_ip"}[group_by]          # whitelist: never client-controlled
        return [dict(r) for r in self.conn.execute(
            f"SELECT line_no, ts, host, username, src_ip, event_type, status, repeat, raw_line FROM events "
            f"WHERE run_id=? AND {col}=? AND ts>=? AND ts<? ORDER BY ts, line_no LIMIT ?",
            (run_id, entity, start, end, int(limit)))]

    def delete_runs(self, kind: str) -> int:
        return self.conn.execute("DELETE FROM runs WHERE kind=?", (kind,)).rowcount

    def list_runs(self, kind: str | None = None, limit: int = 50):
        q, args = "SELECT * FROM runs", []
        if kind:
            q += " WHERE kind=?"; args.append(kind)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY run_id DESC LIMIT ?", [*args, limit])]

    # ---- jobs ----
    def insert_job(self, name: str) -> int:
        cur = self.conn.execute("INSERT INTO jobs(name,status,started_at) VALUES(?,?,?)",
                                (name, "running", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_job(self, job_id: int, status: str, exit_code, log: str):
        self.conn.execute("UPDATE jobs SET status=?, finished_at=?, exit_code=?, log=? WHERE job_id=?",
                          (status, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), exit_code, log[-20000:], job_id))
        self.conn.commit()

    def list_jobs(self, limit: int = 20):
        return [dict(r) for r in self.conn.execute(
            "SELECT job_id,name,status,started_at,finished_at,exit_code FROM jobs ORDER BY job_id DESC LIMIT ?", (limit,))]

    def get_job(self, job_id: int):
        r = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(r) if r else None

    def window_features_for(self, dataset, keys: list[tuple[str, str]], features: list[str]) -> dict:
        out = {}
        cols = ",".join(_ident(f) for f in features)
        for ent, ws in keys:
            r = self.conn.execute(f"SELECT {cols} FROM window_features WHERE dataset=? AND entity=? AND window_start=?",
                                  (dataset, ent, ws)).fetchone()
            if r:
                out[(ent, ws)] = dict(r)
        return out

    def counts(self) -> dict:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("datasets", "window_features", "model_registry", "runs", "detections", "events")}
