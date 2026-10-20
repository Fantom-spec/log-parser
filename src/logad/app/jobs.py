"""Allowlisted job runner: one job at a time, live logs, cancel, history in SQLite.

Jobs are fixed command lines (never built from client input); the client only picks a name.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

from logad.config import PROJECT_ROOT
from logad.storage.db import Database

SCRIPTS = PROJECT_ROOT / "scripts"
DEFAULT_ALLOWLIST = {
    "full_demo": ([sys.executable, str(SCRIPTS / "run_pipeline.py")], "Generate data, train, evaluate, tiers, dashboard (about a minute)"),
    "generate": ([sys.executable, str(SCRIPTS / "generate_synthetic_logs.py")], "Generate synthetic SSH logs + labels + blocklist snapshot"),
    "prepare": ([sys.executable, str(SCRIPTS / "prepare_data.py")], "Parse, window and build features (leak-free split)"),
    "train": ([sys.executable, str(SCRIPTS / "train.py")], "Fit models, calibrate thresholds on validation"),
    "evaluate": ([sys.executable, str(SCRIPTS / "evaluate.py")], "Held-out test metrics and figures"),
    "calibrate": ([sys.executable, str(SCRIPTS / "calibrate_scoring.py")], "Risk tiers + model-only vs model+rules"),
    "ablation": ([sys.executable, str(SCRIPTS / "run_ablation.py")], "Feature-family ablation (slower)"),
    "h1": ([sys.executable, str(SCRIPTS / "run_h1.py")], "H1: per-user vs global baseline"),
    "robustness": ([sys.executable, str(SCRIPTS / "run_robustness.py")], "Scoring latency and detection delay"),
    "fetch_loghub": ([sys.executable, str(SCRIPTS / "fetch_loghub.py")], "Download Loghub OpenSSH (needs internet; one-time)"),
    "fetch_blocklist": ([sys.executable, str(SCRIPTS / "fetch_blocklist.py")], "Download the ipsum blocklist snapshot (needs internet)"),
    "analyze_real": ([sys.executable, str(SCRIPTS / "analyze_real.py")], "Analyse the real unlabeled log configured under real: in config.yaml"),
    "dashboard": ([sys.executable, str(SCRIPTS / "build_dashboard.py")], "Ingest to SQLite and rebuild the static dashboard"),
}


class JobError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


class _Job:
    def __init__(self, job_id, name):
        self.id, self.name, self.status, self.exit_code = job_id, name, "running", None
        self.lines: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.cancelled = False


class JobRunner:
    def __init__(self, db_path, allowlist=None, cwd=PROJECT_ROOT):
        self.db_path, self.cwd = db_path, str(cwd)
        self.allow = allowlist if allowlist is not None else DEFAULT_ALLOWLIST
        self.lock = threading.Lock()
        self.jobs: dict[int, _Job] = {}
        self.current: _Job | None = None

    def catalog(self):
        return [{"name": n, "description": d} for n, (_, d) in self.allow.items()]

    def busy(self) -> bool:
        with self.lock:
            return self.current is not None and self.current.status == "running"

    def start(self, name: str) -> dict:
        if name not in self.allow:
            raise JobError(f"unknown job {name!r}", 404)
        with self.lock:
            if self.current is not None and self.current.status == "running":
                raise JobError(f"job {self.current.id} ({self.current.name}) is still running", 409)
            with Database(self.db_path) as db:
                job = _Job(db.insert_job(name), name)
            self.jobs[job.id] = job
            self.current = job
        threading.Thread(target=self._run, args=(job, self.allow[name][0]), daemon=True).start()
        return {"job_id": job.id, "name": name, "status": "running"}

    def _run(self, job: _Job, argv):
        try:
            job.proc = subprocess.Popen(argv, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        errors="replace", env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
            for line in job.proc.stdout:
                job.lines.append(line.rstrip("\n"))
                if len(job.lines) > 3000:
                    del job.lines[:500]
            job.exit_code = job.proc.wait()
            job.status = "cancelled" if job.cancelled else ("succeeded" if job.exit_code == 0 else "failed")
        except Exception as exc:                                     # noqa: BLE001 - surfaced in the log
            job.lines.append(f"runner error: {exc}")
            job.status = "failed"
        with Database(self.db_path) as db:
            db.finish_job(job.id, job.status, job.exit_code, "\n".join(job.lines))

    def get(self, job_id: int, since: int = 0) -> dict:
        job = self.jobs.get(job_id)
        if job:
            return {"job_id": job.id, "name": job.name, "status": job.status, "exit_code": job.exit_code,
                    "next": len(job.lines), "lines": job.lines[since:]}
        with Database(self.db_path) as db:
            row = db.get_job(job_id)
        if not row:
            raise JobError("no such job", 404)
        lines = (row["log"] or "").split("\n")
        return {"job_id": row["job_id"], "name": row["name"], "status": row["status"], "exit_code": row["exit_code"],
                "next": len(lines), "lines": lines[since:]}

    def cancel(self, job_id: int) -> dict:
        job = self.jobs.get(job_id)
        if not job or job.status != "running" or job.proc is None:
            raise JobError("job is not running", 409)
        job.cancelled = True
        job.proc.terminate()
        try:
            job.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            job.proc.kill()
        return {"job_id": job_id, "status": "cancelling"}

    def history(self):
        with Database(self.db_path) as db:
            return db.list_jobs()

    def wait(self, job_id: int, timeout: float = 30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.jobs.get(job_id)
            if j and j.status != "running":
                time.sleep(0.05)
                return
            time.sleep(0.05)
