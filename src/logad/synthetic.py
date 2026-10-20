"""Deterministic synthetic SSH auth-log generator with event-level attack labels.

Writes OpenSSH-format syslog lines (so the real parser is exercised end to end), a labels
file keyed by line number, and a planted blocklist snapshot with manifest.

Caveat (also in README): attacker IPs are partly planted in the blocklist, so any
threat_intel result on synthetic data is partly circular by construction.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
         "ken", "laura", "mallory", "nina", "oscar", "peggy", "quinn", "rob", "sybil", "trent"]
SCENARIOS = ["brute_force", "distributed_brute_force", "off_hours_login", "new_ip_login",
             "success_after_failures"]


class _Gen:
    def __init__(self, cfg: dict, seed: int, year: int):
        s = cfg["synthetic"]
        self.s, self.host = s, s["host"]
        self.rng = np.random.default_rng(seed)
        mm, dd = (int(x) for x in s["start_month_day"].split("-"))
        self.start = datetime(year, mm, dd)
        self.days = int(s["days"])
        self.records: list[tuple] = []     # (ts, seq, text, raw, label, scenario, incident)
        self.seq = 0
        self.pid = 1000
        self.used: set[str] = set()
        self.planted: set[str] = set()

    # -- primitives ---------------------------------------------------------------
    def ip(self) -> str:
        while True:
            cand = f"{int(self.rng.integers(1, 224))}.{int(self.rng.integers(0, 256))}." \
                   f"{int(self.rng.integers(0, 256))}.{int(self.rng.integers(1, 255))}"
            if cand not in self.used and ipaddress.ip_address(cand).is_global:
                self.used.add(cand)
                return cand

    def attacker_ip(self) -> str:
        ip = self.ip()
        if self.rng.random() < float(self.s["attacker_blocklisted_fraction"]):
            self.planted.add(ip)
        return ip

    def new_pid(self) -> int:
        self.pid += int(self.rng.integers(1, 40))
        return self.pid

    def port(self) -> int:
        return int(self.rng.integers(1024, 65535))

    def add(self, ts, text, label=0, scenario="", incident="", raw=False):
        self.records.append((ts, self.seq, text, raw, label, scenario, incident))
        self.seq += 1

    def line(self, pid, msg) -> str:
        return f"{self.host} sshd[{pid}]: {msg}"

    # -- events -------------------------------------------------------------------
    def failed(self, ts, user, ip, tag):
        pid = self.new_pid()
        self.add(ts, self.line(pid, f"pam_unix(sshd:auth): authentication failure; logname= uid=0 "
                                    f"euid=0 tty=ssh ruser= rhost={ip}  user={user}"), *tag)
        self.add(ts, self.line(pid, f"Failed password for {user} from {ip} port {self.port()} ssh2"), *tag)

    def session(self, t0, user, ip, tag, typos=0, duration=None):
        pid, port = self.new_pid(), self.port()
        for k in range(typos):
            self.failed(t0 - timedelta(seconds=int(self.rng.integers(5, 30)) + 6 * k), user, ip, tag)
        key = self.rng.random() < 0.7
        self.add(t0, self.line(pid, f"Accepted publickey for {user} from {ip} port {port} ssh2: RSA SHA256:abc"
                               if key else f"Accepted password for {user} from {ip} port {port} ssh2"), *tag)
        self.add(t0, self.line(pid, f"pam_unix(sshd:session): session opened for user {user} by (uid=0)"), *tag)
        dur = duration if duration is not None else int(min(self.rng.lognormal(6.0, 0.8), 7200))
        te = t0 + timedelta(seconds=dur)
        self.add(te, self.line(pid, f"Received disconnect from {ip} port {port}:11: disconnected by user"), *tag)
        self.add(te, self.line(pid, f"Disconnected from user {user} {ip} port {port}"), *tag)
        self.add(te, self.line(pid, f"pam_unix(sshd:session): session closed for user {user}"), *tag)


def _fmt(ts: datetime) -> str:
    return f"{MONTHS[ts.month - 1]} {ts.day:>2} {ts:%H:%M:%S}"


def generate_synthetic(cfg: dict, out_dir, threat_intel_dir, seed: int | None = None) -> dict:
    s = cfg["synthetic"]
    seed = int(s["seed"] if seed is None else seed)
    year = int(cfg["dataset"]["log_year"])
    g = _Gen(cfg, seed, year)
    rng = g.rng
    out_dir, ti_dir = Path(out_dir), Path(threat_intel_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ti_dir.mkdir(parents=True, exist_ok=True)

    n_users = int(s["n_users"])
    users = [{"name": NAMES[i] if i < len(NAMES) else f"user{i:02d}",
              "center": float(rng.uniform(9.5, 15.0)),
              "ips": [g.ip() for _ in range(int(rng.integers(1, 3)))]} for i in range(n_users)]
    normal = ("", "", "")  # label stored as 0 below
    tag0 = (0, "", "")

    # normal behaviour
    for u in users:
        for d in range(g.days):
            day = g.start + timedelta(days=d)
            n = int(rng.poisson(0.5 if day.weekday() >= 5 else 3.0))
            for _ in range(n):
                if rng.random() < 0.03:
                    hour = float(rng.uniform(19.0, 22.5)) if rng.random() < 0.5 else float(rng.uniform(5.5, 8.0))
                else:
                    hour = float(np.clip(rng.normal(u["center"], 2.5), 8.0, 18.9))
                ip = str(rng.choice(u["ips"]))
                if rng.random() < float(s["benign_new_ip_prob"]):
                    ip = g.ip()
                typos = (1 + int(rng.random() < 0.3)) if rng.random() < float(s["typo_prob"]) else 0
                g.session(day + timedelta(seconds=int(hour * 3600)), u["name"], ip, tag0, typos=typos)

    # attack incidents
    total_s = g.days * 86400
    counts = {k: 0 for k in SCENARIOS}
    inc_no = 0

    n_inc = int(s["incidents_per_type"])

    def start_time(margin, k=0, n=1):
        """Incident k of n starts in the k-th equal slice of the period (stratified, so every
        chronological split receives attacks instead of leaving the test set to chance)."""
        lo, hi = total_s * k // n, min(total_s * (k + 1) // n, total_s - margin)
        return g.start + timedelta(seconds=int(rng.integers(min(lo, hi - 1), hi)))

    for scen in SCENARIOS:
        for k in range(n_inc):
            inc_no += 1
            tag = (1, scen, f"INC{inc_no:03d}")
            user = users[int(rng.integers(0, n_users))]
            counts[scen] += 1
            if scen == "brute_force":
                t0, dur, ip = start_time(600, k, n_inc), int(rng.integers(60, 240)), g.attacker_ip()
                for off in sorted(rng.integers(0, dur, int(rng.integers(30, 121)))):
                    g.failed(t0 + timedelta(seconds=int(off)), user["name"], ip, tag)
            elif scen == "distributed_brute_force":
                t0, dur = start_time(600, k, n_inc), int(rng.integers(120, 300))
                for _ in range(int(rng.integers(15, 41))):
                    ip = g.attacker_ip()
                    for off in rng.integers(0, dur, int(rng.integers(1, 4))):
                        g.failed(t0 + timedelta(seconds=int(off)), user["name"], ip, tag)
            elif scen == "off_hours_login":
                d_lo = g.days * k // n_inc
                day = g.start + timedelta(days=int(rng.integers(d_lo, max(d_lo + 1, g.days * (k + 1) // n_inc))))
                t0 = day + timedelta(seconds=int(rng.uniform(0.0, 5.0) * 3600))
                g.session(t0, user["name"], user["ips"][0], tag, duration=int(rng.integers(60, 600)))
            elif scen == "new_ip_login":
                g.session(start_time(900, k, n_inc), user["name"], g.attacker_ip(), tag,
                          typos=int(rng.integers(0, 3)), duration=int(rng.integers(60, 600)))
            else:  # success_after_failures
                t0, ip = start_time(900, k, n_inc), g.attacker_ip()
                n_fail = int(rng.integers(6, 16))
                for off in sorted(rng.integers(0, 60, n_fail)):
                    g.failed(t0 + timedelta(seconds=int(off)), user["name"], ip, tag)
                g.session(t0 + timedelta(seconds=61), user["name"], ip, tag, duration=int(rng.integers(60, 600)))

    # non-sshd noise and a few corrupted lines (exercise the parse report)
    for h in range(g.days * 24):
        g.add(g.start + timedelta(hours=h, minutes=17),
              f"{g.host} CRON[{g.new_pid()}]: pam_unix(cron:session): session opened for user root by (uid=0)")
    for _ in range(3):
        g.add(start_time(10), "@@ corrupted syslog fragment ##", raw=True)

    # ordering + files
    recs = sorted(g.records, key=lambda r: (r[0], r[1]))
    lines, lab_rows = [], []
    for i, (ts, _, text, raw, label, scen, inc) in enumerate(recs, 1):
        lines.append(text if raw else f"{_fmt(ts)} {text}")
        lab_rows.append((i, label, scen, inc))
    log_path, labels_path = out_dir / "auth.log", out_dir / "labels.csv"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(lab_rows, columns=["line_no", "label", "scenario", "incident_id"]).to_csv(labels_path, index=False)

    # planted blocklist snapshot (+ decoys that never appear in the logs) and manifest
    decoys = {g.ip() for _ in range(int(s["n_blocklist_decoys"]))}
    bl_ips = sorted(g.planted | decoys, key=lambda x: tuple(int(p) for p in x.split(".")))
    bl_path = ti_dir / "blocklist_synthetic.txt"
    bl_path.write_text("".join(f"{ip}\t{int(rng.integers(1, 9))}\n" for ip in bl_ips), encoding="utf-8")
    digest = hashlib.sha256(bl_path.read_bytes()).hexdigest()
    man_path = ti_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"files": []}
    manifest["files"] = [e for e in manifest["files"] if e.get("file") != bl_path.name]
    manifest["files"].append({
        "file": bl_path.name, "source": "synthetic (planted by generator)", "url": None,
        "date": "synthetic", "seed": seed, "sha256": digest,
        "note": "Attacker IPs are partly planted here: threat_intel results on synthetic data are circular."})
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "seed": seed, "log_year": year, "start": g.start.strftime("%m-%d"), "days": g.days,
        "n_lines": len(lines), "n_attack_lines": int(sum(r[4] for r in recs)),
        "incidents": counts, "planted_blocklist_ips": len(g.planted), "blocklist_ips_total": len(bl_ips),
        "files": {"log": str(log_path), "labels": str(labels_path), "blocklist": str(bl_path)},
    }
    (out_dir / "synthetic_meta.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
