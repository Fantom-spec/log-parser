"""Demo SSH logs built from the TRAINED dataset (real users, their usual IPs and hours)."""
from __future__ import annotations

from datetime import datetime, timedelta

from logad.io import load_json

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DESCRIPTIONS = {
    "normal_day": "A quiet day: known users from their usual IPs at their usual hours. One user mistypes a password once, so a single LOW is realistic.",
    "brute_force": "Normal activity plus one IP hammering a single account with ~90 password attempts.",
    "distributed_brute_force": "Normal activity plus 28 different IPs each trying a single account a few times.",
    "compromise": "Repeated failures from one IP, then a successful login at 03:07 (possible account takeover).",
    "mixed_incident_day": "Everything above in one day, plus a blocklisted IP probing an account. Best for a demo.",
}


def _fmt(ts: datetime) -> str:
    return f"{MONTHS[ts.month - 1]} {ts.day:>2} {ts:%H:%M:%S}"


class _Builder:
    def __init__(self, host: str, day: datetime):
        self.host, self.day, self.rows, self.pid = host, day, [], 4000

    def add(self, ts, msg):
        self.rows.append((ts, len(self.rows), f"{self.host} sshd[{self.pid}]: {msg}"))

    def newpid(self):
        self.pid += 7

    def session(self, user, ip, hour, minutes=20, typo=False):
        t0 = self.day + timedelta(hours=int(hour), minutes=int((hour % 1) * 60))
        self.newpid()
        if typo:
            self.fail(t0 - timedelta(seconds=9), user, ip)
        self.add(t0, f"Accepted publickey for {user} from {ip} port 50022 ssh2: RSA SHA256:demo")
        self.add(t0, f"pam_unix(sshd:session): session opened for user {user} by (uid=0)")
        te = t0 + timedelta(minutes=minutes)
        self.add(te, f"Received disconnect from {ip} port 50022:11: disconnected by user")
        self.add(te, f"pam_unix(sshd:session): session closed for user {user}")

    def fail(self, ts, user, ip):
        self.newpid()
        self.add(ts, f"pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={ip}  user={user}")
        self.add(ts, f"Failed password for {user} from {ip} port 41022 ssh2")

    def text(self):
        return "\n".join(f"{_fmt(r[0])} {r[2]}" for r in sorted(self.rows)) + "\n"


def build_sample(name: str, baselines_path, blocklist_ips, year: int, host: str = "bastion01") -> str:
    if name not in DESCRIPTIONS:
        raise KeyError(name)
    ents = load_json(baselines_path)["entities"]
    users = sorted(ents, key=lambda u: (-ents[u]["n"], u))[:6]
    if len(users) < 5:
        raise ValueError("trained dataset has too few users to build demo logs")
    info = {u: (ents[u]["known_ips"][0], ents[u]["hour_mean"]) for u in users}
    day = datetime(year, 3, 11)
    b = _Builder(host, day)
    for i, u in enumerate(users):
        ip, h = info[u]
        b.session(u, ip, max(8.5, min(h, 17.0)) + (i % 2) * 0.5, typo=(i == 2))
        if i % 2:
            b.session(u, ip, max(8.5, min(h, 16.0)) - 1.0)
    bl = sorted(blocklist_ips)
    victim, other, third, fourth = users[0], users[1], users[3], users[4]
    if name in ("brute_force", "mixed_incident_day"):
        t0 = day + timedelta(hours=10, minutes=20)
        for k in range(90):
            b.fail(t0 + timedelta(seconds=2 * k), victim, "203.0.113.45")
    if name in ("distributed_brute_force", "mixed_incident_day"):
        t0 = day + timedelta(hours=14, minutes=5)
        for k in range(28):
            for r in range(2):
                b.fail(t0 + timedelta(seconds=9 * k + 3 * r), other, f"198.51.100.{k + 10}")
    if name in ("compromise", "mixed_incident_day"):
        t0 = day + timedelta(hours=3, minutes=6)
        for k in range(11):
            b.fail(t0 + timedelta(seconds=4 * k), third, "192.0.2.77")
        b.session(third, "192.0.2.77", 3 + 7 / 60, minutes=6)
    if name == "mixed_incident_day" and bl:
        t0 = day + timedelta(hours=16, minutes=40)
        for k in range(3):
            b.fail(t0 + timedelta(seconds=15 * k), fourth, bl[0])
    return b.text()
