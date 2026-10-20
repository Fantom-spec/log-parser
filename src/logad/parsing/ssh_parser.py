"""OpenSSH syslog parser (Loghub-style and modern formats).

Counting rules (so one failed attempt is never counted several times)
---------------------------------------------------------------------
One failed credential attempt usually produces several log lines:
`Invalid user`, `input_userauth_request`, `pam_unix ... authentication failure`,
`Failed password`, `Connection closed`. Only the authoritative line is a LOGIN_FAILED:

* `Failed password|keyboard-interactive*` (configurable)  -> LOGIN_FAILED  (counted)
* `Failed none`                                           -> AUTH_PROBE   (not counted)
* `Failed publickey` / other methods                      -> AUTH_FAILED_OTHER (not counted;
  clients routinely try several keys before succeeding)
* `Invalid user`, PAM failures, `maximum authentication attempts exceeded`,
  `Disconnecting: Too many authentication failures`      -> own event types (not counted)
* pure duplicates (`input_userauth_request`, `check pass; user unknown`, ...) are *ignored*
  and tallied by reason in the parse report.
* `message repeated N times: [ ... ]` -> one event with repeat=N, so counts stay right.
"""
from __future__ import annotations

import ipaddress
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .normalizer import (
    STATUS_BY_TYPE, Event, ParseReport, events_to_frame, mask_message,
)

SOURCE_TYPE = "openssh_syslog"
SSHD_PROCS = {"sshd", "sshd-session"}
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
DEFAULT_FAILURE_METHODS = ("password", "keyboard-interactive/pam", "keyboard-interactive")

HEADER_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<proc>[^\s:\[\(]+(?:\([^)]*\))?)(?:\[(?P<pid>\d+)\])?:\s*(?P<msg>.*?)\s*$"
)
_IPTOK = r"[0-9a-fA-F\.:]+"
RE_REPEAT = re.compile(r"^message repeated (\d+) times?: \[ ?(.*?) ?\]$")
RE_ERR_PREFIX = re.compile(r"^error: ")
RE_FAILED = re.compile(
    r"^Failed (?P<method>\S+) for (?:(?P<invalid>invalid user) )?(?P<user>\S+) from (?P<ip>\S+)")
RE_ACCEPTED = re.compile(r"^Accepted (?P<method>\S+) for (?P<user>\S+) from (?P<ip>\S+)")
RE_INVALID_USER = re.compile(r"^Invalid user (?P<user>.*?) from (?P<ip>\S+)")
RE_PAM_FAIL = re.compile(r"^(?:pam_unix\(sshd:auth\): )?authentication failure;")
RE_PAM_MORE = re.compile(r"^PAM (\d+) more authentication failures?;")
RE_RHOST = re.compile(r"\brhost=(\S*)")
RE_PAMUSER = re.compile(r"\buser=(\S+)")
RE_BREAKIN = re.compile(r"\[(?P<ip>[0-9a-fA-F\.:]+)\].*POSSIBLE BREAK-IN ATTEMPT")
RE_RECV_DISC = re.compile(rf"^Received disconnect from (?P<ip>{_IPTOK}?)(?: port \d+)?:\s*\d+")
RE_DISCONNECTED = re.compile(
    rf"^Disconnected from (?:(?:invalid user|authenticating user|user) (?P<user>\S+) )?(?P<ip>{_IPTOK})(?: port \d+)?")
RE_DISC_MANY = re.compile(r"^Disconnecting: Too many authentication failures for (?P<user>\S+)")
RE_DISCONNECTING = re.compile(r"^Disconnecting(?: [^:]*)?:")
RE_CONN_CLOSED = re.compile(
    rf"^Connection closed by (?:(?:invalid user|authenticating user) (?P<user>\S+) )?(?P<ip>{_IPTOK})(?: port \d+)?")
RE_MAX_AUTH = re.compile(
    r"^maximum authentication attempts exceeded for (?:invalid user )?(?P<user>\S+) from (?P<ip>\S+)")
RE_PROTO = re.compile(
    r"^(?:Did not receive identification string from|Bad protocol version identification .* from|"
    r"Unable to negotiate with|banner exchange: [^:]*: invalid format from?)\s*(?P<ip>\S+)?")
RE_SESSION = re.compile(r"^(?:pam_unix\(sshd:session\): )?session (?P<what>opened|closed) for user (?P<user>\S+)")
IGNORE_PATTERNS = [
    ("input_userauth_request", re.compile(r"^input_userauth_request: invalid user")),
    ("pam_check_pass", re.compile(r"^pam_unix\(sshd:auth\): check pass; user unknown")),
    ("pam_ignoring_max_retries", re.compile(r"^PAM service\(sshd\) ignoring max retries")),
    ("fatal_read_from_socket", re.compile(r"^fatal: Read from socket failed")),
    ("pam_unix_auth_misc", re.compile(r"^pam_unix\(sshd:auth\): (?:authentication failure|check pass)")),
]


def _ip(token: str | None) -> str | None:
    if not token:
        return None
    try:
        return str(ipaddress.ip_address(token.strip()))
    except ValueError:
        return None


def _user(token: str | None) -> str | None:
    token = (token or "").strip()
    return token or None


class SSHParser:
    """Stateful parser (state = running year for Dec->Jan rollover). Feed lines in file order."""

    def __init__(self, year: int, tz_offset_hours: float = 0.0,
                 failure_methods: Iterable[str] = DEFAULT_FAILURE_METHODS, dedupe: bool = False):
        self.dedupe = bool(dedupe)
        self._seen: set[str] = set()
        self._prev_ts: datetime | None = None
        self.base_year = int(year)
        self._year = int(year)
        self._prev_month: int | None = None
        self.tz_offset = timedelta(hours=float(tz_offset_hours))
        self.failure_methods = set(failure_methods)
        self.report = ParseReport()
        self.events: list[Event] = []

    # -- timestamps ---------------------------------------------------------------
    def _timestamp(self, mon: str, day: str, hms: str) -> datetime | None:
        month = MONTHS.get(mon)
        if month is None:
            return None
        year = self._year
        if self._prev_month is not None:
            if self._prev_month - month >= 6:          # Dec -> Jan
                self._year += 1
                year = self._year
                self.report.year_rollovers += 1
                self._prev_month = month
            elif month - self._prev_month >= 6 and self._year > self.base_year:
                year = self._year - 1                  # late Dec line after rollover; keep state
            else:
                self._prev_month = month
        else:
            self._prev_month = month
        try:
            h, m, s = (int(x) for x in hms.split(":"))
            local = datetime(year, month, int(day), h, m, s)
        except ValueError:
            return None
        return (local - self.tz_offset).replace(tzinfo=timezone.utc)

    # -- message classification -----------------------------------------------------
    def _classify(self, msg: str):
        """-> ('event', type, user, ip) | ('ignored', reason) | None (unsupported)."""
        m = RE_FAILED.match(msg)
        if m:
            method = m.group("method")
            if method in self.failure_methods:
                etype = "LOGIN_FAILED"
            elif method == "none":
                etype = "AUTH_PROBE"
            else:
                etype = "AUTH_FAILED_OTHER"
            return ("event", etype, _user(m.group("user")), _ip(m.group("ip")))
        m = RE_ACCEPTED.match(msg)
        if m:
            return ("event", "LOGIN_SUCCESS", _user(m.group("user")), _ip(m.group("ip")))
        m = RE_INVALID_USER.match(msg)
        if m:
            return ("event", "INVALID_USER", _user(m.group("user")), _ip(m.group("ip")))
        if RE_PAM_FAIL.match(msg) or RE_PAM_MORE.match(msg):
            et = "PAM_AUTH_FAILURE" if RE_PAM_FAIL.match(msg) else "PAM_MORE_FAILURES"
            rh, us = RE_RHOST.search(msg), RE_PAMUSER.search(msg)
            return ("event", et, _user(us.group(1)) if us else None, _ip(rh.group(1)) if rh else None)
        m = RE_BREAKIN.search(msg)
        if m:
            return ("event", "BREAKIN_WARNING", None, _ip(m.group("ip")))
        m = RE_SESSION.match(msg)
        if m:
            et = "SESSION_OPENED" if m.group("what") == "opened" else "SESSION_CLOSED"
            return ("event", et, _user(m.group("user")), None)
        m = RE_MAX_AUTH.match(msg)
        if m:
            return ("event", "MAX_AUTH_EXCEEDED", _user(m.group("user")), _ip(m.group("ip")))
        m = RE_DISC_MANY.match(msg)
        if m:
            return ("event", "MAX_AUTH_EXCEEDED", _user(m.group("user")), None)
        m = RE_RECV_DISC.match(msg)
        if m:
            return ("event", "DISCONNECT", None, _ip(m.group("ip")))
        m = RE_DISCONNECTED.match(msg)
        if m:
            return ("event", "DISCONNECT", _user(m.group("user")), _ip(m.group("ip")))
        if RE_DISCONNECTING.match(msg):
            return ("event", "DISCONNECT", None, None)
        m = RE_CONN_CLOSED.match(msg)
        if m:
            return ("event", "CONNECTION_CLOSED", _user(m.group("user")), _ip(m.group("ip")))
        m = RE_PROTO.match(msg)
        if m:
            return ("event", "PROTOCOL_ERROR", None, _ip(m.group("ip")))
        for reason, rx in IGNORE_PATTERNS:
            if rx.match(msg):
                return ("ignored", reason)
        return None

    # -- public API ---------------------------------------------------------------
    def parse_line(self, line: str, line_no: int) -> Event | None:
        r = self.report
        r.total_lines += 1
        raw = line.rstrip("\r\n")
        if not raw.strip():
            r.blank_lines += 1
            return None
        if self.dedupe:
            if raw in self._seen:
                r.duplicate_lines_dropped += 1
                return None
            self._seen.add(raw)
        m = HEADER_RE.match(raw)
        if not m:
            r.malformed_lines += 1
            if len(r.malformed_samples) < 5:
                r.malformed_samples.append(raw[:120])
            return None
        ts = self._timestamp(m.group("mon"), m.group("day"), m.group("time"))
        if ts is None:
            r.invalid_timestamp_lines += 1
            return None
        if self._prev_ts is not None and ts < self._prev_ts:
            r.timestamp_regressions += 1          # out-of-order line (events are re-sorted by time afterwards)
        else:
            self._prev_ts = ts
        if m.group("proc").split("(")[0] not in SSHD_PROCS:
            r.not_sshd_lines += 1
            return None
        msg = m.group("msg")
        repeat = 1
        rm = RE_REPEAT.match(msg)
        if rm:
            repeat, msg = int(rm.group(1)), rm.group(2)
        msg = RE_ERR_PREFIX.sub("", msg)
        res = self._classify(msg)
        if res is None:
            r.unsupported_sshd_lines += 1
            r.unsupported_patterns[mask_message(msg)] += 1
            return None
        if res[0] == "ignored":
            r.ignored_redundant[res[1]] += 1
            return None
        _, etype, user, ip = res
        ev = Event(
            line_no=line_no, ts=ts, host=m.group("host"), username=user, src_ip=ip, dst_ip=None,
            event_type=etype, status=STATUS_BY_TYPE.get(etype, "UNKNOWN"), source_type=SOURCE_TYPE,
            native_event_id=m.group("pid"), raw_line=raw, repeat=max(1, repeat),
        )
        r.event_lines += 1
        r.events_by_type[etype] += 1
        self.events.append(ev)
        return ev

    def parse_lines(self, lines: Iterable[str]) -> "SSHParser":
        for i, line in enumerate(lines, 1):
            self.parse_line(line, i)
        return self

    def frame(self) -> pd.DataFrame:
        return events_to_frame(self.events)


def parse_file(path: str | Path, year: int, tz_offset_hours: float = 0.0,
               failure_methods: Iterable[str] = DEFAULT_FAILURE_METHODS, dedupe: bool = False):
    """Parse a log file -> (events DataFrame, ParseReport)."""
    parser = SSHParser(year, tz_offset_hours, failure_methods, dedupe)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        parser.parse_lines(fh)
    return parser.frame(), parser.report
