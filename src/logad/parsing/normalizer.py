"""Normalized event schema, parse report and label attachment."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

# line_no and repeat are additions to the 10-field schema: line_no gives traceability and
# label joins; repeat carries rsyslog "message repeated N times" so counts stay correct.
EVENT_COLUMNS = [
    "line_no", "ts", "host", "username", "src_ip", "dst_ip", "event_type",
    "status", "source_type", "native_event_id", "raw_line", "repeat",
]

STATUS_BY_TYPE = {
    "LOGIN_SUCCESS": "SUCCESS",
    "SESSION_OPENED": "SUCCESS",
    "LOGIN_FAILED": "FAILURE",
    "AUTH_FAILED_OTHER": "FAILURE",
    "AUTH_PROBE": "FAILURE",
    "INVALID_USER": "FAILURE",
    "PAM_AUTH_FAILURE": "FAILURE",
    "PAM_MORE_FAILURES": "FAILURE",
    "MAX_AUTH_EXCEEDED": "FAILURE",
}  # everything else -> UNKNOWN


@dataclass(slots=True)
class Event:
    line_no: int
    ts: datetime
    host: str | None
    username: str | None
    src_ip: str | None
    dst_ip: str | None
    event_type: str
    status: str
    source_type: str
    native_event_id: str | None
    raw_line: str
    repeat: int = 1

    def as_row(self) -> tuple:
        return tuple(getattr(self, c) for c in EVENT_COLUMNS)


_MASK_IP = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
_MASK_NUM = re.compile(r"\d+")


def mask_message(msg: str, limit: int = 70) -> str:
    """Collapse IPs/numbers so similar unsupported lines group together in the report."""
    return _MASK_NUM.sub("#", _MASK_IP.sub("<ip>", msg))[:limit]


@dataclass
class ParseReport:
    """Everything the parser saw. Nothing is dropped silently."""

    total_lines: int = 0
    blank_lines: int = 0
    malformed_lines: int = 0
    invalid_timestamp_lines: int = 0
    not_sshd_lines: int = 0
    unsupported_sshd_lines: int = 0
    event_lines: int = 0
    year_rollovers: int = 0
    duplicate_lines_dropped: int = 0
    timestamp_regressions: int = 0
    ignored_redundant: Counter = field(default_factory=Counter)
    events_by_type: Counter = field(default_factory=Counter)
    unsupported_patterns: Counter = field(default_factory=Counter)
    malformed_samples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        ignored = int(sum(self.ignored_redundant.values()))
        nonblank = self.total_lines - self.blank_lines
        sshd_seen = self.event_lines + ignored + self.unsupported_sshd_lines
        return {
            "total_lines": self.total_lines,
            "blank_lines": self.blank_lines,
            "malformed_lines": self.malformed_lines,
            "invalid_timestamp_lines": self.invalid_timestamp_lines,
            "not_sshd_lines": self.not_sshd_lines,
            "unsupported_sshd_lines": self.unsupported_sshd_lines,
            "event_lines": self.event_lines,
            "ignored_redundant_lines": ignored,
            "year_rollovers": self.year_rollovers,
            "duplicate_lines_dropped": self.duplicate_lines_dropped,
            "timestamp_regressions": self.timestamp_regressions,
            "coverage_all_lines": round((self.event_lines + ignored) / nonblank, 6) if nonblank else None,
            "coverage_sshd_lines": round((self.event_lines + ignored) / sshd_seen, 6) if sshd_seen else None,
            "ignored_redundant_by_reason": dict(self.ignored_redundant),
            "events_by_type": dict(self.events_by_type),
            "unsupported_top_patterns": dict(self.unsupported_patterns.most_common(20)),
            "malformed_samples": list(self.malformed_samples),
        }


def events_to_frame(events: list[Event]) -> pd.DataFrame:
    """Events -> DataFrame (UTC timestamps), stably sorted by (ts, line_no)."""
    if not events:
        df = pd.DataFrame({c: [] for c in EVENT_COLUMNS})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    df = pd.DataFrame([e.as_row() for e in events], columns=EVENT_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["repeat"] = df["repeat"].astype("int64")
    return df.sort_values(["ts", "line_no"], kind="stable").reset_index(drop=True)


def add_empty_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Unlabeled data: labels are NaN ('unknown'), never 'normal'."""
    out = events.copy()
    out["label"] = np.nan
    out["scenario"] = None
    return out


def attach_labels(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join event-level labels on line_no. Lines absent from the labels file stay NaN."""
    need = {"line_no", "label"}
    if not need.issubset(labels.columns):
        raise ValueError(f"labels file needs columns {sorted(need)}; got {list(labels.columns)}")
    if labels["line_no"].duplicated().any():
        raise ValueError("labels file has duplicate line_no values")
    if not labels["label"].isin([0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    lab = labels.copy()
    if "scenario" not in lab.columns:
        lab["scenario"] = None
    lab = lab[["line_no", "label", "scenario"]]
    out = events.merge(lab, on="line_no", how="left")
    out["label"] = out["label"].astype("float64")
    return out
