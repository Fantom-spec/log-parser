"""Learned per-entity baselines. Fitted on training windows ONLY, then frozen.

Update policy: baselines are frozen after the training split; validation/test/scoring never
update them (documented in README). Robust statistics: median and MAD (scaled 1.4826).

Cold start: an entity with fewer than `min_history_windows` training windows (or none)
falls back to the global training baseline; `low_history` marks these rows so results can
be reported separately.

new_ip_count on TRAIN windows uses an expanding history (only strictly earlier training
windows), otherwise every training IP would be "known" by definition. On val/test it uses the
frozen training baseline. event_rate_z / hour_deviation on train rows are in-sample.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _robust(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)) * 1.4826)


def _circ_mean_hour(h: np.ndarray) -> float:
    ang = 2 * np.pi * h / 24.0
    return float((math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) * 24 / (2 * math.pi)) % 24)


@dataclass
class Baselines:
    group_by: str
    min_history: int
    global_stats: dict
    entities: dict
    _sets: dict = field(default_factory=dict, repr=False)

    @classmethod
    def fit(cls, raw: pd.DataFrame, fit_mask, group_by: str, min_history: int) -> "Baselines":
        fit_df = raw.loc[np.asarray(fit_mask)]
        if fit_df.empty:
            raise ValueError("no training windows available to fit baselines")
        ents = {}
        for ent, g in fit_df.groupby("entity", sort=True):
            med, mad = _robust(g["event_count"].to_numpy(float))
            ents[str(ent)] = {
                "n": int(len(g)), "median": med, "mad": mad,
                "hour_mean": _circ_mean_hour(g["hour_local"].to_numpy(float)),
                "known_ips": sorted(set().union(*g["_ips"])),
            }
        med, mad = _robust(fit_df["event_count"].to_numpy(float))
        glob = {"n": int(len(fit_df)), "median": med, "mad": mad,
                "hour_mean": _circ_mean_hour(fit_df["hour_local"].to_numpy(float)),
                "known_ips": sorted(set().union(*fit_df["_ips"]))}
        return cls(group_by, int(min_history), glob, ents)

    def stats_for(self, entity: str):
        e = self.entities.get(entity)
        if e is not None and e["n"] >= self.min_history:
            return e, True
        return self.global_stats, False

    def known_ips(self, entity: str) -> set:
        st, own = self.stats_for(entity)
        use_entity = own and self.group_by == "user"
        key = entity if use_entity else "__global__"
        if key not in self._sets:
            self._sets[key] = set((st if use_entity else self.global_stats)["known_ips"])
        return self._sets[key]

    def to_dict(self) -> dict:
        return {"group_by": self.group_by, "min_history_windows": self.min_history,
                "global": self.global_stats, "entities": self.entities}

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Baselines":
        return cls(d["group_by"], d["min_history_windows"], d["global"], d["entities"])


def _stat_features(raw: pd.DataFrame, b: "Baselines", z_clip: float):
    """event_rate_z, hour_deviation, history_windows, low_history from FROZEN baselines."""
    n = len(raw)
    ents = raw["entity"].to_numpy()
    ec = raw["event_count"].to_numpy(float)
    hl = raw["hour_local"].to_numpy(float)
    z, hd = np.zeros(n), np.zeros(n)
    hist, low = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    for i in range(n):
        st, own = b.stats_for(ents[i])
        z[i] = np.clip((ec[i] - st["median"]) / max(st["mad"], 1.0), -z_clip, z_clip)
        d = abs(hl[i] - st["hour_mean"]) % 24
        hd[i] = min(d, 24 - d)
        e = b.entities.get(ents[i])
        hist[i] = e["n"] if e else 0
        low[i] = not own
    return z, hd, hist, low


def apply_frozen_baselines(raw: pd.DataFrame, baselines: "Baselines", z_clip: float) -> pd.DataFrame:
    """Scoring-time path: every window is treated like a validation/test window (frozen baselines)."""
    z, hd, hist, low = _stat_features(raw, baselines, z_clip)
    ents, ips = raw["entity"].to_numpy(), raw["_ips"].to_numpy()
    new_ip = np.array([len(ips[i] - baselines.known_ips(ents[i])) for i in range(len(raw))], dtype=int)
    out = raw.copy()
    out["event_rate_z"], out["hour_deviation"], out["new_ip_count"] = z, hd, new_ip
    out["history_windows"], out["low_history"] = hist, low
    return out


def add_baseline_features(raw: pd.DataFrame, split, fit_mask, group_by: str,
                          min_history: int, z_clip: float):
    """Fit baselines on `fit_mask` rows, add baseline-dependent features. -> (frame, Baselines)."""
    split = np.asarray(split)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask[split != "train"].any():
        raise ValueError("fit_mask may only select training rows")  # leakage guard
    b = Baselines.fit(raw, fit_mask, group_by, min_history)

    n = len(raw)
    ents = raw["entity"].to_numpy()
    ips = raw["_ips"].to_numpy()
    z, hd, hist, low = _stat_features(raw, b, z_clip)

    new_ip = np.zeros(n, dtype=int)
    for i in np.flatnonzero(split != "train"):                 # frozen training baseline
        new_ip[i] = len(ips[i] - b.known_ips(ents[i]))
    tr = np.flatnonzero(split == "train")                      # expanding history on train
    ws = raw["window_start"].to_numpy()[tr]
    bounds = list(np.flatnonzero(np.r_[True, ws[1:] != ws[:-1]])) + [len(tr)]
    ent_seen: dict[str, set] = {}
    ent_n: dict[str, int] = {}
    gseen: set = set()
    for a, c in zip(bounds[:-1], bounds[1:]):
        batch = tr[a:c]
        for i in batch:
            use_ent = group_by == "user" and ent_n.get(ents[i], 0) >= min_history
            new_ip[i] = len(ips[i] - (ent_seen[ents[i]] if use_ent else gseen))
        for i in batch:                                        # update after the whole timestamp
            if fit_mask[i]:
                ent_seen.setdefault(ents[i], set()).update(ips[i])
                ent_n[ents[i]] = ent_n.get(ents[i], 0) + 1
                gseen.update(ips[i])

    out = raw.copy()
    out["event_rate_z"], out["hour_deviation"], out["new_ip_count"] = z, hd, new_ip
    out["history_windows"], out["low_history"] = hist, low
    return out, b
