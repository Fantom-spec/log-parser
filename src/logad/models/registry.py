"""Model registry: joblib artefacts + a deterministic JSON index (no wall-clock values)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from logad.io import dump_json, load_json


def model_version(entry: dict) -> str:
    key = json.dumps({k: entry[k] for k in ("family", "params", "seed", "features", "fit_data_sha256")},
                     sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:10]


class ModelRegistry:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "registry.json"
        self.entries: dict = {}

    def register(self, model_id: str, model, entry: dict) -> dict:
        entry = dict(entry, model_id=model_id)
        entry["version"] = model_version(entry)
        joblib.dump(model, self.dir / f"{model_id}.joblib")
        self.entries[model_id] = entry
        return entry

    def save(self) -> None:
        dump_json({"models": self.entries}, self.index_path)

    @classmethod
    def load(cls, directory) -> "ModelRegistry":
        reg = cls(directory)
        reg.entries = load_json(reg.index_path)["models"]
        return reg

    def load_model(self, model_id: str):
        return joblib.load(self.dir / f"{model_id}.joblib")
