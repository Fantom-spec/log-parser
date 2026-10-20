"""scoring/explain.py and config/thresholds.yaml."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _path  # noqa: F401
from logad.config import DEFAULTS, PROJECT_ROOT, load_config
from logad.scoring.explain import explain_window, feature_profile


class ExplainTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.fit = pd.DataFrame({"failed_count": rng.poisson(1, 200).astype(float), "event_count": rng.poisson(5, 200).astype(float)})
        self.feats = ["failed_count", "event_count"]
        self.prof = feature_profile(self.fit, self.feats)

    def test_profile_has_floor_on_scale(self):
        const = feature_profile(pd.DataFrame({"x": [2.0] * 10}), ["x"])
        self.assertEqual(const["x"]["median"], 2.0)
        self.assertGreaterEqual(const["x"]["scale"], 0.1)

    def test_only_large_deviations_are_listed_and_capped(self):
        row = pd.Series({"failed_count": 60.0, "event_count": 5.0})
        ev = explain_window(row, [], self.feats, self.prof, {}, top_k=3)
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0].startswith("failed_count=60"))
        calm = pd.Series({"failed_count": 1.0, "event_count": 5.0})
        self.assertEqual(explain_window(calm, [], self.feats, self.prof, {}, top_k=3), [])


class ThresholdsFileTests(unittest.TestCase):
    def test_file_exists_and_matches_defaults(self):
        f = PROJECT_ROOT / "config" / "thresholds.yaml"
        self.assertTrue(f.exists())
        cfg = load_config()
        for k in ("scoring", "unlabeled"):
            self.assertEqual(cfg[k], DEFAULTS[k])

    def test_thresholds_yaml_overrides_config_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "config.yaml").write_text(yaml.safe_dump({"scoring": {"evidence_top_k": 9}}))
            (d / "thresholds.yaml").write_text(yaml.safe_dump({"scoring": {"evidence_top_k": 5}}))
            self.assertEqual(load_config(d / "config.yaml")["scoring"]["evidence_top_k"], 5)
            (d / "thresholds.yaml").unlink()
            self.assertEqual(load_config(d / "config.yaml")["scoring"]["evidence_top_k"], 9)


if __name__ == "__main__":
    unittest.main()
