import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import _path  # noqa: F401
from logad.config import load_config
from logad.evaluation.metrics import (
    aggregate, bootstrap_ci, calibrate_threshold, per_scenario_recall, threshold_metrics)
from logad.models.registry import ModelRegistry
from logad.preprocessing.pipeline import Preprocessor, load_dataset
from logad.services.evaluate import evaluate_models
from logad.services.train import train_models
from test_pipeline import build

FAST = {"models": {"seeds": [42, 43, 44]}, "evaluation": {"bootstrap_samples": 60}}


def prepared(tmp, sub="a", **kw):
    ov = {"synthetic": {"days": 10, "n_users": 10, "incidents_per_type": 6, "n_blocklist_decoys": 20}}
    for k, v in FAST.items():
        ov[k] = v
    cfg, gen, meta, out = build(Path(tmp), ov, sub=sub, **kw)
    cfg = load_config(overrides={**FAST})
    return cfg, out


class MetricTests(unittest.TestCase):
    def test_threshold_metrics_hand_computed(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        s = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.1])
        m = threshold_metrics(y, s, 0.5, span_days=2.0)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 4))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["false_positive_rate"], 1 / 5)
        self.assertAlmostEqual(m["alerts_per_day"], 1.5)
        self.assertAlmostEqual(m["prevalence"], 3 / 8)

    def test_calibration_picks_max_f1_with_ge_rule(self):
        y = np.array([0, 0, 0, 1, 1, 0, 1])
        s = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7])
        c = calibrate_threshold(y, s)
        self.assertEqual(threshold_metrics(y, s, c["threshold"])["f1"], 1.0)
        self.assertAlmostEqual(c["threshold"], 0.7)            # highest threshold reaching F1=1

    def test_calibration_needs_both_classes(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(np.zeros(5), np.arange(5.0))

    def test_per_scenario_recall_and_bootstrap_and_aggregate(self):
        y = np.array([1, 1, 1, 0])
        pred = np.array([True, False, True, False])
        r = per_scenario_recall(np.array(["a", "a", "b", ""], dtype=object), y, pred)
        self.assertEqual((r["a"]["recall"], r["b"]["recall"]), (0.5, 1.0))
        ci = bootstrap_ci(np.array([1, 1, 0, 0] * 10), np.array([.9, .8, .2, .1] * 10), 0.5, 50, 0)
        self.assertAlmostEqual(ci["f1"]["low"], 1.0)
        agg = aggregate([{"precision": 0.5, "recall": 1.0}, {"precision": 0.7, "recall": 1.0}], ("precision", "recall"))
        self.assertAlmostEqual(agg["precision"]["mean"], 0.6)
        self.assertEqual(agg["recall"]["std"], 0.0)


class PreprocessorTests(unittest.TestCase):
    def test_fit_on_train_only_statistics(self):
        tr = pd.DataFrame({"a": [0.0, 1.0, 3.0], "b": [1.0, 2.0, 3.0]})
        te = pd.DataFrame({"a": [1000.0], "b": [100.0]})
        p = Preprocessor(["a", "b"], ["a"]).fit(tr)
        self.assertAlmostEqual(float(p.scaler.mean_[0]), float(np.log1p(tr.a).mean()))
        self.assertAlmostEqual(float(p.scaler.mean_[1]), 2.0)          # test row did not influence the fit
        self.assertTrue(np.isfinite(p.transform_scaled(te)).all())

    def test_negative_log_feature_rejected(self):
        with self.assertRaises(ValueError):
            Preprocessor(["a"], ["a"]).fit(pd.DataFrame({"a": [-1.0, 1.0]}))


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg, cls.out = prepared(cls.tmp.name)
        cls.info = train_models(cls.out, cls.cfg)
        cls.rep = Path(cls.tmp.name) / "reports"
        cls.res = evaluate_models(cls.out, cls.cfg, reports_root=cls.rep)
        cls.metrics = json.loads((cls.rep / "metrics" / "t" / "all" / "metrics.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_and_figures_exist(self):
        m = self.rep / "metrics" / "t" / "all"
        for f in ("metrics.json", "comparison.csv", "seeds.csv", "latency.json"):
            self.assertTrue((m / f).exists(), f)
        for f in ("pr_curves.png", "score_distributions.png", "confusion_matrices.png", "threshold_tradeoff.png"):
            self.assertGreater((self.rep / "figures" / "t" / "all" / f).stat().st_size, 1000, f)
        self.assertEqual(len(self.info["model_ids"]), 1 + 3 + 2)       # rule + 3 seeds + ocsvm + lof

    def test_registry_records_features_params_threshold(self):
        reg = ModelRegistry.load(Path(self.info["models_dir"]))
        names = json.loads((self.out / "feature_names.json").read_text())["feature_names"]
        for mid, e in reg.entries.items():
            self.assertEqual(e["features"], names)
            self.assertIn("threshold", e)
            self.assertEqual(len(e["version"]), 10)
        self.assertEqual(reg.entries["iforest_seed42"]["params"]["n_estimators"], 200)

    def test_all_models_rank_attacks_above_normal_on_average(self):
        # direction convention: higher = more anomalous
        # the rule baseline sees only failed_count, so it cannot rank off-hours / new-IP attacks
        for fam, d in self.metrics["models"].items():
            self.assertGreater(d["metrics"]["roc_auc"], 0.5 if fam == "rule" else 0.8, fam)

    def test_models_beat_chance_pr_auc(self):
        for fam, d in self.metrics["models"].items():
            self.assertGreater(d["metrics"]["pr_auc"], 2 * d["metrics"]["prevalence"], fam)

    def test_reported_threshold_is_the_validation_threshold(self):
        reg = ModelRegistry.load(Path(self.info["models_dir"]))
        for fam, d in self.metrics["models"].items():
            self.assertEqual(d["metrics"]["threshold"], reg.entries[d["model_id"]]["threshold"])

    def test_test_rows_never_influence_training_artifacts(self):
        """Corrupt every test-split feature and label, retrain: thresholds + registry must be identical."""
        with tempfile.TemporaryDirectory() as t2:
            cfg, out2 = prepared(t2, sub="b")
            a = train_models(out2, cfg, tag="before")
            csv = out2 / "window_features.csv"
            df = pd.read_csv(csv, float_precision="round_trip")
            test = df.split == "test"
            rng = np.random.default_rng(1)
            names = json.loads((out2 / "feature_names.json").read_text())["feature_names"]
            for f in names:
                df[f] = df[f].astype(float)
                df.loc[test, f] = df.loc[test, f] * 37 + rng.random(int(test.sum()))
            df.loc[test, "label"] = 1 - df.loc[test, "label"].fillna(0)
            df.to_csv(csv, index=False)
            train_models(out2, cfg, tag="after")
            for f in ("thresholds.json", "registry.json"):
                x = (out2 / "models" / "before" / f).read_text().replace('"tag": "before"', '"tag": "T"')
                y = (out2 / "models" / "after" / f).read_text().replace('"tag": "after"', '"tag": "T"')
                self.assertEqual(x, y, f)

    def test_training_is_deterministic(self):
        with tempfile.TemporaryDirectory() as t2:
            cfg, out2 = prepared(t2, sub="c")
            train_models(out2, cfg, tag="x"); train_models(out2, cfg, tag="y")
            for f in ("thresholds.json",):
                self.assertEqual((out2 / "models" / "x" / f).read_bytes(), (out2 / "models" / "y" / f).read_bytes())

    def test_unlabeled_dataset_refused(self):
        with tempfile.TemporaryDirectory() as t2:
            cfg, out2 = prepared(t2, sub="d", labels=False)
            with self.assertRaises(ValueError):
                train_models(out2, cfg)

    def test_ablation_ready_feature_subset_and_rule_skipped_when_feature_removed(self):
        names = json.loads((self.out / "feature_names.json").read_text())["feature_names"]
        sub = [f for f in names if f not in ("failed_count",)]
        info = train_models(self.out, self.cfg, features=sub, tag="no_failed_count")
        self.assertNotIn("rule", info["model_ids"])
        with self.assertRaises(ValueError):
            train_models(self.out, self.cfg, features=["not_a_feature"], tag="bad")


if __name__ == "__main__":
    unittest.main()
