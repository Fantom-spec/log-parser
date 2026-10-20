import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import _path  # noqa: F401
from logad.evaluation.ablation import make_variants, paired_bootstrap, resample_indices, verdict
from logad.preprocessing.pipeline import load_dataset
from logad.services.experiments import run_ablation, run_h1
from test_models_eval import prepared

FAMILIES = {"a": ["f1", "f2"], "b": ["f3"], "c": ["f4"]}
ORDER = ["f1", "f2", "f3", "f4"]


class HelperTests(unittest.TestCase):
    def test_make_variants(self):
        v = {x["name"]: x for x in make_variants(FAMILIES, ORDER)}
        self.assertEqual(v["all"]["features"], ORDER)
        self.assertEqual(v["drop_a"]["features"], ["f3", "f4"])
        self.assertEqual(v["only_b"]["features"], ["f3"])
        self.assertEqual(len(v), 1 + 3 + 3)

    def test_make_variants_skips_empty_drop(self):
        names = [x["name"] for x in make_variants({"a": ["f1"]}, ["f1"])]
        self.assertEqual(names, ["all", "only_a"])

    def test_verdict_labels(self):
        self.assertEqual(verdict([0.01, 0.05]), "supported")
        self.assertEqual(verdict([-0.05, -0.01]), "reversed")
        self.assertEqual(verdict([-0.05, 0.01]), "inconclusive")
        self.assertEqual(verdict([0.0, 0.0]), "no_change")
        self.assertEqual(verdict([0.01, 0.05], "helps", "hurts"), "helps")

    def test_paired_bootstrap_detects_clear_gap_and_zero_for_identical(self):
        rng = np.random.default_rng(0)
        y = np.array([1] * 40 + [0] * 160)
        good = np.r_[rng.uniform(0.6, 1, 40), rng.uniform(0, 0.5, 160)]
        bad = rng.uniform(0, 1, 200)
        idx = resample_indices(y, 200, 1)
        d = paired_bootstrap(y, good, 0.55, bad, 0.55, idx)
        self.assertTrue(d["delta_pr_auc_distinguishable"])
        self.assertGreater(d["delta_pr_auc"], 0.3)
        same = paired_bootstrap(y, good, 0.55, good, 0.55, idx)
        self.assertEqual((same["delta_pr_auc"], same["delta_pr_auc_ci"]), (0.0, [0.0, 0.0]))

    def test_resamples_are_shared_and_contain_attacks(self):
        y = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        a, b = resample_indices(y, 50, 3), resample_indices(y, 50, 3)
        self.assertTrue(all((x == z).all() for x, z in zip(a, b)))
        self.assertTrue(all(y[i].sum() > 0 for i in a))


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg, cls.out = prepared(cls.tmp.name)
        cls.rep = Path(cls.tmp.name) / "reports"
        cls.abl = run_ablation(cls.out, cls.cfg, reports_root=cls.rep)
        cls.h1 = run_h1(cls.out, cls.cfg, reports_root=cls.rep, work_dir=Path(cls.tmp.name) / "h1work")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ablation_variants_use_exactly_the_requested_features(self):
        fam = json.loads((self.out / "feature_names.json").read_text())["families"]
        names = json.loads((self.out / "feature_names.json").read_text())["feature_names"]
        by = {r["variant"]: r for r in self.abl["results"]}
        self.assertEqual(set(by["drop_threat_intel"]["features"]), set(names) - set(fam["threat_intel"]))
        self.assertEqual(by["only_time_patterns"]["features"], fam["time_patterns"])
        self.assertIn("threat_intel", fam)                       # the new family is part of the ablation
        self.assertIn("drop_threat_intel", by)

    def test_rule_baseline_skipped_when_its_feature_is_removed(self):
        by = {r["variant"]: r for r in self.abl["results"]}
        self.assertNotIn("rule", by["drop_failed_logins"]["models"])
        self.assertIn("rule", by["drop_time_patterns"]["models"])
        self.assertEqual(by["drop_time_patterns"]["delta_vs_all"]["rule"]["delta_pr_auc"], 0.0)

    def test_ablation_outputs(self):
        m = Path(self.abl["metrics_dir"])
        data = json.loads((m / "ablation.json").read_text())
        self.assertGreater(data["n_test_attacks"], 0)
        self.assertTrue(data["caveats"])
        self.assertIn("seed_paired", data["variants"][1]["delta_vs_all"]["iforest"])
        csv = pd.read_csv(m / "ablation.csv")
        self.assertTrue({"variant", "model", "pr_auc", "delta_pr_auc", "delta_pr_auc_ci_low"} <= set(csv.columns))
        self.assertGreater((Path(self.abl["figures_dir"]) / "ablation.png").stat().st_size, 1000)

    def test_ablation_is_deterministic(self):
        again = run_ablation(self.out, self.cfg, reports_root=Path(self.tmp.name) / "reports2")
        a = (Path(self.abl["metrics_dir"]) / "ablation.json").read_bytes()
        b = (Path(again["metrics_dir"]) / "ablation.json").read_bytes()
        self.assertEqual(a, b)

    def test_h1_arms_are_row_aligned_and_differ_only_in_baseline_features(self):
        pu, _, _ = load_dataset(Path(self.tmp.name) / "h1work" / "per_user")
        gl, _, _ = load_dataset(Path(self.tmp.name) / "h1work" / "global")
        keys = ["entity", "window_start", "split", "label"]
        self.assertTrue(pu[keys].equals(gl[keys]))
        allowed = {"event_rate_z", "hour_deviation", "new_ip_count"}
        self.assertTrue(set(self.h1["features_that_differ"]) <= allowed)
        self.assertTrue(gl["low_history"].all())                 # global arm: nobody uses an own baseline

    def test_h1_outputs_have_verdicts_and_history_breakdown(self):
        for fam in ("iforest", "ocsvm", "lof"):
            e = self.h1["models"][fam]
            self.assertIn(e["delta"]["verdict_pr_auc"], {"supported", "reversed", "inconclusive", "no_change"})
            self.assertIn("low_history", e["by_history"]["per_user"])
        self.assertGreater((Path(self.h1["figures_dir"]) / "h1.png").stat().st_size, 1000)

    def test_experiments_refuse_unlabeled_datasets(self):
        with tempfile.TemporaryDirectory() as t2:
            cfg, out2 = prepared(t2, sub="u", labels=False)
            with self.assertRaises(ValueError):
                run_ablation(out2, cfg, reports_root=Path(t2) / "r")


if __name__ == "__main__":
    unittest.main()
