import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import _path  # noqa: F401
from logad.config import load_config
from logad.features.baselines import add_baseline_features
from logad.services.data import generate_synthetic_dataset
from logad.services.prepare import prepare_dataset
from logad.threatintel.blocklist import BlocklistError, load_blocklist

SMALL = {"synthetic": {"days": 8, "n_users": 8, "incidents_per_type": 4, "n_blocklist_decoys": 20}}


def build(tmp: Path, overrides=None, sub="a", labels=True):
    ov = {"synthetic": dict(SMALL["synthetic"])}
    for k, v in (overrides or {}).items():
        ov.setdefault(k, {}).update(v)
    cfg = load_config(overrides=ov)
    cfg["threat_intel"]["blocklist_path"] = str(tmp / sub / "ti" / "blocklist_synthetic.txt")
    cfg["threat_intel"]["manifest_path"] = str(tmp / sub / "ti" / "manifest.json")
    gen = generate_synthetic_dataset(cfg, tmp / sub / "syn", tmp / sub / "ti")
    meta = prepare_dataset(gen["files"]["log"], tmp / sub / "out", cfg,
                           gen["files"]["labels"] if labels else None, "t")
    return cfg, gen, meta, tmp / sub / "out"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg, cls.gen, cls.meta, cls.out = build(Path(cls.tmp.name))
        cls.df = pd.read_csv(cls.out / "window_features.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_feature_names_json_matches_csv_columns(self):
        names = json.loads((self.out / "feature_names.json").read_text())["feature_names"]
        extra = ["entity", "window_start", "split", "label", "attack_scenario", "history_windows", "low_history"]
        self.assertEqual(list(self.df.columns), extra + names)
        self.assertNotIn("geo_country_change", names)          # dropped, not emitted as a constant
        self.assertNotIn("distinct_users", names)

    def test_chronological_split_has_no_overlap(self):
        d = self.df.assign(t=pd.to_datetime(self.df.window_start))
        tr, va, te = (d[d.split == s].t for s in ("train", "val", "test"))
        self.assertLess(tr.max(), va.min())
        self.assertLess(va.max(), te.min())

    def test_every_split_has_attacks_and_labels_are_binary(self):
        for sp in ("train", "val", "test"):
            self.assertGreater((self.df[(self.df.split == sp)].label == 1).sum(), 0, sp)
        self.assertTrue(set(self.df.label.dropna().unique()) <= {0, 1})

    def test_attack_windows_look_different(self):
        m = self.df.groupby("label")[["failed_count", "event_rate_z", "new_ip_count"]].mean()
        self.assertGreater(m.loc[1, "failed_count"], m.loc[0, "failed_count"] * 5)

    def test_no_nan_in_features(self):
        names = json.loads((self.out / "feature_names.json").read_text())["feature_names"]
        self.assertFalse(self.df[names].isna().any().any())

    def test_baselines_exclude_attack_windows_and_non_train(self):
        n_fit = self.meta["baseline_fit"]["windows_used"]
        tr = self.df[self.df.split == "train"]
        self.assertEqual(n_fit, int((tr.label != 1).sum()))

    def test_deterministic_rerun_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as t2:
            *_, out2 = build(Path(t2))
            for f in ("window_features.csv", "feature_names.json", "baselines.json", "parse_report.json"):
                self.assertEqual((self.out / f).read_bytes(), (out2 / f).read_bytes(), f)

    def test_parse_report_records_noise(self):
        rep = json.loads((self.out / "parse_report.json").read_text())
        self.assertEqual(rep["malformed_lines"], 3)
        self.assertGreater(rep["not_sshd_lines"], 0)
        self.assertGreater(rep["ignored_redundant_lines"] + rep["event_lines"], 0)

    def test_unlabeled_run_has_unknown_labels(self):
        with tempfile.TemporaryDirectory() as t2:
            _, _, meta, out2 = build(Path(t2), labels=False)
            df = pd.read_csv(out2 / "window_features.csv")
            self.assertTrue(df.label.isna().all())
            self.assertFalse(meta["has_labels"])
            self.assertEqual(meta["split"]["per_split"]["train"]["normal_windows"], 0)

    def test_year_rollover_end_to_end(self):
        with tempfile.TemporaryDirectory() as t2:
            _, _, meta, out2 = build(Path(t2), {"synthetic": {"start_month_day": "12-28"}, "dataset": {"log_year": 2025}})
            ts = pd.to_datetime(pd.read_csv(out2 / "window_features.csv").window_start)
            self.assertEqual((ts.min().year, ts.max().year), (2025, 2026))
            self.assertLess((ts.max() - ts.min()).days, 10)

    def test_src_ip_grouping_runs(self):
        with tempfile.TemporaryDirectory() as t2:
            _, _, _, out2 = build(Path(t2), {"windowing": {"group_by": "src_ip"}})
            names = json.loads((out2 / "feature_names.json").read_text())["feature_names"]
            self.assertIn("distinct_users", names)
            self.assertNotIn("distinct_src_ips", names)


class LeakageTests(unittest.TestCase):
    """Changing validation/test windows must not change anything learned or any train feature."""

    def test_train_features_and_baselines_ignore_val_test(self):
        from logad.features.extractors import compute_raw_windows
        from logad.features.split import chronological_split
        from logad.features.windowing import assign_windows
        from logad.parsing.normalizer import attach_labels
        from logad.parsing.ssh_parser import parse_file
        with tempfile.TemporaryDirectory() as t:
            cfg, gen, _, _ = build(Path(t))
            ev, _ = parse_file(gen["files"]["log"], 2025)
            ev = attach_labels(ev, pd.read_csv(gen["files"]["labels"]))
            ev, _ = assign_windows(ev, 300, "user")
            raw = compute_raw_windows(ev, group_by="user", tz_offset_hours=0, success_after_failed_min=5,
                                      off_hours_start=8, off_hours_end=19)
            split, _ = chronological_split(raw["window_start"], 0.6, 0.2, 0.2)
            raw["split"] = split
            fit = (split == "train") & ~(raw["label"] == 1).to_numpy()
            a, ba = add_baseline_features(raw, split, fit, "user", 5, 20.0)

            mutated = raw.copy()
            mask = (split != "train")
            mutated.loc[mask, "event_count"] = mutated.loc[mask, "event_count"] * 50 + 999
            mutated.loc[mask, "hour_local"] = 3.0
            mutated.loc[mask, "_ips"] = pd.Series([frozenset({"9.9.9.9"})] * int(mask.sum()), index=mutated.index[mask])
            b, bb = add_baseline_features(mutated, split, fit, "user", 5, 20.0)

            self.assertEqual(ba.to_dict(), bb.to_dict())
            cols = ["event_rate_z", "hour_deviation", "new_ip_count"]
            pd.testing.assert_frame_equal(a.loc[~mask, cols], b.loc[~mask, cols])

    def test_fit_mask_cannot_select_non_train_rows(self):
        raw = pd.DataFrame({"entity": ["a", "a"], "event_count": [1, 1], "hour_local": [9.0, 9.0],
                            "_ips": [frozenset(), frozenset()],
                            "window_start": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)})
        with self.assertRaises(ValueError):
            add_baseline_features(raw, np.array(["train", "test"]), np.array([True, True]), "user", 1, 20.0)


class BlocklistTests(unittest.TestCase):
    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            cfg, gen, _, _ = build(Path(t))
            bl = Path(gen["files"]["blocklist"])
            ok = load_blocklist(bl, bl.parent / "manifest.json")
            self.assertTrue(ok.verified)
            bl.write_text(bl.read_text() + "8.8.8.8\t1\n")
            with self.assertRaises(BlocklistError):
                load_blocklist(bl, bl.parent / "manifest.json")


if __name__ == "__main__":
    unittest.main()
