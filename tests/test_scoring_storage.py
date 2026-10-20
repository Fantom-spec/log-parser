import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import _path  # noqa: F401
from logad.dashboard import TEMPLATE, _safe_json, build_dashboard
from logad.io import load_json
from logad.preprocessing.pipeline import load_dataset
from logad.scoring.rules import TIERS, apply_rules, rule_floor_index
from logad.scoring.scorer import Scorer
from logad.scoring.tiers import assign_tier_index, calibrate_tiers, thresholds_from_json
from logad.services.ingest import ingest_dataset, score_log_to_db
from logad.services.scoring import calibrate_scoring, evaluate_scoring
from logad.services.train import train_models
from logad.storage.db import Database
from test_models_eval import prepared

RULES = {"r1": {"enabled": True, "min_failed": 10, "min_distinct_ips": 5}, "r2": {"enabled": True},
         "r3": {"enabled": True, "min_failed": 1}}
FLOORS = {"R1": "HIGH", "R2": "CRITICAL", "R3": "MEDIUM"}


class TierRuleUnitTests(unittest.TestCase):
    def test_tiers_are_validation_precision_bands(self):
        y = np.array([0] * 20 + [1] * 10)
        s = np.r_[np.linspace(0.0, 0.5, 20), np.linspace(0.4, 1.0, 10)]
        c = calibrate_tiers(y, s, 0.4, {"MEDIUM": 0.75, "HIGH": 0.9, "CRITICAL": 0.97}, min_tp=3)
        t = c["thresholds"]
        self.assertLessEqual(t["LOW"], t["MEDIUM"]); self.assertLessEqual(t["MEDIUM"], t["HIGH"]); self.assertLessEqual(t["HIGH"], t["CRITICAL"])
        m = s >= t["HIGH"]
        self.assertGreaterEqual(y[m].mean(), 0.9)
        self.assertGreaterEqual(y[m].sum(), 3)

    def test_unreachable_tier_is_infinite_and_reported(self):
        y = np.array([0, 1, 0, 1, 0, 0]); s = np.array([0.1, 0.5, 0.6, 0.55, 0.2, 0.7])
        c = calibrate_tiers(y, s, 0.5, {"MEDIUM": 0.75, "HIGH": 0.9, "CRITICAL": 0.97}, min_tp=3)
        self.assertEqual(c["thresholds"]["CRITICAL"], float("inf"))
        self.assertFalse(c["info"]["CRITICAL"]["reachable"])
        self.assertEqual(assign_tier_index([0.99], c["thresholds"])[0], 1)      # only LOW is reachable by score

    def test_merged_cutpoints_are_flagged_and_higher_tier_wins(self):
        y = np.array([0] * 10 + [1] * 5); s = np.r_[np.linspace(0, .4, 10), np.linspace(.8, 1, 5)]
        c = calibrate_tiers(y, s, 0.8, {"MEDIUM": 0.75, "HIGH": 0.9, "CRITICAL": 0.97}, min_tp=3)
        self.assertEqual(c["info"]["CRITICAL"].get("merged_with"), "HIGH")
        self.assertEqual(TIERS[assign_tier_index([1.0], c["thresholds"])[0]], "CRITICAL")

    def test_json_null_threshold_restored_to_inf(self):
        self.assertEqual(thresholds_from_json({"LOW": 0.5, "CRITICAL": None})["CRITICAL"], float("inf"))

    def test_rules_and_floors(self):
        df = pd.DataFrame({"failed_count": [0, 12, 3, 2], "distinct_src_ips": [1, 6, 1, 1],
                           "success_after_failed": [0, 0, 1, 0], "blocklisted_ip_count": [0, 0, 0, 1]})
        h, off = apply_rules(df, RULES)
        self.assertEqual(h["R1"].tolist(), [False, True, False, False])
        self.assertEqual(h["R2"].tolist(), [False, False, True, False])
        self.assertEqual(h["R3"].tolist(), [False, False, False, True])
        self.assertEqual(rule_floor_index(h, FLOORS).tolist(), [0, 3, 4, 2])
        self.assertEqual(off, [])

    def test_rules_unavailable_without_features_or_when_disabled(self):
        df = pd.DataFrame({"failed_count": [5], "success_after_failed": [0]})
        h, off = apply_rules(df, {**RULES, "r2": {"enabled": False}})
        self.assertFalse(h.to_numpy().any())
        self.assertTrue(any("R1 unavailable" in o for o in off) and any("R2 disabled" in o for o in off))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg, cls.out = prepared(cls.tmp.name)
        train_models(cls.out, cls.cfg)
        cls.cal = calibrate_scoring(cls.out, cls.cfg)
        cls.rep = Path(cls.tmp.name) / "reports"
        cls.ev = evaluate_scoring(cls.out, cls.cfg, reports_root=cls.rep)["result"]
        cls.db = Path(cls.tmp.name) / "t.db"
        cls.ing = ingest_dataset(cls.out, cls.cfg, db_path=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_replay_features_match_training_features_on_val_and_test(self):
        df, names, meta = load_dataset(self.out)
        sc = Scorer(self.out, self.cfg)
        table, _ = sc.feature_table_from_log(meta["inputs"]["log"])
        df["ws"] = df.window_start.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        m = df.merge(table, left_on=["entity", "ws"], right_on=["entity", "window_start"], suffixes=("", "_log"))
        self.assertEqual(len(m), len(df))
        for sp in ("val", "test"):
            s = m[m.split == sp]
            for f in names:
                np.testing.assert_allclose(s[f].to_numpy(float), s[f + "_log"].to_numpy(float), atol=1e-9, err_msg=f)

    def test_model_tier_low_equals_phase2_threshold_decision(self):
        df, _, _ = load_dataset(self.out)
        sc = Scorer(self.out, self.cfg)
        scored = sc.score_features(df[df.split == "test"])
        thr = load_json(self.out / "models" / "all" / "thresholds.json")[sc.model_id]["threshold"]
        self.assertEqual((scored.model_tier_idx >= 1).tolist(), (scored.score >= thr).tolist())

    def test_rules_never_lower_a_tier_and_combined_alerts_superset(self):
        for sp in self.ev["splits"].values():
            mo, mr = sp["configs"]["model_only"], sp["configs"]["model_plus_rules"]
            self.assertGreaterEqual(mr["n_alerts"], mo["n_alerts"])
            self.assertGreaterEqual(mr["recall"], mo["recall"])
            tm = sum(v["windows"] for t, v in sp["model_tiers"].items() if t != "NORMAL")
            tf = sum(v["windows"] for t, v in sp["tiers"].items() if t != "NORMAL")
            self.assertGreaterEqual(tf, tm)

    def test_scoring_output_structure(self):
        t = self.ev["splits"]["test"]
        self.assertEqual(t["role"], "held-out")
        self.assertEqual(sum(v["windows"] for v in t["tiers"].values()), t["n_windows"])
        self.assertTrue((self.rep / "figures" / "t" / "all" / "scoring.png").stat().st_size > 1000)
        self.assertTrue(self.ev["caveats"])

    def test_tier_calibration_ignores_test_rows(self):
        with tempfile.TemporaryDirectory() as t2:
            cfg, out2 = prepared(t2, sub="z")
            train_models(out2, cfg)
            a = calibrate_scoring(out2, cfg)
            csv = out2 / "window_features.csv"
            df = pd.read_csv(csv, float_precision="round_trip")
            names = load_json(out2 / "feature_names.json")["feature_names"]
            mask = df.split == "test"
            for f in names:
                df[f] = df[f].astype(float); df.loc[mask, f] = df.loc[mask, f] * 50 + 3
            df.loc[mask, "label"] = 1 - df.loc[mask, "label"].fillna(0)
            df.to_csv(csv, index=False)
            b = calibrate_scoring(out2, cfg)
            self.assertEqual(a["tier_thresholds"], b["tier_thresholds"])
            self.assertEqual(a["profile"], b["profile"])

    def test_evidence_lists_rules_and_deviations_for_flagged_only(self):
        df, _, _ = load_dataset(self.out)
        scored = Scorer(self.out, self.cfg).score_features(df[df.split == "test"])
        flagged = scored[scored.tier != "NORMAL"]
        self.assertGreater(len(flagged), 0)
        self.assertTrue(all(e == [] for e in scored[scored.tier == "NORMAL"].evidence))
        any_rule = flagged[flagged.rules.map(len) > 0]
        if len(any_rule):
            self.assertTrue(any(x.startswith("R") for x in any_rule.evidence.iloc[0]))

    # ---- storage ----------------------------------------------------------------------
    def test_ingest_counts_and_idempotence(self):
        c = self.ing["counts"]
        df, names, _ = load_dataset(self.out)
        self.assertEqual((c["window_features"], c["detections"], c["runs"]), (len(df), len(df), 1))
        again = ingest_dataset(self.out, self.cfg, db_path=self.db)
        self.assertEqual(again["counts"]["window_features"], len(df))
        self.assertEqual(again["counts"]["runs"], 1)                  # previous replay run replaced
        self.assertEqual(again["counts"]["detections"], len(df))      # old detections cascaded away

    def test_feature_columns_follow_feature_list(self):
        with Database(self.db) as db:
            cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(window_features)")}
            names = load_json(self.out / "feature_names.json")["feature_names"]
            self.assertTrue(set(names) <= cols)
            self.assertNotIn("geo_country_change", cols)
            db.ensure_feature_columns(["brand_new_feature"])
            self.assertIn("brand_new_feature", {r["name"] for r in db.conn.execute("PRAGMA table_info(window_features)")})
            with self.assertRaises(ValueError):
                db.ensure_feature_columns(['x"; DROP TABLE runs; --'])

    def test_values_are_bound_not_interpolated(self):
        with tempfile.TemporaryDirectory() as t2:
            p = Path(t2) / "i.db"
            evil = "bob'; DROP TABLE detections; --"
            scored = pd.DataFrame({"entity": [evil], "window_start": ["2025-01-01T00:00:00Z"], "split": ["scored"],
                                   "label": [np.nan], "attack_scenario": [None], "score": [0.9], "model_tier": ["LOW"],
                                   "tier": ["LOW"], "rules": [[]], "evidence": [["<img src=x onerror=alert(1)>"]]})
            with Database(p) as db:
                rid = db.start_run("t", "d", "m", "v")
                db.put_detections(rid, scored)
                rows = db.detections(rid)
            self.assertEqual(rows[0]["entity"], evil)
            self.assertEqual(sqlite3.connect(p).execute("SELECT COUNT(*) FROM detections").fetchone()[0], 1)

    def test_score_log_to_db_stores_events_and_escapes_nothing_into_markup(self):
        with tempfile.TemporaryDirectory() as t2:
            log = Path(t2) / "x.log"
            lines = ["Mar  3 10:00:%02d bastion01 sshd[1]: Failed password for <script>alert(1)</script> from 6.6.6.6 port 1 ssh2" % s
                     for s in range(12)]
            log.write_text("\n".join(lines) + "\n")
            r = score_log_to_db(self.out, self.cfg, log, db_path=Path(t2) / "s.db", year=2025)
            with Database(Path(t2) / "s.db") as db:
                ev = db.conn.execute("SELECT username, raw_line FROM events WHERE run_id=?", (r["run_id"],)).fetchall()
            self.assertEqual(len(ev), 12)
            self.assertIn("<script>", ev[0]["username"])              # stored verbatim; escaping is the renderer's job
            self.assertEqual(len(r["scored"]), 1)

    def test_run_delete_cascades(self):
        with tempfile.TemporaryDirectory() as t2:
            shutil.copy(self.db, Path(t2) / "c.db")
            with Database(Path(t2) / "c.db") as db:
                db.conn.execute("DELETE FROM runs")
                self.assertEqual(db.counts()["detections"], 0)

    # ---- dashboard ----------------------------------------------------------------------
    def test_dashboard_build_and_embedded_json_is_script_safe(self):
        out = Path(self.tmp.name) / "dash.html"
        path = build_dashboard(self.cfg, "t", "all", out, reports_root=self.rep, db_path=self.db)
        h = Path(path).read_text()
        blob = h.split('<script id="data" type="application/json">')[1].split("</script>")[0]
        self.assertNotIn("<", blob)
        self.assertIn("Content-Security-Policy", h)
        self.assertNotIn("innerHTML", h)
        self.assertEqual(json.loads(blob)["dataset"], "t")

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_dashboard_js_runs_and_never_injects_markup(self):
        out = Path(self.tmp.name) / "dash2.html"
        build_dashboard(self.cfg, "t", "all", out, reports_root=self.rep, db_path=self.db)
        r = subprocess.run(["node", str(Path(__file__).parent / "dashboard_shim.js"), str(out)],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        self.assertGreater(res["nodes"], 100)
        self.assertTrue(all(i.startswith("data:image/png;base64") for i in res["imgs"]))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_hostile_log_text_stays_text(self):
        evil = "</script><img src=x onerror=alert(1)>"
        data = {"dataset": "d", "tag": "all", "models": None, "scoring": None, "ablation": None, "h1": None, "figures": [],
                "run": {"run_id": 1, "model_id": "m", "model_version": "v", "created_at": "x"},
                "counts": {"windows": 1, "shown": 1, "by_tier": {"NORMAL": 0, "LOW": 1, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}},
                "detections": [{"entity": evil, "window_start": "2025-01-01T00:00:00Z", "split": "scored", "label": None,
                                "scenario": evil, "score": 0.9, "model_tier": "LOW", "tier": "LOW", "rules": [],
                                "evidence": [evil], "features": {evil: 1.0}}]}
        p = Path(self.tmp.name) / "evil.html"
        p.write_text(TEMPLATE.replace("__DATA__", _safe_json(data)))
        r = subprocess.run(["node", str(Path(__file__).parent / "dashboard_shim.js"), str(p)],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        self.assertIn(evil, res["texts"])                          # present as a text node, never as an element
        self.assertEqual(res["imgs"], [])


if __name__ == "__main__":
    unittest.main()
