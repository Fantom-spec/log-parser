import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import _path  # noqa: F401
from logad.app.service import AppState, ApiError, realdata_get, realdata_list
from logad.config import merge_config
from logad.io import load_json
from logad.parsing.ssh_parser import SSHParser, parse_file
from logad.scoring.scorer import Scorer
from logad.services.fetch import download, fetch_blocklist, fetch_loghub
from logad.services.realdata import run_real_analysis
from logad.services.robustness import measure
from logad.services.scoring import calibrate_scoring
from logad.services.train import train_models
from logad.threatintel.blocklist import load_blocklist
from test_models_eval import prepared

LINE = "Mar  3 10:00:{s:02d} h sshd[1]: Failed password for {u} from {ip} port 1 ssh2"


def keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from keys(v)


class ParserRobustness(unittest.TestCase):
    def test_dedupe_exact_lines_and_default_off(self):
        lines = [LINE.format(s=s, u="bob", ip="1.2.3.4") for s in range(5)]
        once = SSHParser(2025).parse_lines(lines)
        twice = SSHParser(2025).parse_lines(lines + lines)
        dd = SSHParser(2025, dedupe=True).parse_lines(lines + lines)
        self.assertEqual(len(twice.events), 10)                     # off by default: identical real events are kept
        self.assertEqual(len(dd.events), 5)
        self.assertEqual(dd.report.duplicate_lines_dropped, 5)
        self.assertEqual(once.frame()["ts"].tolist(), dd.frame()["ts"].tolist())

    def test_out_of_order_lines_are_counted_and_resorted(self):
        lines = [LINE.format(s=s, u="bob", ip="1.2.3.4") for s in (5, 1, 9, 3)]
        p = SSHParser(2025).parse_lines(lines)
        self.assertEqual(p.report.timestamp_regressions, 2)
        self.assertEqual([t.second for t in p.frame()["ts"]], [1, 3, 5, 9])

    def test_report_accounts_for_every_line_under_fuzz(self):
        rnd = random.Random(7)
        junk = ["", " ", "\x00\x01\x02", "é" * 50, "Mar  3 25:61:61 h sshd[1]: x", "Mar 32 10:00:00 h sshd[1]: x",
                "Foo  3 10:00:00 h sshd[1]: x", "Mar  3 10:00:00 h sshd[1]: " + "A" * 200_000,
                "Mar  3 10:00:00 h sshd[1]: Failed password for " + "ü" * 5000 + " from not-an-ip port 1 ssh2",
                "Mar  3 10:00:00 h sshd[1]: message repeated 99999999999 times: [ Failed password for a from 1.1.1.1 port 1 ssh2]",
                "Mar  3 10:00:00 h sshd[1]: Invalid user  from 1.1.1.1", "Mar  3 10:00:00 h CRON[1]: x"]
        lines = [rnd.choice(junk) if rnd.random() < .7 else LINE.format(s=rnd.randrange(60), u=rnd.choice("abc"), ip="1.1.1.1")
                 for _ in range(400)]
        p = SSHParser(2025, dedupe=True).parse_lines(lines)
        r = p.report
        accounted = (r.blank_lines + r.malformed_lines + r.invalid_timestamp_lines + r.not_sshd_lines + r.unsupported_sshd_lines
                     + r.event_lines + sum(r.ignored_redundant.values()) + r.duplicate_lines_dropped)
        self.assertEqual(accounted, r.total_lines)                  # nothing silently dropped
        df = p.frame()
        self.assertTrue((df["repeat"] >= 1).all())

    def test_binary_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "bin.log"
            f.write_bytes(os.urandom(50_000) + b"\n" + LINE.format(s=1, u="a", ip="1.1.1.1").encode() + b"\n")
            df, rep = parse_file(f, 2025)
            self.assertEqual(rep.total_lines, rep.blank_lines + rep.malformed_lines + rep.invalid_timestamp_lines + rep.not_sshd_lines
                             + rep.unsupported_sshd_lines + rep.event_lines + sum(rep.ignored_redundant.values()))


class Fetchers(unittest.TestCase):
    def test_blocklist_snapshot_via_file_url_writes_verifiable_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src = t / "ipsum.txt"
            src.write_text("# header\n1.2.3.4\t5\n5.6.7.8\t3\nnot-an-ip\t1\n")
            cfg = merge_config({"fetch": {"max_bytes": 10_000, "timeout_seconds": 5}, "real": {}}, {})
            info = fetch_blocklist(cfg, src.as_uri(), t / "out" / "ipsum.txt", t / "out" / "manifest.json")
            bl = load_blocklist(t / "out" / "ipsum.txt", t / "out" / "manifest.json")
            self.assertTrue(bl.verified); self.assertEqual(bl.ips, frozenset({"1.2.3.4", "5.6.7.8"}))
            e = load_json(t / "out" / "manifest.json")["files"][0]
            self.assertEqual((e["sha256"], e["url"]), (info["sha256"], src.as_uri()))
            (t / "out" / "ipsum.txt").write_text("9.9.9.9\n")        # tampering after the snapshot is detected
            with self.assertRaises(Exception):
                load_blocklist(t / "out" / "ipsum.txt", t / "out" / "manifest.json")

    def test_download_guards(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); f = t / "a.txt"; f.write_text("x" * 1000)
            with self.assertRaises(ValueError):
                download("ftp://example.com/x", t / "o", 10)
            with self.assertRaises(ValueError):
                download(f.as_uri(), t / "o", 100)                   # size cap
            self.assertFalse((t / "o").exists() or (t / "o.part").exists())
            e = t / "e.txt"; e.write_text("")
            with self.assertRaises(ValueError):
                download(e.as_uri(), t / "o2", 100)

    def test_fetch_loghub(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); src = t / "s.log"; src.write_text(LINE.format(s=1, u="a", ip="1.1.1.1") + "\n")
            cfg = {"fetch": {"max_bytes": 10_000, "timeout_seconds": 5, "loghub_url": src.as_uri()}, "real": {"log_path": str(t / "raw" / "o.log")}}
            info = fetch_loghub(cfg)
            self.assertEqual(Path(info["path"]).read_text(), src.read_text())


class Phase5Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        cls.cfg, cls.out = prepared(cls.t)
        train_models(cls.out, cls.cfg); calibrate_scoring(cls.out, cls.cfg)
        cls.meta = load_json(cls.out / "meta.json")
        cls.log = cls.meta["inputs"]["log"]
        ti = cls.meta["config"]["threat_intel"]
        cls.cfg2 = merge_config(cls.cfg, {"real": {"blocklist_path": ti["blocklist_path"], "manifest_path": ti["manifest_path"]}})
        cls.rep = cls.t / "rep"
        cls.res = run_real_analysis(cls.log, cls.cfg2, "realx", 2025, "src_ip", out_root=cls.t / "realx", reports_root=cls.rep,
                                    cross_domain_dir=cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_accuracy_metrics_on_unlabeled_data(self):
        bad = {"precision", "recall", "f1", "pr_auc", "false_positive_rate"} & set(keys(self.res))
        self.assertEqual(bad, set())
        self.assertIn("none (unlabeled)", self.res["labels"])
        self.assertIn("WEAK PROXY", self.res["weak_proxy"]["warning"])

    def test_report_contents(self):
        r = self.res
        self.assertEqual(sum(r["tiers"]["all"].values()), r["n_windows"])
        q = list(r["score_distribution"]["quantiles"].values())
        self.assertEqual(q, sorted(q))
        sc = [w["score"] for w in r["top_windows"]]
        self.assertEqual(sc, sorted(sc, reverse=True)); self.assertLessEqual(len(sc), 25)
        self.assertTrue(r["top_windows"][0]["evidence"] or r["top_windows"][0]["rules"])
        self.assertTrue(r["parse"]["coverage_sshd_lines"] > 0.9)
        self.assertIn("unattributed_events", r)
        self.assertTrue(any("R1" in u for u in r["rules"]["unavailable"]))                           # src_ip grouping: R1 unavailable
        self.assertEqual(r["blocklist_overlap"]["windows_with_blocklisted_ip"] > 0, True)
        cd = r["cross_domain"]
        self.assertTrue(cd["available"]); self.assertIn("limitation", cd); self.assertEqual(len(cd["most_shifted_features"]), 8)

    def test_quantile_tiers_flag_a_bounded_share_of_validation(self):
        thr = self.res["tier_thresholds"]
        self.assertLessEqual(thr["LOW"], thr["MEDIUM"]); self.assertLessEqual(thr["HIGH"], thr["CRITICAL"])
        df = pd.read_csv(self.t / "realx" / "window_features.csv")
        self.assertTrue(df["label"].isna().all())                  # unlabeled stays unknown, never "normal"
        self.assertIn("quantile", json.dumps(load_json(self.t / "realx" / "models" / "unlabeled" / "scoring.json")["tier_calibration"]))

    def test_user_grouping_real_analysis_and_figure(self):
        r = run_real_analysis(self.log, self.cfg2, "realu", 2025, "user", out_root=self.t / "realu", reports_root=self.rep, cross_domain_dir=self.out)
        self.assertEqual(r["group_by"], "user")
        self.assertFalse(any("R1" in u for u in r["rules"]["unavailable"]))
        self.assertGreater((self.rep / "figures" / "realu" / "realdata" / "realdata.png").stat().st_size, 1000)

    def test_app_service_endpoints(self):
        st = AppState(self.cfg, "t", "all", self.out, self.t / "a.db", self.rep, self.t / "u")
        self.assertIn("realx", realdata_list(st)["datasets"])
        self.assertTrue(realdata_get(st, "realx")["figure"])
        with self.assertRaises(ApiError):
            realdata_get(st, "../etc")
        with self.assertRaises(ApiError):
            realdata_get(st, "missing")

    # ---- robustness of scoring ----
    def test_cold_start_user_is_scored_with_global_baseline(self):
        sc = Scorer(self.out, self.cfg)
        with tempfile.TemporaryDirectory() as t2:
            f = Path(t2) / "new.log"
            f.write_text("".join(LINE.format(s=s, u="never-seen-user", ip="9.9.9.9") + "\n" for s in range(12)))
            scored, _ = sc.score_log(f, 2025)
        r = scored.iloc[0]
        self.assertTrue(bool(r["low_history"]) and r["history_windows"] == 0)
        self.assertTrue(np.isfinite(scored[sc.features].to_numpy(float)).all())
        self.assertNotEqual(r["tier"], "NORMAL")                  # 12 failures from a brand-new IP is not normal

    def test_replay_is_idempotent_and_order_independent(self):
        sc = Scorer(self.out, self.cfg)
        a, _ = sc.score_log(self.log); b, _ = sc.score_log(self.log)
        self.assertTrue(np.array_equal(a["score"].to_numpy(), b["score"].to_numpy()))
        with tempfile.TemporaryDirectory() as t2:
            lines = Path(self.log).read_text().splitlines()[:600]
            # swap neighbouring lines (small out-of-order jitter): same windows/features after re-sorting by time
            sw = lines[:]
            for i in range(0, len(sw) - 1, 7):
                sw[i], sw[i + 1] = sw[i + 1], sw[i]
            f1, f2 = Path(t2) / "a.log", Path(t2) / "b.log"
            f1.write_text("\n".join(lines) + "\n"); f2.write_text("\n".join(sw) + "\n")
            x, _ = sc.feature_table_from_log(f1, 2025); y, _ = sc.feature_table_from_log(f2, 2025)
        key = ["entity", "window_start"]
        m = x.sort_values(key).reset_index(drop=True); n = y.sort_values(key).reset_index(drop=True)
        self.assertTrue(m[key].equals(n[key]))
        for f in sc.features:
            if f not in ("consecutive_failed_max",):                # run-order dependent by definition
                np.testing.assert_allclose(m[f].to_numpy(float), n[f].to_numpy(float), atol=1e-9, err_msg=f)

    def test_latency_and_detection_delay(self):
        r = measure(self.out, self.cfg, repeats=1, reports_root=self.rep)
        L = r["latency"]
        self.assertGreater(L["events_per_second_end_to_end"], 100)
        d = r["detection_delay"]
        self.assertEqual(d["detected"] + d["missed"], d["incidents"])
        self.assertGreater(d["detected"], d["incidents"] * 0.5)
        self.assertGreaterEqual(d["delay_seconds"]["min"], 0)
        self.assertLessEqual(d["delay_seconds"]["median"], 2 * d["window_seconds"])
        self.assertTrue((self.rep / "metrics" / "t" / "all" / "robustness.json").exists())


if __name__ == "__main__":
    unittest.main()
