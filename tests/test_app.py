import http.client
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import _path  # noqa: F401
from logad.app import service
from logad.app.jobs import JobError, JobRunner
from logad.app.server import make_server
from logad.app.service import AppState
from logad.services.evaluate import evaluate_models
from logad.services.ingest import ingest_dataset
from logad.services.realdata import run_real_analysis
from logad.services.scoring import calibrate_scoring, evaluate_scoring
from logad.services.train import train_models
from test_models_eval import prepared

HERE = Path(__file__).parent
APP_JS = HERE.parent / "src" / "logad" / "app" / "static" / "app.js"
H = {"X-Requested-With": "logad", "Content-Type": "application/json"}


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.cfg, out = prepared(t)
        train_models(out, cls.cfg); calibrate_scoring(out, cls.cfg)
        rep = t / "reports"
        evaluate_models(out, cls.cfg, reports_root=rep); evaluate_scoring(out, cls.cfg, reports_root=rep)
        ingest_dataset(out, cls.cfg, db_path=t / "app.db")
        from logad.io import load_json
        run_real_analysis(load_json(out / "meta.json")["inputs"]["log"], cls.cfg, "rt", 2025, "src_ip",
                          out_root=t / "rt", reports_root=rep, cross_domain_dir=out)
        cls.state = AppState(cls.cfg, "t", "all", out, t / "app.db", rep, t / "up")
        allow = {"echo": ([sys.executable, "-c", "print('hello'); print('world')"], "echo"),
                 "slow": ([sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"], "slow")}
        cls.jobs = JobRunner(t / "app.db", allow, cwd=t)
        cls.srv = make_server(cls.state, cls.jobs, port=0)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close(); cls.tmp.cleanup()

    def call(self, path, body=None, headers=None, method=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, headers=headers if headers is not None else (H if body is not None else {}),
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if r.headers["Content-Type"].startswith("application/json") else raw), r.headers
        except urllib.error.HTTPError as e:
            raw = e.read()
            return e.code, (json.loads(raw) if raw[:1] == b"{" else raw), e.headers

    # ---- API behaviour ----
    def test_status_and_security_headers(self):
        s, d, h = self.call("/api/status")
        self.assertEqual(s, 200); self.assertTrue(d["ready"]); self.assertEqual(d["model_id"], "iforest_seed42")
        self.assertIn("default-src 'none'", h["Content-Security-Policy"])
        self.assertEqual(h["X-Content-Type-Options"], "nosniff")
        self.assertNotIn("Access-Control-Allow-Origin", h)

    def test_static_pages_served(self):
        s, body, h = self.call("/")
        self.assertEqual(s, 200); self.assertIn(b"/app.js", body)
        self.assertEqual(self.call("/app.js")[0], 200); self.assertEqual(self.call("/style.css")[0], 200)

    def test_bad_host_rejected(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port); c.request("GET", "/api/status", headers={"Host": "evil.example"})
        self.assertEqual(c.getresponse().status, 403)

    def test_post_requires_custom_header_and_json(self):
        self.assertEqual(self.call("/api/score", {"sample": "normal_day"}, {"Content-Type": "application/json"})[0], 403)
        self.assertEqual(self.call("/api/score", b"x", {"X-Requested-With": "logad", "Content-Type": "text/plain"})[0], 415)

    def test_oversize_upload_rejected_without_reading_it(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port)
        c.putrequest("POST", "/api/score"); c.putheader("X-Requested-With", "logad"); c.putheader("Content-Type", "application/json")
        c.putheader("Content-Length", str(50 * 1024 * 1024)); c.endheaders()
        self.assertEqual(c.getresponse().status, 413)

    def test_samples_behave(self):
        expect = {"normal_day": (set(), set()), "brute_force": ({"CRITICAL"}, set()),
                  "distributed_brute_force": ({"CRITICAL"}, {"R1"}), "compromise": ({"CRITICAL"}, {"R2"})}
        for name, (tiers, rules) in expect.items():
            s, d, _ = self.call("/api/score", {"sample": name})
            self.assertEqual(s, 200, d)
            got = {t for t, n in d["by_tier"].items() if n}
            if name == "normal_day":
                # one benign password typo is built in, so one low-grade alert is expected; never CRITICAL, no rules
                self.assertEqual(d["by_tier"]["CRITICAL"], 0)
                self.assertLessEqual(sum(n for t, n in d["by_tier"].items() if t != "NORMAL"), 2)
                self.assertEqual(sum(d["rules_hit"].values()), 0)
            else:
                self.assertTrue(tiers & got, (name, d["by_tier"]))
            self.assertTrue(rules <= {r for r, n in d["rules_hit"].items() if n}, (name, d["rules_hit"]))
            self.assertEqual(d["parse"]["malformed_lines"], 0)

    def test_mixed_sample_hits_every_rule_and_has_evidence(self):
        s, d, _ = self.call("/api/score", {"sample": "mixed_incident_day"})
        self.assertEqual({r for r, n in d["rules_hit"].items() if n}, {"R1", "R2", "R3"})
        top = d["rows"][0]
        self.assertTrue(top["evidence"]); self.assertEqual(top["tier"], "CRITICAL")
        s, w, _ = self.call(f"/api/window?run={d['run_id']}&entity={top['entity']}&start={top['window_start']}")
        self.assertEqual(s, 200); self.assertGreater(len(w["events"]), 0)

    def test_upload_text_roundtrip_and_hostile_content_is_data(self):
        evil = "<img/src=x/onerror=alert(1)>"
        text = "".join(f"Mar  3 10:00:{s:02d} h sshd[1]: Failed password for {evil} from 6.6.6.6 port 1 ssh2\n" for s in range(15))
        s, d, h = self.call("/api/score", {"text": text, "name": "../../etc/passwd<x>.log", "year": 2025})
        self.assertEqual(s, 200, d)
        self.assertEqual(d["label"], "upload:.._.._etc_passwd_x_.log")           # name sanitised; never used as a path
        self.assertTrue(h["Content-Type"].startswith("application/json"))
        self.assertEqual(d["rows"][0]["entity"], evil)
        self.assertEqual(list(self.state.upload_dir.glob("*")), [])              # temp upload removed

    def test_unparsable_log_gives_clear_422(self):
        s, d, _ = self.call("/api/score", {"text": "hello world\nnot a log\n"})
        self.assertEqual(s, 422); self.assertIn("hint", d)
        self.assertEqual(self.call("/api/score", {"text": "   "})[0], 400)
        self.assertEqual(self.call("/api/score", {"text": "x", "year": 1800})[0], 400)
        self.assertEqual(self.call("/api/score", {"sample": "nope"})[0], 404)

    def test_detections_filters_and_paging(self):
        s, d, _ = self.call("/api/detections?tier=FLAGGED&limit=5")
        self.assertEqual(s, 200); self.assertLessEqual(len(d["rows"]), 5)
        self.assertTrue(all(r["tier"] != "NORMAL" for r in d["rows"]))
        s, m, _ = self.call("/api/detections?missed=1&tier=ANY&limit=500")
        self.assertTrue(all(r["label"] == 1 and r["tier"] == "NORMAL" for r in m["rows"]))
        s, t, _ = self.call("/api/detections?tier=ANY&split=test&limit=500")
        self.assertTrue(all(r["split"] == "test" for r in t["rows"]))
        self.assertEqual(self.call("/api/detections?q=%25%27%3B--&tier=ANY")[1]["total"], 0)     # injection-looking text is just text
        self.assertEqual(self.call("/api/detections?limit=abc")[0], 400)

    def test_results_and_figures(self):
        s, r, _ = self.call("/api/results")
        self.assertEqual(s, 200); self.assertIn("rule", r["models"]["models"]); self.assertTrue(r["figures"])
        s, png, h = self.call(r["figures"][0]["url"])
        self.assertEqual((s, h["Content-Type"]), (200, "image/png")); self.assertTrue(png[:4] == b"\x89PNG")

    def test_figure_path_traversal_and_unknown_routes(self):
        for p in ("/figures/../../etc/passwd", "/figures/t/all/..%2f..%2fapp.db", "/figures/t/all/x.png", "/nope", "/api/nope"):
            self.assertEqual(self.call(p)[0], 404, p)

    def test_settings_maintenance_and_purge(self):
        self.assertIn("security", self.call("/api/settings")[1])
        self.call("/api/score", {"sample": "normal_day"})
        s, d, _ = self.call("/api/maintenance/purge", {})
        self.assertEqual(s, 200); self.assertGreaterEqual(d["deleted_runs"], 1)
        self.assertEqual(self.call("/api/maintenance")[1]["counts"]["events"], 0)

    def test_realdata_endpoints(self):
        s, d, _ = self.call("/api/realdata")
        self.assertEqual((s, d["datasets"]), (200, ["rt"]))
        s, r, _ = self.call("/api/realdata/rt")
        self.assertEqual(s, 200); self.assertIn("none (unlabeled)", r["labels"]); self.assertTrue(r["figure"])
        self.assertEqual(self.call(r["figure"])[0], 200)
        self.assertEqual(self.call("/api/realdata/nope")[0], 404)
        self.assertEqual(self.call("/api/realdata/..%2f..")[0], 404)

    def test_job_catalog_includes_new_jobs(self):
        from logad.app.jobs import DEFAULT_ALLOWLIST
        for n in ("analyze_real", "fetch_loghub", "fetch_blocklist", "robustness"):
            self.assertIn(n, DEFAULT_ALLOWLIST)

    # ---- jobs ----
    def test_job_runs_logs_and_history(self):
        s, d, _ = self.call("/api/jobs", {"name": "echo"})
        self.assertEqual(s, 202); self.jobs.wait(d["job_id"])
        r = self.call(f"/api/jobs/{d['job_id']}")[1]
        self.assertEqual((r["status"], r["lines"]), ("succeeded", ["hello", "world"]))
        self.assertTrue(any(h["job_id"] == d["job_id"] and h["status"] == "succeeded" for h in self.call("/api/jobs")[1]["history"]))

    def test_job_allowlist_busy_and_cancel(self):
        self.assertEqual(self.call("/api/jobs", {"name": "rm -rf /"})[0], 404)
        s, d, _ = self.call("/api/jobs", {"name": "slow"})
        self.assertEqual(s, 202)
        self.assertEqual(self.call("/api/jobs", {"name": "echo"})[0], 409)       # one job at a time
        time.sleep(0.5)
        self.assertEqual(self.call(f"/api/jobs/{d['job_id']}/cancel", {})[0], 200)
        self.jobs.wait(d["job_id"], 10)
        self.assertEqual(self.call(f"/api/jobs/{d['job_id']}")[1]["status"], "cancelled")
        self.assertEqual(self.call("/api/jobs/99999")[0], 404)

    # ---- the real UI code against the live server ----
    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_ui_walkthrough_in_node(self):
        r = subprocess.run(["node", str(HERE / "ui_shim.js"), self.base, str(APP_JS)], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        out = json.loads(r.stdout)
        self.assertEqual(out["steps"][:7], ["home", "results", "detections", "real", "ip", "settings", "pipeline"])
        self.assertTrue(out["flagged"] and out["detail"] and not out["hasError"])
        joined = "\n".join(out["texts"])
        self.assertIn("CRITICAL", joined); self.assertIn("Why this was flagged", joined)
        self.assertIn("Failed password for", joined)                              # raw log lines shown as text
        self.assertTrue(out["realOk"] and out["realNoPR"])                         # real-logs page renders, never shows P/R
        self.assertEqual((out["imgTags"], out["scriptTags"]), (0, 0))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_ui_hostile_log_stays_text(self):
        evil = "<img/src=x/onerror=alert(1)>"
        text = "".join(f"Mar  3 10:00:{s:02d} h sshd[1]: Failed password for {evil} from 6.6.6.6 port 1 ssh2\n" for s in range(15))
        env = {"LOG_TEXT": text, "PATH": __import__("os").environ["PATH"]}
        r = subprocess.run(["node", str(HERE / "ui_shim.js"), self.base, str(APP_JS), "#score"], capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        out = json.loads(r.stdout)
        self.assertIn(evil, out["texts"])
        self.assertEqual((out["imgTags"], out["scriptTags"]), (0, 0))


class JobRunnerUnit(unittest.TestCase):
    def test_unknown_job_raises(self):
        with tempfile.TemporaryDirectory() as t:
            jr = JobRunner(Path(t) / "j.db", {"a": ([sys.executable, "-c", "pass"], "d")}, cwd=t)
            with self.assertRaises(JobError):
                jr.start("b")


if __name__ == "__main__":
    unittest.main()
