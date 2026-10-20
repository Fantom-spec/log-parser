"""Stdlib HTTP server for the local app (127.0.0.1 only). JSON API + static UI, no dependencies."""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from logad.app import service
from logad.app.jobs import JobError, JobRunner
from logad.app.service import ApiError, AppState

STATIC = Path(__file__).parent / "static"
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8")}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
_FIG = re.compile(r"^/figures/([A-Za-z0-9_.:-]+)/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+\.png)$")


class Handler(BaseHTTPRequestHandler):
    state: AppState
    jobs: JobRunner
    verbose = False
    server_version = "logad"

    def log_message(self, fmt, *args):
        if self.verbose:
            super().log_message(fmt, *args)

    # ---- plumbing ----
    def _headers(self, status, ctype, length, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

    def _json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode()
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _err(self, status, msg, extra=None):
        self._json({"error": msg, **(extra or {})}, status)

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _dispatch(self, fn):
        if not self._host_ok():
            return self._err(403, "bad Host header")
        try:
            fn()
        except ApiError as e:
            self._err(e.status, str(e), e.extra)
        except JobError as e:
            self._err(e.status, str(e))
        except (ValueError, KeyError) as e:
            self._err(400, f"bad request: {e}")
        except BrokenPipeError:
            pass
        except Exception as e:                                          # noqa: BLE001 - never leak a traceback
            self.log_error("unhandled: %r", e)
            self._err(500, "internal error")

    # ---- GET ----
    def do_GET(self):
        self._dispatch(self._get)

    def _get(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p, st = u.path, self.state
        if p in STATIC_FILES:
            name, ctype = STATIC_FILES[p]
            body = (STATIC / name).read_bytes()
            return self._headers(200, ctype, len(body)) or self.wfile.write(body)
        m = _FIG.match(p)
        if m:
            root = (st.reports_root / "figures").resolve()
            f = (root / m.group(1) / m.group(2) / m.group(3)).resolve()
            if root not in f.parents or not f.is_file():
                return self._err(404, "not found")
            body = f.read_bytes()
            return self._headers(200, "image/png", len(body)) or self.wfile.write(body)
        if p == "/api/status":
            return self._json(service.status(st, self.jobs.busy()))
        if p == "/api/results":
            return self._json(service.results(st))
        if p == "/api/samples":
            return self._json(service.samples(st))
        m = re.fullmatch(r"/api/samples/([a-z_]+)", p)
        if m:
            return self._json({"name": m.group(1), "text": service.sample_text(st, m.group(1))})
        if p == "/api/detections":
            return self._json(service.detections(st, q))
        if p == "/api/window":
            return self._json(service.window_events(st, int(q.get("run", "")), q.get("entity", ""), q.get("start", "")))
        if p == "/api/realdata":
            return self._json(service.realdata_list(st))
        m = re.fullmatch(r"/api/realdata/([A-Za-z0-9_-]+)", p)
        if m:
            return self._json(service.realdata_get(st, m.group(1)))
        if p == "/api/maintenance":
            return self._json(service.maintenance(st))
        if p == "/api/settings":
            return self._json(service.settings(st))
        if p == "/api/jobs":
            return self._json({"catalog": self.jobs.catalog(), "history": self.jobs.history(), "busy": self.jobs.busy()})
        m = re.fullmatch(r"/api/jobs/(\d+)", p)
        if m:
            return self._json(self.jobs.get(int(m.group(1)), int(q.get("since", "0"))))
        self._err(404, "not found")

    # ---- POST ----
    def do_POST(self):
        self._dispatch(self._post)

    def _body(self) -> dict:
        if self.headers.get("X-Requested-With") != "logad":
            raise ApiError("missing X-Requested-With header", 403)
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise ApiError("Content-Type must be application/json", 415)
        try:
            n = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            n = -1
        if n < 0:
            raise ApiError("Content-Length required", 411)
        if n > service.MAX_UPLOAD_BYTES + 4096:
            raise ApiError("request too large (limit 5 MB)", 413)
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise ApiError("JSON object expected")
        return data

    def _post(self):
        p, st = urlparse(self.path).path, self.state
        body = self._body()
        if p == "/api/score":
            year = body.get("year")
            year = int(year) if year not in (None, "") else None
            if body.get("sample"):
                text, name = service.sample_text(st, str(body["sample"])), f"sample_{body['sample']}.log"
                label = None
                out = service.score_text(st, text, name, year)
                out["label"] = "sample:" + str(body["sample"])
                return self._json(out)
            return self._json(service.score_text(st, str(body.get("text", "")), str(body.get("name") or "pasted.log"), year))
        if p == "/api/jobs":
            return self._json(self.jobs.start(str(body.get("name", ""))), 202)
        m = re.fullmatch(r"/api/jobs/(\d+)/cancel", p)
        if m:
            return self._json(self.jobs.cancel(int(m.group(1))))
        if p == "/api/maintenance/purge":
            return self._json(service.purge_uploads(st))
        self._err(404, "not found")


def make_server(state: AppState, jobs: JobRunner | None = None, port: int = 8000, verbose=False) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state, "verbose": verbose,
                                                "jobs": jobs or JobRunner(state.db_path)})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
