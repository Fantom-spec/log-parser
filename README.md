# logad: SSH login anomaly detection

Phase 1 turns raw OpenSSH logs into leak-free `(entity, 5-minute window)` feature vectors.
Phase 2 trains and compares four detectors on them. Phase 3 runs the feature-family ablation and the H1 experiment. Phase 4 adds risk tiers, rules, replay scoring, SQLite storage and a static dashboard.

## Run the app (showcase)
```bash
pip install -r requirements.txt
python run_app.py --open          # http://127.0.0.1:8000 ; stdlib server, no other packages
```
The repo ships with the trained synthetic demo, so it works immediately. If `data/processed` is missing, open the **Pipeline** tab and run `full_demo` (about 15 seconds).

**Three-minute demo:** Home (what it is, tier legend) -> **Score logs** -> click *mixed incident day* (a day of normal logins plus a 90-attempt brute force, a 28-IP distributed attack,
a failures-then-success takeover and a blocklisted IP) -> click a flagged row to see the evidence and the raw log lines -> **Results** (honest held-out numbers, ablation, H1, caveats)
-> **Detections** (filter by tier, tick *missed attacks*) -> **Pipeline** (re-run any stage with live logs).
You can also paste or upload your own OpenSSH `auth.log`; set the year if the log is not from the training year.

Pages: Home, Score logs, Results, Detections, Pipeline (allowlisted jobs, one at a time, live log, cancel), Settings/Maintenance.
Security: binds 127.0.0.1 only; Host header checked; no CORS headers and POSTs need a custom header; uploads are content only (no client file paths, 5 MB limit);
strict CSP; all log text is rendered with `textContent`. It is a single-user local tool with no login, not a production service.

## Quick start (command line)
```bash
python scripts/generate_synthetic_logs.py        # data/synthetic/{auth.log,labels.csv} + data/threat_intel/*
python scripts/prepare_data.py                   # data/processed/synthetic/*
python scripts/train.py                          # fit on train rows, calibrate thresholds on validation rows
python scripts/evaluate.py                       # held-out test metrics + figures (or: scripts/run_pipeline.py for all steps)
python scripts/run_ablation.py                   # drop-one / only-one feature family, paired deltas
python scripts/run_h1.py                         # per-user vs global baseline
python scripts/calibrate_scoring.py              # risk tiers (validation) + model-only vs model+rules
python scripts/build_dashboard.py                # ingest to SQLite, write reports/dashboard/index.html
python scripts/score_log.py data/synthetic/auth.log   # score any raw SSH log with the trained model
python scripts/run_robustness.py                 # scoring latency + detection delay
python scripts/analyze_real.py --log data/raw/OpenSSH.log --year <year> --dataset loghub   # real unlabeled log
python -m unittest discover -s tests             # 117 tests (includes the app, a live server and the UI code run in node) (pytest also works if installed)
```
Real logs (e.g. Loghub OpenSSH, unlabeled, attack-heavy):
```bash
python scripts/prepare_data.py --dataset loghub --log data/raw/SSH.log --group-by src_ip --year <log year> --no-threat-intel
```
Use `--no-threat-intel` until a real blocklist snapshot plus manifest entry is committed.

## Outputs (`data/processed/<dataset>/`)
| File | Purpose |
|---|---|
| `window_features.csv` | one row per window: entity, window_start, split, label, attack_scenario, history_windows, low_history, then features |
| `feature_names.json` | source of truth for feature order, families and dropped features (with reasons) |
| `baselines.json` | training-only baselines (per entity + global) for later scoring |
| `parse_report.json` | line-level accounting: malformed, non-sshd, unsupported patterns, ignored duplicates, coverage |
| `meta.json` | inputs + sha256, config snapshot, split cutoffs, per-split counts, warnings |

Reruns on identical inputs and config give byte-identical files (no wall-clock values are written).

## Design decisions
* **Counting rules.** Only `Failed password` / `keyboard-interactive` is a `LOGIN_FAILED`. `Failed none`, `Failed publickey`,
  `Invalid user`, PAM failures and `maximum authentication attempts` are separate event types and never add to `failed_count`.
  Pure duplicates (`input_userauth_request`, `check pass; user unknown`) are ignored and tallied in the report.
  `message repeated N times` becomes one event with `repeat=N`.
* **Year.** Syslog lines have no year: it comes from `dataset.log_year`; Dec to Jan rollover is detected, and a late December
  line after rollover keeps the old year. Assumes roughly time-ordered files.
* **Windows.** Tumbling 5 minutes, only where an attributed event exists. Events with no entity (e.g. a preauth disconnect with
  no user) are counted in `meta.json` (`unattributed_events`), not hidden.
* **Split.** Chronological 60/20/20 on window start, done before anything is learned. Windows sharing a start time never straddle a boundary.
* **Baselines.** Median/MAD, fitted on training windows only (normal-only when labels exist), then frozen. Entities with fewer than
  `min_history_windows` training windows fall back to the global baseline and are flagged `low_history`.
  `new_ip_count` on train rows uses an expanding history (strictly earlier train windows); val/test use the frozen baseline.
  `event_rate_z` and `hour_deviation` are in-sample on train rows.
* **Labels.** Window label = 1 if any event is an attack, 0 only if every event is labeled normal, otherwise empty (unknown).
  Unlabeled data is never "normal".
* **Features.** 19 emitted for `group_by=user`. Constant-by-construction features are dropped, not emitted:
  GeoIP (deferred), `distinct_users` (user mode), `distinct_src_ips` and `max_events_per_ip` (src_ip mode),
  threat_intel when disabled or no snapshot. Features that are merely constant on the training split are kept and listed in `meta.json` warnings.
* **Threat intel.** Static snapshot read offline, sha256-checked against `manifest.json`; a mismatch is an error.
  On synthetic data the blocklist is planted from attacker IPs, so those features are circular by construction.
* **Schema additions.** `line_no` (label joins, traceability) and `repeat` are added to the 10-field event schema.

## Configuration
`config/config.yaml` holds pipeline settings; `config/thresholds.yaml` holds tier targets, rules, rule floors, evidence size and unlabeled-log quantiles. `load_config()` merges them over built-in defaults (config.yaml, then thresholds.yaml, then overrides). Calibrated numeric score cut-points are written to `data/processed/<dataset>/models/<tag>/{thresholds,scoring}.json`.

## Architecture document
`docs/architecture.md` is the design spec plus an appendix listing what was built, what differs (e.g. static dashboard and local app instead of Streamlit) and what is open.

## Layout
`src/logad/{parsing,features,threatintel,services}` hold Phase 1 code; the other package folders are placeholders for Phases 2 to 5.
`services/` is the shared layer the scripts and tests call (the API will use it later).

## Phase 2: models and evaluation
Two stages, so the test split is touched exactly once:
1. `scripts/train.py` (`services/train.py`) reads **train and validation rows only**. Preprocessing and models are fitted on training
   windows (normal-only when labels exist); each model's threshold is calibrated on validation (max F1, ties -> highest threshold).
   Writes `data/processed/<dataset>/models/<tag>/` with joblib models, `registry.json` (features, params, threshold, version, validation metrics),
   `thresholds.json`, `fit_times.json`.
2. `scripts/evaluate.py` (`services/evaluate.py`) scores the test split with the frozen thresholds and writes
   `reports/metrics/<dataset>/<tag>/{metrics.json,comparison.csv,seeds.csv,latency.json}` and four figures in `reports/figures/<dataset>/<tag>/`.

| Model | Input | Notes |
|---|---|---|
| Rule baseline | raw `failed_count` | threshold tuned on validation |
| Isolation Forest (primary) | raw features | 200 trees, max_samples 256, 10 seeds; seed 42 is the reported primary |
| One-Class SVM | log1p + standardised | RBF, nu 0.05, gamma scale |
| Local Outlier Factor | log1p + standardised | novelty mode, 20 neighbours |

Reported per model: precision, recall, F1, false-positive rate, alerts/day, PR-AUC, ROC-AUC, confusion counts, per-scenario recall, bootstrap CIs at the
fixed threshold, and Isolation Forest mean +/- std over seeds. Latency is wall-clock and kept in `latency.json`, outside the reproducible metrics.
Models are stored with joblib only. Pass `features=[...]` and a `tag` to `train_models` for ablations (the rule baseline is skipped if `failed_count` is removed).

Known limits: one chronological split; bootstrap CIs resample windows independently (optimistic); unlabeled datasets (Loghub) are refused by
training/evaluation because there is nothing to calibrate or score against; they get the score-distribution analysis later.

## Phase 3: ablation and H1
Both experiments refit every model from scratch per variant (train rows only), recalibrate each threshold on validation, and use the test split once per variant.
All comparisons are paired: same test windows, same seeds, shared bootstrap resamples.

* **Ablation** (`scripts/run_ablation.py` -> `reports/metrics/<dataset>/ablation/{ablation.json,ablation.csv}`, `reports/figures/<dataset>/ablation/ablation.png`).
  Variants: `all`, `drop_<family>` and `only_<family>` for failed_logins, event_frequency, ip_activity, time_patterns and threat_intel.
  The rule baseline is skipped when `failed_count` is removed. Each delta (variant minus all) carries a bootstrap CI; for Isolation Forest the point
  estimate is the seed-paired mean +/- std over 10 seeds while the CI comes from a window bootstrap on the primary seed, so the two kinds of uncertainty stay separate.
  Verdicts are `dropping_helps` / `dropping_hurts` only when the CI excludes 0, otherwise `inconclusive`.
* **H1** (`scripts/run_h1.py` -> `reports/metrics/<dataset>/h1/h1.json`, figure `h1.png`): per-user baselines improve detection over one global baseline.
  The log is re-prepared twice (`<dataset>/h1/{per_user,global}`) with identical windows and split; the global arm sets `min_history_windows` so high that
  every entity falls back to the global baseline. The run aborts if the arms are not row-aligned. Decision rule fixed in advance: supported only if the
  paired PR-AUC delta CI is above 0. Results are also split by `low_history` and by attack scenario.

Limits: bootstrap CIs resample windows independently (optimistic); few test attacks make small deltas noise; on synthetic data the threat_intel family is circular and
users are generated with distinct IPs and hours, which favours per-user baselines. The H1 arms differ only in the baseline-dependent features that actually change
(on the current synthetic data that is `hour_deviation` alone, see `features_that_differ` in `h1.json`).

## Phase 4: scoring, storage, dashboard
* **Rules** (`scoring/rules.py`, thresholds in `config/thresholds.yaml`, never tuned on test): R1 distributed brute force (>=10 failures from >=5 IPs), R2 success after repeated failures,
  R3 blocklisted IP with failures. A rule whose features are missing (R1 when grouping by src_ip, R3 without a blocklist) is reported unavailable.
* **Risk tiers** (`scoring/tiers.py`): NORMAL / LOW / MEDIUM / HIGH / CRITICAL on the model score. LOW is the Phase 2 max-F1 threshold; MEDIUM/HIGH/CRITICAL are the lowest
  validation thresholds reaching precision 0.75 / 0.90 / 0.97 (with at least 3 validation attacks flagged). Unreachable tiers get an infinite threshold; tiers that share a cut-point are
  flagged `merged_with` (the higher tier wins, the lower one is empty by score). A rule hit raises the tier to at least its floor (R1 HIGH, R2 CRITICAL, R3 MEDIUM); rules never lower a tier.
* **Model only vs model + rules** are evaluated separately on validation (labelled as calibration data) and on test, with rules-only alongside
  (`reports/metrics/<dataset>/<tag>/scoring.json`, `scoring.png`).
* **Replay scoring** (`scoring/scorer.py`, `scripts/score_log.py`): raw log -> same parser, windows, frozen baselines, blocklist snapshot (hash-checked) and preprocessing as training.
  A test checks that replayed features equal the training table on validation/test windows (train rows differ only in `new_ip_count`, which uses an expanding history there).
  The registry feature list must equal `feature_names.json` or scoring refuses to run.
* **Evidence** (`scoring/explain.py`): for flagged windows, triggered rules plus the features that deviate most from normal training windows (robust z >= 3). This is feature-deviation evidence, not a causal attribution.
* **SQLite** (`storage/db.py`, `data/logad.db`): datasets, window_features (columns follow the feature list; names validated and quoted), model_registry, runs, detections, events.
  All values are bound parameters; re-ingesting replaces the previous replay run.
* **Dashboard** (`dashboard.py`, `reports/dashboard/index.html`): one self-contained file (no server, no external requests, restrictive CSP). Data is embedded as escaped JSON and rendered with
  `textContent` only, so log text cannot become markup. It shows models, tiers, model-only vs model+rules, ablation and H1 verdicts, figures and a filterable detections explorer with evidence.

Limits: the static dashboard needs a rebuild after new runs and was exercised with a DOM shim in node, not in a real browser. Tier cut-points rest on few validation attacks and are noisy.
On the current synthetic data every rule hit is already above the model's LOW threshold, so rules change tiers, not the alert set.

## Local app (`python run_app.py`)
Stdlib HTTP server on **127.0.0.1 only** (FastAPI could not be installed in the build sandbox, so there is no third-party dependency), JSON API plus a plain HTML/CSS/JS UI with no build step.
Pages: **Home** (status, tiers), **Score logs** (paste/upload an SSH log or load a demo log; flagged windows with evidence and the raw log lines), **Results** (models, tiers, model vs model+rules, ablation, H1, figures),
**Detections** (filter/page all windows, "missed attacks only"), **Pipeline** (allowlisted jobs, one at a time, live log, cancel, history in SQLite), **Settings/maintenance** (config view, DB counts, purge uploads).
Security: loopback bind, Host-header check (DNS-rebinding guard), no CORS headers and POSTs need a custom header, 5 MB upload limit, uploads are content only (server-chosen temp file, sanitised label, no client paths),
bound SQL parameters, log text only ever rendered with `textContent`, strict CSP, models loaded with joblib from local artefacts. Tests drive the real front-end code against a live server in node, including hostile log text.
Demo logs (`normal_day`, `brute_force`, `distributed_brute_force`, `compromise`, `mixed_incident_day`) are built from the trained users' real IPs and hours.

## Phase 5: real logs, fetchers, robustness
* **Real, unlabeled logs** (`scripts/analyze_real.py`, `services/realdata.py`, `services/train_unlabeled.py`): prepare (leak-free chronological split) -> Isolation Forest on training windows ->
  tiers that are **score quantiles on the validation split** (LOW 95%, MEDIUM 98%, HIGH 99%, CRITICAL 99.5%, see `unlabeled:` in config). Quantile tiers mean "unusual for this log" and flag a fixed share of windows by construction.
  The report contains **no precision/recall/F1** (a test asserts this): score distribution, windows per tier per split, top flagged windows with evidence, rule hits, parse coverage (unsupported patterns listed),
  blocklist overlap, a clearly labelled **weak proxy** (window with >= 5 failed logins; circular because failed_count is a model input, so agreement is not accuracy) and a **cross-domain check** that scores the log with the synthetic-trained model
  and lists the most shifted features (a distribution-shift demonstration, not an evaluation). Results: `reports/metrics/<dataset>/realdata.json`, figure `realdata/realdata.png`, app page **Real logs**.
  The shipped `synthetic_unlabeled_demo` is the synthetic log with its labels ignored, only there to show the page; it is not real data.
* **Fetchers** (`scripts/fetch_loghub.py`, `scripts/fetch_blocklist.py`, `services/fetch.py`): the only code that downloads anything, run explicitly. http/https/file only, size cap, atomic write, and a manifest entry (url, sha256, date) so a snapshot is reproducible
  and tampering is detected. The Loghub OpenSSH_2k file was obtained via `git clone` of `logpai/loghub` and loaded with `fetch_loghub.py --url file://...` (the raw.githubusercontent.com URL was blocked in the build sandbox, so the default URLs remain unverified; the ipsum URL was not reachable at all). Tests use file:// URLs.
  `real.year` in config is an assumption (syslog has no year): set it from the dataset documentation.
* **Parser hardening**: optional `parser.dedupe_exact_lines` (off by default because two real identical events in one second would merge), out-of-order lines counted (`timestamp_regressions`) and re-sorted by time,
  and an invariant under fuzzing (random bytes, 200 KB lines, bad dates, huge repeat counts): every input line is accounted for in the parse report. New or unseen entities are scored with the global baseline and flagged `low_history`.
  Replaying a log twice gives identical scores; small out-of-order jitter gives identical features (except `consecutive_failed_max`, which depends on event order by definition).
* **Latency and detection delay** (`scripts/run_robustness.py`, `reports/metrics/<dataset>/<tag>/robustness.json`): on the synthetic log, about 17,000 events/s end to end (0.34 ms per window, single process, whole log in memory);
  59 of 60 labelled incidents had a flagged window, with delay (first attack event to the close of the first flagged window) median 146 s, p95 291 s, max 297 s, all within one 5-minute window. Late events would add a grace period that is not implemented.
  These cover all splits, so they show timing, not accuracy. "Real-time" is not claimed: scoring is per closed window over files.

**Real run done:** Loghub `OpenSSH_2k.log` (2,000 lines, ~4 hours, sha256 `1e491272...4264f`, year 2015 assumed) was analysed grouped by source IP: 99.9% sshd coverage, 52 windows (31/11/10 train/val/test, all low-history), 10 flagged CRITICAL by quantile tier (top ones are 129-141 failures in 5 minutes from one IP). This is a pipeline demonstration on real syslog with no labels, not an accuracy result; the ipsum blocklist could not be downloaded, so threat-intel features were off. Results: `reports/metrics/loghub/realdata.json`.

Limits: the full Loghub OpenSSH log has not been run, and there is no labelled public benchmark; there is no streaming/tail mode and no late-event grace period;
the app is local single-user (no authentication); weak-proxy and cross-domain numbers are sanity checks only.

## Not yet built
Live file tailing / streaming with late-event grace, GeoIP, Drain3 template mining, other log sources (Windows/AD), authentication for the app, and a verified run on the real Loghub file.
# log-parser
