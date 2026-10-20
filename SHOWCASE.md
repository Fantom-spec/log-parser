# Showcase guide (about 5 minutes)

Start: `python run_app.py --open` (http://127.0.0.1:8000). Everything runs locally; nothing is sent anywhere.

1. **Home** – one-line pitch: *unsupervised detector that learns normal SSH login behaviour per 5-minute window, flags deviations, explains why.*
   Point at the risk-tier table: LOW is the best-F1 point on validation; MEDIUM/HIGH/CRITICAL are validation-precision bands.
2. **Score logs → "mixed incident day"** (one click). In seconds you get ~23 windows: 4 CRITICAL, 2 LOW, the rest NORMAL.
   Click the top rows: each shows the evidence (rule text plus the features that deviate most) and the actual log lines of that window.
   You will see a 90-attempt brute force, a 28-IP distributed attack (rule R1), a success after repeated failures at 03:07 (rule R2), and a blocklisted IP (rule R3).
   Then try **normal day**: only one low-grade alert from a user who mistyped a password once. Say so; it is a realistic false positive.
3. **Score logs → paste your own lines** (any OpenSSH `auth.log`); set the year if the log is not from the trained year.
4. **Results** – the honest numbers on the held-out test split (23 attack windows): Isolation Forest recall 0.96 but precision 0.47; One-Class SVM is the best balanced model (F1 0.76);
   the rule baseline is precise (1.00) but blind to off-hours and new-IP logins. Show ablation and H1.
5. **Detections** – tick *missed attacks only* to show what was not caught.
6. **Real logs** – shows how an *unlabeled* log is handled: no accuracy numbers, quantile tiers, top windows with evidence, a labelled weak-proxy check and a cross-domain (distribution-shift) check. Dataset `loghub` is the real OpenSSH_2k run; `synthetic_unlabeled_demo` is only a stand-in.
7. **Pipeline** – run a job (e.g. `calibrate`) and watch the live log; `full_demo` rebuilds everything from scratch in ~15 s.

## Say these out loud (they are in the app too)
- Data is **synthetic** with generator-defined attacks; numbers show the pipeline works, not real-world accuracy.
- The blocklist is planted from the same attacker IPs, so threat-intel features are circular on this data (dropping them changes PR-AUC by about 0.01 or less).
- H1 (per-user vs global baselines) came out **inconclusive**; on this data only one feature differs between the two arms.
- Few test attacks (23), so confidence intervals are wide and optimistic (windows resampled independently).
- Tier cut-points rest on 7–16 validation attacks and are noisy; MEDIUM was only 2/10 attacks on test.
- Real Loghub **OpenSSH_2k** was run (2,000 lines, about 4 hours, 52 windows by source IP): it is unlabeled, so the page shows distributions and sanity checks, never accuracy. `real.year` (2015) is an assumption, the full log and a blocklist snapshot were not run, and the default fetch URLs are unverified.
- Timing: ~17k events/s on one core; incidents are flagged within one 5-minute window (median 146 s after the first attack event); no streaming mode or late-event grace yet.
