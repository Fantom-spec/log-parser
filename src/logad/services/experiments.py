"""Service: ablation by feature family and the H1 (per-user vs global baseline) experiment.

Every variant is fitted and threshold-calibrated from scratch on train / validation rows only; the
test split is used once per variant for metrics. Nothing is persisted except reports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from logad.config import PROJECT_ROOT, merge_config
from logad.evaluation import plots
from logad.evaluation.ablation import make_variants, paired_bootstrap, resample_indices, verdict
from logad.evaluation.metrics import aggregate, calibrate_threshold, per_scenario_recall, threshold_metrics
from logad.io import dump_json, load_json
from logad.preprocessing.pipeline import Preprocessor, load_dataset, training_mask
from logad.services.prepare import prepare_dataset
from logad.services.train import _span_days, build_models

ML = ("iforest", "ocsvm", "lof")


def _fit_variant(df, feats, cfg) -> dict:
    """Fit all models on train rows, calibrate on validation rows, score test rows."""
    fit_df = df[training_mask(df)]
    val_df = df[(df["split"] == "val") & df["label"].notna()]
    test_df = df[(df["split"] == "test") & df["label"].notna()].reset_index(drop=True)
    pre = Preprocessor(feats, cfg["preprocessing"]["log1p_features"]).fit(fit_df)
    mats = {k: {"raw": pre.transform_raw(d), "scaled": pre.transform_scaled(d)}
            for k, d in (("train", fit_df), ("val", val_df), ("test", test_df))}
    y_val = val_df["label"].astype(int).to_numpy()
    models = {}
    for mid, model in build_models(cfg, feats):
        model.fit(mats["train"][model.input_kind])
        s_val = model.score(mats["val"][model.input_kind])
        cal = calibrate_threshold(y_val, s_val)
        models[mid] = {"family": model.family, "seed": getattr(model, "seed", None),
                       "threshold": cal["threshold"], "test_scores": model.score(mats["test"][model.input_kind])}
    return {"models": models, "test": test_df}


def _summarize(fit, y, span, primary_seed) -> dict:
    """Per family: metrics (IF: primary seed + mean/std over seeds + per-seed list)."""
    out = {}
    per = {mid: threshold_metrics(y, d["test_scores"], d["threshold"], span) for mid, d in fit["models"].items()}
    for fam in ("rule", "ocsvm", "lof"):
        if fam in per:
            out[fam] = {"model_id": fam, "metrics": per[fam]}
    ids = sorted((m for m in per if m.startswith("iforest")), key=lambda m: fit["models"][m]["seed"])
    prim = f"iforest_seed{primary_seed}" if f"iforest_seed{primary_seed}" in per else ids[0]
    out["iforest"] = {"model_id": prim, "metrics": per[prim], "seed_stats": aggregate([per[m] for m in ids]),
                      "per_seed": {m: per[m] for m in ids}}
    return out


def _delta(fit_v, fit_r, summ_v, summ_r, y, idx_list, fam, level, labels=("supported", "reversed")) -> dict | None:
    """Variant minus reference for one family, with seed-paired and bootstrap uncertainty."""
    if fam not in summ_v or fam not in summ_r:
        return None
    mv, mr = summ_v[fam]["model_id"], summ_r[fam]["model_id"]
    fv, fr = fit_v["models"][mv], fit_r["models"][mr]
    d = paired_bootstrap(y, fv["test_scores"], fv["threshold"], fr["test_scores"], fr["threshold"], idx_list, level)
    if fam == "iforest":
        ids = [m for m in summ_v[fam]["per_seed"] if m in summ_r[fam]["per_seed"]]
        dap = np.array([summ_v[fam]["per_seed"][m]["pr_auc"] - summ_r[fam]["per_seed"][m]["pr_auc"] for m in ids])
        df1 = np.array([summ_v[fam]["per_seed"][m]["f1"] - summ_r[fam]["per_seed"][m]["f1"] for m in ids])
        d["seed_paired"] = {"n_seeds": len(ids), "delta_pr_auc_mean": float(dap.mean()),
                            "delta_pr_auc_std": float(dap.std(ddof=1)) if len(ids) > 1 else 0.0,
                            "delta_f1_mean": float(df1.mean()),
                            "delta_f1_std": float(df1.std(ddof=1)) if len(ids) > 1 else 0.0}
    d["verdict_pr_auc"] = verdict(d["delta_pr_auc_ci"], *labels)
    return d


def _require_labels(meta):
    if not meta["has_labels"]:
        raise ValueError("experiments need a labelled dataset (threshold calibration and metrics)")


def run_ablation(dataset_dir, cfg: dict, reports_root=None) -> dict:
    dataset_dir = Path(dataset_dir)
    df, names, meta = load_dataset(dataset_dir)
    _require_labels(meta)
    families = load_json(dataset_dir / "feature_names.json")["families"]
    variants = make_variants(families, names)
    seeds = cfg["models"]["seeds"]
    ev = cfg["evaluation"]
    fits = {v["name"]: _fit_variant(df, v["features"], cfg) for v in variants}
    test = fits["all"]["test"]
    y = test["label"].astype(int).to_numpy()
    span = _span_days(test, meta["window_stats"]["window_seconds"])
    idx_list = resample_indices(y, ev["bootstrap_samples"], ev["bootstrap_seed"])
    summ = {n: _summarize(f, y, span, seeds[0]) for n, f in fits.items()}

    results = []
    for v in variants:
        row = {"variant": v["name"], "mode": v["mode"], "family": v["family"],
               "n_features": len(v["features"]), "features": v["features"], "models": {}, "delta_vs_all": {}}
        for fam, s in summ[v["name"]].items():
            m = s["metrics"]
            row["models"][fam] = {k: m[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc",
                                                    "false_positive_rate", "alerts_per_day", "threshold")}
            if fam == "iforest":
                row["models"][fam]["seed_stats"] = s["seed_stats"]
        if v["name"] != "all":
            for fam in ("rule",) + ML:
                d = _delta(fits[v["name"]], fits["all"], summ[v["name"]], summ["all"], y, idx_list, fam, ev["ci_level"],
                           ("dropping_helps", "dropping_hurts") if v["mode"] == "drop" else ("only_better", "only_worse"))
                if d:
                    row["delta_vs_all"][fam] = d
        results.append(row)

    root = Path(reports_root) if reports_root else PROJECT_ROOT / "reports"
    mdir, fdir = root / "metrics" / meta["dataset"] / "ablation", root / "figures" / meta["dataset"] / "ablation"
    caveats = [
        "Each variant is refitted and its threshold recalibrated on validation; test is used once per variant.",
        "Deltas: IsolationForest uses seed-paired differences (mean +/- std over seeds) for the point estimate, and a "
        "window bootstrap on the primary seed for the interval; the two sources of uncertainty are separate.",
        "Bootstrap resamples windows independently although incidents span several windows: intervals are optimistic.",
        "Leave-one-family-out can understate a family whose signal is duplicated elsewhere; see the only_* variants.",
        f"The test split has only {int(y.sum())} attack windows; small deltas are noise unless the CI excludes 0.",
    ]
    if meta["dataset"] == "synthetic":
        caveats.append("Synthetic data: attacker IPs are partly planted in the blocklist, so the threat_intel family is circular.")
    dump_json({"dataset": meta["dataset"], "n_test_windows": int(len(y)), "n_test_attacks": int(y.sum()),
               "seeds": seeds, "families": families, "variants": results, "caveats": caveats}, mdir / "ablation.json")
    _write_ablation_csv(results, mdir / "ablation.csv")
    plots.plot_ablation(results, fdir / "ablation.png")
    return {"metrics_dir": str(mdir), "figures_dir": str(fdir), "results": results}


def _write_ablation_csv(results, path):
    import pandas as pd
    rows = []
    for r in results:
        for fam, m in r["models"].items():
            d = r["delta_vs_all"].get(fam, {})
            ss = m.get("seed_stats", {})
            rows.append({
                "variant": r["variant"], "mode": r["mode"], "family": r["family"], "n_features": r["n_features"],
                "model": fam, "pr_auc": m["pr_auc"], "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                "false_positive_rate": m["false_positive_rate"],
                "pr_auc_seed_mean": ss.get("pr_auc", {}).get("mean"), "pr_auc_seed_std": ss.get("pr_auc", {}).get("std"),
                "delta_pr_auc": d.get("delta_pr_auc"),
                "delta_pr_auc_ci_low": (d.get("delta_pr_auc_ci") or [None, None])[0],
                "delta_pr_auc_ci_high": (d.get("delta_pr_auc_ci") or [None, None])[1],
                "delta_pr_auc_seed_mean": d.get("seed_paired", {}).get("delta_pr_auc_mean"),
                "delta_pr_auc_seed_std": d.get("seed_paired", {}).get("delta_pr_auc_std"),
                "delta_f1": d.get("delta_f1"), "verdict_pr_auc": d.get("verdict_pr_auc")})
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")


# ---------------------------------------------------------------- H1 -------------------------
GLOBAL_MIN_HISTORY = 10 ** 9      # nobody reaches it, so every window uses the global baseline


def run_h1(dataset_dir, cfg: dict, reports_root=None, work_dir=None) -> dict:
    """H1: per-user baselines improve detection over a single global baseline.

    Both datasets are rebuilt from the same raw log with identical windows, split and features;
    only the baseline-dependent features (event_rate_z, hour_deviation, new_ip_count) change.
    Decision rule (fixed before looking at results): H1 'supported' for a model if the paired
    delta in test PR-AUC (per_user - global) has a bootstrap CI above 0.
    """
    dataset_dir = Path(dataset_dir)
    df0, names, meta = load_dataset(dataset_dir)
    _require_labels(meta)
    mc = meta["config"]
    base = merge_config(cfg, {k: mc[k] for k in ("dataset", "windowing", "parser", "features", "split",
                                                  "threat_intel") if k in mc})
    work = Path(work_dir) if work_dir else dataset_dir / "h1"
    log, labels = meta["inputs"]["log"], meta["inputs"]["labels"]
    cfgs = {"per_user": base,
            "global": merge_config(base, {"features": {"min_history_windows": GLOBAL_MIN_HISTORY}})}
    data = {}
    for mode, c in cfgs.items():
        prepare_dataset(log, work / mode, c, labels, f"{meta['dataset']}_h1_{mode}")
        data[mode] = load_dataset(work / mode)[0]

    keys = ["entity", "window_start", "split", "label"]
    a, b = (data[m][keys].reset_index(drop=True) for m in ("per_user", "global"))
    if not a.equals(b):
        raise RuntimeError("H1 datasets are not row-aligned; the paired comparison would be invalid")
    seeds, ev = cfg["models"]["seeds"], cfg["evaluation"]
    fits = {m: _fit_variant(data[m], names, cfg) for m in data}
    test = fits["per_user"]["test"]
    y = test["label"].astype(int).to_numpy()
    span = _span_days(test, meta["window_stats"]["window_seconds"])
    idx_list = resample_indices(y, ev["bootstrap_samples"], ev["bootstrap_seed"])
    summ = {m: _summarize(f, y, span, seeds[0]) for m, f in fits.items()}

    low = test["low_history"].to_numpy().astype(bool)          # per-user dataset: entity lacked history
    out_models = {}
    for fam in ("rule",) + ML:
        if fam not in summ["per_user"]:
            continue
        d = _delta(fits["per_user"], fits["global"], summ["per_user"], summ["global"], y, idx_list, fam, ev["ci_level"])
        entry = {"per_user": {k: summ["per_user"][fam]["metrics"][k] for k in ("precision", "recall", "f1", "pr_auc", "false_positive_rate")},
                 "global": {k: summ["global"][fam]["metrics"][k] for k in ("precision", "recall", "f1", "pr_auc", "false_positive_rate")},
                 "delta": d, "per_scenario_recall": {}, "by_history": {}}
        for mode in ("per_user", "global"):
            mid = summ[mode][fam]["model_id"]
            f = fits[mode]["models"][mid]
            pred = f["test_scores"] >= f["threshold"]
            entry["per_scenario_recall"][mode] = per_scenario_recall(test["attack_scenario"].to_numpy(), y, pred)
            entry["by_history"][mode] = {
                name: {"windows": int(m.sum()), "attack_windows": int(y[m].sum()),
                       "recall": float(pred[m & (y == 1)].mean()) if (m & (y == 1)).any() else None,
                       "false_positive_rate": float(pred[m & (y == 0)].mean()) if (m & (y == 0)).any() else None}
                for name, m in (("enough_history", ~low), ("low_history", low))}
        if fam == "iforest":
            entry["per_user"]["seed_stats"] = summ["per_user"][fam]["seed_stats"]
            entry["global"]["seed_stats"] = summ["global"][fam]["seed_stats"]
        out_models[fam] = entry

    changed = [f for f in names if not np.allclose(data["per_user"][f].to_numpy(float), data["global"][f].to_numpy(float))]
    root = Path(reports_root) if reports_root else PROJECT_ROOT / "reports"
    mdir, fdir = root / "metrics" / meta["dataset"] / "h1", root / "figures" / meta["dataset"] / "h1"
    caveats = [
        "Only baseline-dependent features differ between the two arms; windows, split, models and seeds are identical.",
        "Same decision rule for every model: supported only if the paired PR-AUC delta CI is above 0.",
        "Bootstrap resamples windows independently although incidents span several windows: intervals are optimistic.",
        f"The test split has {int(y.sum())} attack windows; {int(low.sum())} test windows are low-history in the per-user arm.",
    ]
    if meta["dataset"] == "synthetic":
        caveats.append("Synthetic users are generated with individual home IPs and active hours, which favours per-user "
                       "baselines by construction; real users may be less distinct.")
    dump_json({"dataset": meta["dataset"], "hypothesis": "per-user baselines improve detection over a global baseline",
               "features_that_differ": changed, "n_test_windows": int(len(y)), "n_test_attacks": int(y.sum()),
               "seeds": seeds, "models": out_models, "caveats": caveats}, mdir / "h1.json")
    plots.plot_h1(out_models, fdir / "h1.png")
    return {"metrics_dir": str(mdir), "figures_dir": str(fdir), "models": out_models, "features_that_differ": changed}
