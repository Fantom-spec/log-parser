"""Service: final held-out evaluation using the thresholds frozen during training."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from logad.config import PROJECT_ROOT
from logad.evaluation.compare_models import comparison_table
from logad.evaluation.latency import ms_per_1000, time_call
from logad.evaluation.metrics import (
    aggregate, bootstrap_ci, per_scenario_recall, threshold_metrics)
from logad.evaluation import plots
from logad.io import dump_json, load_json
from logad.models.registry import ModelRegistry
from logad.preprocessing.pipeline import load_dataset
from logad.services.train import _span_days

FAMILIES = ["rule", "iforest", "ocsvm", "lof"]


def evaluate_models(dataset_dir, cfg: dict, tag: str = "all", reports_root=None) -> dict:
    dataset_dir = Path(dataset_dir)
    models_dir = dataset_dir / "models" / tag
    reg = ModelRegistry.load(models_dir)
    thresholds = load_json(models_dir / "thresholds.json")
    pre = joblib.load(models_dir / "preprocessor.joblib")
    df, names, meta = load_dataset(dataset_dir)

    test = df[(df["split"] == "test") & df["label"].notna()].reset_index(drop=True)
    if test["label"].sum() == 0:
        raise ValueError("test split has no attack windows; metrics are undefined")
    y = test["label"].astype(int).to_numpy()
    span = _span_days(test, meta["window_stats"]["window_seconds"])
    dataset = meta["dataset"]
    root = Path(reports_root) if reports_root else PROJECT_ROOT / "reports"
    mdir, fdir = root / "metrics" / dataset / tag, root / "figures" / dataset / tag

    t_raw = lambda: pre.transform_raw(test)
    t_scaled = lambda: pre.transform_scaled(test)
    X = {"raw": t_raw(), "scaled": t_scaled()}
    transform_ms = {k: ms_per_1000(time_call(f), len(test)) for k, f in (("raw", t_raw), ("scaled", t_scaled))}

    scores, metrics, latency = {}, {}, {}
    for mid, entry in reg.entries.items():
        model = reg.load_model(mid)
        Xi = X[model.input_kind]
        scores[mid] = model.score(Xi)
        metrics[mid] = threshold_metrics(y, scores[mid], thresholds[mid]["threshold"], span)
        latency[mid] = {"transform_ms_per_1000": transform_ms[model.input_kind],
                        "score_ms_per_1000": ms_per_1000(time_call(lambda: model.score(Xi)), len(test))}

    if_ids = sorted((m for m in reg.entries if reg.entries[m]["family"] == "iforest"),
                    key=lambda m: reg.entries[m]["seed"])
    primary_seed = cfg["models"]["seeds"][0]
    if_primary = f"iforest_seed{primary_seed}" if f"iforest_seed{primary_seed}" in reg.entries else if_ids[0]
    primary_ids = {f: (if_primary if f == "iforest" else f) for f in FAMILIES if (f != "iforest" and f in reg.entries) or f == "iforest"}

    ev = cfg["evaluation"]
    primary = {}
    for fam, mid in primary_ids.items():
        thr = thresholds[mid]["threshold"]
        primary[fam] = {
            "model_id": mid, "version": reg.entries[mid]["version"], "metrics": metrics[mid],
            "bootstrap_ci": bootstrap_ci(y, scores[mid], thr, ev["bootstrap_samples"], ev["bootstrap_seed"], ev["ci_level"]),
            "per_scenario_recall": per_scenario_recall(test["attack_scenario"].to_numpy(), y, scores[mid] >= thr),
        }
    seed_stats = aggregate([metrics[m] for m in if_ids]) if if_ids else None

    table = comparison_table(primary, seed_stats)
    mdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(mdir / "comparison.csv", index=False, float_format="%.6g")
    import pandas as pd
    pd.DataFrame([{"model_id": m, "seed": reg.entries[m]["seed"], **{k: metrics[m][k] for k in (
        "threshold", "precision", "recall", "f1", "pr_auc", "false_positive_rate", "n_alerts")}}
        for m in if_ids]).to_csv(mdir / "seeds.csv", index=False, float_format="%.6g")

    balance = {sp: {"windows": int((df.split == sp).sum()), "attack": int(((df.split == sp) & (df.label == 1)).sum()),
                    "normal": int(((df.split == sp) & (df.label == 0)).sum())} for sp in ("train", "val", "test")}
    caveats = [
        "Thresholds were calibrated on the validation split (labels used there, documented); test labels were used only for these metrics.",
        "Bootstrap intervals resample windows independently although an incident spans several windows, so they are optimistic.",
        "Scores from different models are not comparable; each model has its own threshold and operating point.",
        "Single chronological split; results depend on which incidents fall in the test period.",
    ]
    if dataset == "synthetic":
        caveats.append("Synthetic data: attacker IPs are partly planted in the blocklist, so threat_intel features "
                       "are circular here; attack patterns are generator-defined, not real traffic.")
    dump_json({
        "dataset": dataset, "tag": tag, "features": reg.entries[if_primary]["features"],
        "class_balance": balance, "test_window_span_days": span, "thresholds": thresholds,
        "models": primary, "iforest_seed_stats": seed_stats, "caveats": caveats,
        "model_ids_all": list(reg.entries),
    }, mdir / "metrics.json")
    dump_json({"per_model": latency, "n_test_windows": len(test),
               "fit_seconds": load_json(models_dir / "fit_times.json"),
               "note": "wall-clock on this machine; not part of the reproducible metrics"}, mdir / "latency.json")

    p_scores = {f: scores[m] for f, m in primary_ids.items()}
    p_thr = {f: thresholds[m]["threshold"] for f, m in primary_ids.items()}
    plots.plot_pr_curves(p_scores, y, fdir / "pr_curves.png")
    if len(p_scores) == 4:
        plots.plot_score_distributions(p_scores, y, p_thr, fdir / "score_distributions.png")
        val = df[(df.split == "val") & df.label.notna()]
        Xv = {"raw": pre.transform_raw(val), "scaled": pre.transform_scaled(val)}
        v_scores = {f: reg.load_model(m).score(Xv[reg.entries[m]["input_kind"]]) for f, m in primary_ids.items()}
        plots.plot_threshold_tradeoff(v_scores, val["label"].astype(int).to_numpy(), p_thr, fdir / "threshold_tradeoff.png")
    plots.plot_confusion_matrices({f: {k: primary[f]["metrics"][k] for k in ("tp", "fp", "fn", "tn")} for f in primary},
                                  fdir / "confusion_matrices.png")
    return {"metrics_dir": str(mdir), "figures_dir": str(fdir), "comparison": table, "primary": primary,
            "seed_stats": seed_stats}
