"""Matplotlib figures for the model comparison (Agg backend, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

NAMES = {"rule": "Rule (failed_count)", "iforest": "Isolation Forest",
         "ocsvm": "One-Class SVM", "lof": "Local Outlier Factor"}


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curves(scores: dict, y, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{NAMES[k]} (AP={average_precision_score(y, s):.3f})")
    ax.axhline(np.mean(y), ls="--", c="grey", lw=1, label=f"chance ({np.mean(y):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall, held-out test", ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, path)


def plot_score_distributions(scores: dict, y, thresholds: dict, path):
    y = np.asarray(y).astype(bool)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (k, s) in zip(axes.ravel(), scores.items()):
        bins = np.histogram_bin_edges(s, bins=40)
        ax.hist(s[~y], bins=bins, alpha=0.65, label="normal")
        ax.hist(s[y], bins=bins, alpha=0.65, label="attack")
        ax.axvline(thresholds[k], c="k", ls="--", lw=1, label="threshold")
        ax.set_yscale("log")
        ax.set(title=NAMES[k], xlabel="anomaly score (higher = more anomalous)", ylabel="windows")
        ax.legend(fontsize=7)
    _save(fig, path)


def plot_confusion_matrices(cms: dict, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3.2))
    for ax, (k, c) in zip(np.atleast_1d(axes), cms.items()):
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "alert"],
               yticklabels=["normal", "attack"], title=NAMES[k], xlabel="predicted", ylabel="actual")
    _save(fig, path)


def plot_threshold_tradeoff(val_scores: dict, y_val, thresholds: dict, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (k, s) in zip(axes.ravel(), val_scores.items()):
        p, r, t = precision_recall_curve(y_val, s)
        f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12), 0)
        ax.plot(t, p[:-1], label="precision"); ax.plot(t, r[:-1], label="recall"); ax.plot(t, f1, label="F1")
        ax.axvline(thresholds[k], c="k", ls="--", lw=1, label="chosen")
        ax.set(title=NAMES[k], xlabel="threshold (validation)", ylim=(0, 1.02))
        ax.legend(fontsize=7)
    _save(fig, path)


def plot_ablation(results: list, path):
    """Left: drop-one-family change in PR-AUC with bootstrap CI. Right: PR-AUC of each family alone."""
    models = [m for m in ("iforest", "ocsvm", "lof") if m in results[0]["models"]]
    drops = [r for r in results if r["mode"] == "drop"]
    onlys = [r for r in results if r["mode"] == "only"]
    ref = results[0]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.8 / len(models)
    ax = axes[0]
    for j, m in enumerate(models):
        xs = np.arange(len(drops)) + j * w
        d = [r["delta_vs_all"][m]["delta_pr_auc"] for r in drops]
        lo = [r["delta_vs_all"][m]["delta_pr_auc_ci"][0] for r in drops]
        hi = [r["delta_vs_all"][m]["delta_pr_auc_ci"][1] for r in drops]
        ax.bar(xs, d, w, label=NAMES[m], color=f"C{j}")
        ax.vlines(xs, lo, hi, color="k", lw=1)          # CI drawn as bounds: a percentile CI need not bracket the point
        ax.plot(xs, lo, "_", c="k", ms=4); ax.plot(xs, hi, "_", c="k", ms=4)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(np.arange(len(drops)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([r["family"] for r in drops], rotation=20, fontsize=8)
    ax.set(ylabel="change in test PR-AUC vs all features", title="Drop one family (95% bootstrap CI)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for j, m in enumerate(models):
        xs = np.arange(len(onlys)) + j * w
        ax.bar(xs, [r["models"][m]["pr_auc"] for r in onlys], w, label=NAMES[m])
        ax.axhline(ref[m]["pr_auc"], ls="--", lw=0.8, c=f"C{j}")
    ax.set_xticks(np.arange(len(onlys)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([r["family"] for r in onlys], rotation=20, fontsize=8)
    ax.set(ylabel="test PR-AUC", title="One family alone (dashed = all features)", ylim=(0, 1.02))
    _save(fig, path)


def plot_h1(models: dict, path):
    """Left: PR-AUC per model, per-user vs global baseline. Right: per-scenario recall (Isolation Forest)."""
    ms = [m for m in ("iforest", "ocsvm", "lof") if m in models]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax, w = axes[0], 0.38
    for k, mode in enumerate(("per_user", "global")):
        ax.bar(np.arange(len(ms)) + k * w, [models[m][mode]["pr_auc"] for m in ms], w,
               label=mode.replace("_", "-"))
    for i, m in enumerate(ms):
        d = models[m]["delta"]
        lo, hi = d["delta_pr_auc_ci"]
        ax.text(i + w / 2, 1.0, f"Δ={d['delta_pr_auc']:+.3f}\nCI[{lo:+.3f},{hi:+.3f}]\n{d['verdict_pr_auc']}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(ms)) + w / 2)
    ax.set_xticklabels([NAMES[m] for m in ms], fontsize=8)
    ax.set(ylabel="test PR-AUC", ylim=(0, 1.25), title="H1: per-user vs global baseline")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    sc = sorted(models["iforest"]["per_scenario_recall"]["per_user"])
    for k, mode in enumerate(("per_user", "global")):
        ax.bar(np.arange(len(sc)) + k * w, [models["iforest"]["per_scenario_recall"][mode][s]["recall"] for s in sc], w,
               label=mode.replace("_", "-"))
    ax.set_xticks(np.arange(len(sc)) + w / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in sc], fontsize=7)
    ax.set(ylabel="recall at calibrated threshold", ylim=(0, 1.05), title="Isolation Forest recall by attack scenario")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_scoring(split_result: dict, path):
    """Left: attack share per final tier. Right: model-only vs model+rules vs rules-only."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tiers = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ax = axes[0]
    rates = [split_result["tiers"][t]["attack_rate"] or 0 for t in tiers]
    ax.bar(tiers, rates, color=["#7a8", "#cb4", "#e92", "#d44", "#a1c"])
    for i, t in enumerate(tiers):
        tt = split_result["tiers"][t]
        ax.text(i, rates[i] + 0.02, f"{tt['attack_windows']}/{tt['windows']}", ha="center", fontsize=8)
    ax.set(ylim=(0, 1.15), ylabel="share of windows that are attacks", title="Held-out test: attacks per tier")
    ax = axes[1]
    cfgs = [("model_only", "model only"), ("rules_only", "rules only"), ("model_plus_rules", "model + rules")]
    w = 0.26
    for j, (k, lab) in enumerate(cfgs):
        m = split_result["configs"][k]
        ax.bar(np.arange(3) + j * w, [m["precision"], m["recall"], m["f1"]], w, label=lab)
    ax.set_xticks(np.arange(3) + w); ax.set_xticklabels(["precision", "recall", "F1"])
    ax.set(ylim=(0, 1.05), title="Alert decision: tier >= LOW (or any rule)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_realdata(scores, tier_thr: dict, tiers: dict, path):
    """Left: score histogram with tier cut-points. Right: windows per tier by split."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(scores, bins=40, color="#6aa5ff")
    for t, c in (("LOW", "#b7791f"), ("MEDIUM", "#d9730d"), ("HIGH", "#c53030"), ("CRITICAL", "#7e22ce")):
        v = tier_thr.get(t)
        if v is not None and np.isfinite(v):
            ax.axvline(v, color=c, lw=1.2, label=t)
    ax.set(yscale="log", xlabel="anomaly score", ylabel="windows", title="Score distribution (unlabeled real log)")
    ax.legend(fontsize=8)
    ax = axes[1]
    names = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    splits = [s for s in ("train", "val", "test") if s in tiers]
    w = 0.8 / max(len(splits), 1)
    for j, sp in enumerate(splits):
        ax.bar(np.arange(5) + j * w, [tiers[sp][t] for t in names], w, label=sp)
    ax.set_xticks(np.arange(5) + w * (len(splits) - 1) / 2); ax.set_xticklabels(names, fontsize=8)
    ax.set(yscale="log", ylabel="windows", title="Windows per tier (quantile tiers)")
    ax.legend(fontsize=8)
    _save(fig, path)
