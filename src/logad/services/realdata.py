"""Service: analysis of real, UNLABELED logs (e.g. Loghub OpenSSH).

Deliberately reports NO precision/recall/F1 (there are no labels). It reports score distributions,
windows per tier, top flagged windows with evidence, rule hits, parse coverage, blocklist overlap, a clearly
labelled weak-proxy agreement check, and a cross-domain check with the synthetic-trained model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from logad.config import PROJECT_ROOT, merge_config, resolve_path
from logad.evaluation import plots
from logad.io import dump_json, load_json
from logad.preprocessing.pipeline import load_dataset
from logad.scoring.rules import TIERS, apply_rules
from logad.scoring.scorer import Scorer
from logad.services.prepare import prepare_dataset
from logad.services.train_unlabeled import train_unlabeled

QS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]


def _tier_counts(scored, split=None):
    s = scored if split is None else scored[scored["split"] == split]
    return {t: int((s["tier"] == t).sum()) for t in TIERS}


def _cross_domain(real_df, real_meta, cfg, cross_dir: Path, log_path, year) -> dict:
    """Score the real log with the synthetic-trained model; summarise the shift (a limitation, not a result)."""
    try:
        sc = Scorer(cross_dir, cfg, "all")
    except Exception as exc:                                                  # noqa: BLE001
        return {"available": False, "reason": f"cross-domain model not usable: {exc}"}
    if sc.group_by != "user":
        return {"available": False, "reason": "cross-domain model is not grouped by user"}
    try:
        table, info = sc.feature_table_from_log(log_path, year)
    except Exception as exc:                                                  # noqa: BLE001
        return {"available": False, "reason": f"scoring failed: {exc}"}
    scored = sc.score_features(table)
    shifts = []
    for f in sc.features:
        p = sc.profile[f]
        shifts.append({"feature": f, "real_median": float(np.median(table[f])), "synthetic_normal_median": p["median"],
                       "shift_in_robust_units": float((np.median(table[f]) - p["median"]) / p["scale"])})
    shifts.sort(key=lambda d: -abs(d["shift_in_robust_units"]))
    flagged = float((scored["tier"] != "NORMAL").mean())
    return {"available": True, "model": sc.model_id, "grouping": "user", "windows": int(len(scored)),
            "tiers": {t: int((scored["tier"] == t).sum()) for t in TIERS}, "share_flagged": flagged,
            "score_quantiles": {str(q): float(np.quantile(scored["score"], q)) for q in (0.5, 0.9, 0.99)},
            "most_shifted_features": shifts[:8],
            "limitation": "The model was trained on synthetic normal behaviour grouped by user; a large flagged share mostly reflects "
                          "domain shift (different users, rates and attack-heavy traffic), not detection accuracy."}


def analyze_real(dataset_dir, cfg: dict, tag: str = "unlabeled", cross_domain_dir=None, reports_root=None) -> dict:
    dataset_dir = Path(dataset_dir)
    df, names, meta = load_dataset(dataset_dir)
    scorer = Scorer(dataset_dir, cfg, tag)
    scored = scorer.score_features(df)
    parse = load_json(dataset_dir / "parse_report.json")
    sc_json = load_json(dataset_dir / "models" / tag / "scoring.json")
    s = scored["score"].to_numpy()
    hist, edges = np.histogram(s, bins=20)
    out = {
        "dataset": meta["dataset"], "group_by": scorer.group_by, "n_windows": int(len(df)), "model_id": scorer.model_id,
        "labels": "none (unlabeled): no precision, recall or F1 are computed",
        "parse": {k: parse[k] for k in ("total_lines", "event_lines", "malformed_lines", "not_sshd_lines", "unsupported_sshd_lines",
                                        "ignored_redundant_lines", "coverage_all_lines", "coverage_sshd_lines", "year_rollovers",
                                        "duplicate_lines_dropped", "timestamp_regressions") if k in parse},
        "unsupported_top_patterns": parse.get("unsupported_top_patterns", {}),
        "unattributed_events": meta["window_stats"]["unattributed_events"],
        "score_distribution": {"quantiles": {str(q): float(np.quantile(s, q)) for q in QS},
                               "histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
                               "by_split_median": {sp: float(np.median(scored.loc[scored.split == sp, "score"]))
                                                   for sp in ("train", "val", "test") if (scored.split == sp).any()}},
        "tier_thresholds": scorer.tier_thr, "tier_basis": sc_json["note"],
        "tiers": {"all": _tier_counts(scored), **{sp: _tier_counts(scored, sp) for sp in ("train", "val", "test")}},
    }
    hits, off = apply_rules(df, scorer.rules_cfg)
    out["rules"] = {"hits_total": {r: int(hits[r].sum()) for r in hits.columns}, "unavailable": off,
                    "flagged_by_rule_only": int(((scored["model_tier_idx"] == 0) & (scored["tier_idx"] > 0)).sum())}
    top = scored.sort_values("score", ascending=False).head(25)
    out["top_windows"] = [{"entity": r.entity, "window_start": str(r.window_start), "score": round(float(r.score), 4), "tier": r.tier,
                           "rules": list(r.rules), "evidence": list(r.evidence)} for r in top.itertuples()]
    # blocklist overlap (only if the threat_intel family exists for this dataset)
    if "blocklisted_ip_count" in df.columns and "blocklisted_ip_count" in names:
        bl = (df["blocklisted_ip_count"] > 0).to_numpy()
        fl = (scored["tier_idx"] > 0).to_numpy()
        out["blocklist_overlap"] = {"windows_with_blocklisted_ip": int(bl.sum()), "share_of_windows": float(bl.mean()),
                                    "share_of_flagged_with_blocklisted_ip": float(bl[fl].mean()) if fl.any() else None,
                                    "share_of_unflagged_with_blocklisted_ip": float(bl[~fl].mean()) if (~fl).any() else None,
                                    "note": "Old logs vs current blocklists can give a low match rate."}
    else:
        out["blocklist_overlap"] = {"available": False, "reason": "threat_intel features not enabled for this dataset"}
    # clearly labelled weak proxy: NOT ground truth, and correlated with features the model sees
    k = cfg["unlabeled"]["proxy_min_failed"]
    if "failed_count" in df.columns:
        proxy = (df["failed_count"] >= k).to_numpy().astype(int)
        auc = float(roc_auc_score(proxy, s)) if 0 < proxy.sum() < len(proxy) else None
        by_tier = {t: {"windows": int((scored.tier == t).sum()),
                       "proxy_rate": (float(proxy[(scored.tier == t).to_numpy()].mean()) if (scored.tier == t).any() else None)} for t in TIERS}
        out["weak_proxy"] = {"definition": f"window has at least {k} failed logins", "roc_auc_of_score_vs_proxy": auc,
                             "proxy_rate_by_tier": by_tier, "proxy_windows": int(proxy.sum()),
                             "warning": "WEAK PROXY, NOT GROUND TRUTH: the proxy is built from a feature the model uses, so agreement is partly circular and is not accuracy."}
    cd_dir = Path(cross_domain_dir) if cross_domain_dir else resolve_path(cfg["paths"]["processed_dir"]) / cfg["real"]["cross_domain_dataset"]
    out["cross_domain"] = _cross_domain(df, meta, cfg, cd_dir, meta["inputs"]["log"], meta["config"]["dataset"]["log_year"]) \
        if cd_dir.exists() else {"available": False, "reason": f"no model at {cd_dir}"}
    out["caveats"] = [
        "No labels: nothing here measures detection accuracy. Tiers are score quantiles on the validation split.",
        "Attack-heavy logs make 'normal' hard to define; the model learns whatever dominates the training windows.",
        "The weak proxy is built from a model input and is only a sanity check.",
        "Cross-domain scoring with the synthetic-trained model is a distribution-shift demonstration, not an evaluation.",
    ]
    root = Path(reports_root) if reports_root else PROJECT_ROOT / "reports"
    mdir, fdir = root / "metrics" / meta["dataset"], root / "figures" / meta["dataset"]
    dump_json(out, mdir / "realdata.json")
    plots.plot_realdata(s, scorer.tier_thr, out["tiers"], fdir / "realdata" / "realdata.png")
    return out


def run_real_analysis(log_path, cfg: dict, name: str | None = None, year: int | None = None, group_by: str | None = None,
                      out_root=None, reports_root=None, cross_domain_dir=None) -> dict:
    """Raw unlabeled log -> prepare (leak-free split) -> train unlabeled -> analysis report."""
    r = cfg["real"]
    name, year, group_by = name or r["dataset"], year or r["year"], group_by or r["group_by"]
    bl = resolve_path(r["blocklist_path"])
    over = {"dataset": {"name": name, "log_year": int(year)}, "windowing": {"group_by": group_by}}
    if bl.exists():
        over["threat_intel"] = {"blocklist_path": str(bl), "manifest_path": str(resolve_path(r["manifest_path"]))}
        over["features"] = {"threat_intel_enabled": True}
    else:
        over["features"] = {"threat_intel_enabled": False}
    c = merge_config(cfg, over)
    out = Path(out_root) if out_root else resolve_path(cfg["paths"]["processed_dir"]) / name
    meta = prepare_dataset(log_path, out, c, None, name)
    train_unlabeled(out, c)
    res = analyze_real(out, c, "unlabeled", cross_domain_dir, reports_root)
    res["prepare_warnings"] = meta["warnings"]
    return res
