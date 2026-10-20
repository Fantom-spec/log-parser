"""Comparison table across models (primary model per family + IF seed statistics)."""
from __future__ import annotations

import pandas as pd

COLUMNS = ["model", "model_id", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc",
           "false_positive_rate", "alerts_per_day", "tp", "fp", "fn", "tn", "prevalence"]
DISPLAY = {"rule": "Rule (failed_count)", "iforest": "Isolation Forest", "ocsvm": "One-Class SVM",
           "lof": "Local Outlier Factor"}


def comparison_table(primary: dict, seed_stats: dict | None) -> pd.DataFrame:
    """primary: {family: {"model_id", "metrics"}}. Adds an IF mean over seeds row when available."""
    rows = []
    for fam, d in primary.items():
        m = d["metrics"]
        rows.append({"model": DISPLAY[fam], "model_id": d["model_id"], **{c: m.get(c) for c in COLUMNS[2:]}})
    if seed_stats:
        row = {"model": f"Isolation Forest (mean over {seed_stats['n_seeds']} seeds)", "model_id": "iforest_mean"}
        for k in ("precision", "recall", "f1", "pr_auc", "false_positive_rate", "alerts_per_day"):
            if k in seed_stats:
                row[k] = seed_stats[k]["mean"]
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
