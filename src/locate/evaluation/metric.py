
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support

def _per_class_table(y_true, y_pred, labels, *, label_type: str):
    """Build a per-class precision/recall/F1/support table for given labels order."""
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average=None
    )
    tbl = pd.DataFrame({
        label_type: labels,
        "precision": p,
        "recall": r,
        "F1": f1,
        "support": sup.astype(int),
    })
    return tbl

def score_cn_predictions(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    sample_name: str = "Sample",
    pos_col_truth: str = "pos",
    maj_col_truth: str = "major",
    min_col_truth: str = "minor",
    pos_col_pred: str = "pos",
    maj_col_pred: str = "CN_Major",
    min_col_pred: str = "CN_minor",
) -> Dict[str, pd.DataFrame]:
    """
    Compare true allele-specific CN (per-position) vs inferred CN.

    Returns a dict of DataFrames:
      - 'summary' : one-row global metrics
      - 'per_class_total' : per total-CN precision/recall/F1/support
      - 'per_class_pair'  : per (major:minor) precision/recall/F1/support
      - 'confusion_total' : confusion matrix (truth rows vs pred cols) on total CN
      - 'confusion_pair'  : confusion matrix on allele pairs (major:minor)
    """

    # --- prepare truth ---
    truth = truth_df[[pos_col_truth, maj_col_truth, min_col_truth]].copy()
    truth = truth.rename(columns={
        pos_col_truth: "pos",
        maj_col_truth: "t_major",
        min_col_truth: "t_minor"
    })
    # ensure integers and sort so major >= minor
    truth[["t_major", "t_minor"]] = truth[["t_major", "t_minor"]].astype(int)
    truth[["t_major", "t_minor"]] = np.sort(truth[["t_major", "t_minor"]].values, axis=1)[:, ::-1]
    truth["t_total"] = truth["t_major"] + truth["t_minor"]
    truth["t_pair"]  = truth["t_major"].astype(str) + ":" + truth["t_minor"].astype(str)

    # --- prepare predictions ---
    pred = pred_df[[pos_col_pred, maj_col_pred, min_col_pred]].copy()
    pred = pred.rename(columns={
        pos_col_pred: "pos",
        maj_col_pred: "p_major",
        min_col_pred: "p_minor"
    })
    pred[["p_major", "p_minor"]] = pred[["p_major", "p_minor"]].astype(int)
    pred[["p_major", "p_minor"]] = np.sort(pred[["p_major", "p_minor"]].values, axis=1)[:, ::-1]
    pred["p_total"] = pred["p_major"] + pred["p_minor"]
    pred["p_pair"]  = pred["p_major"].astype(str) + ":" + pred["p_minor"].astype(str)

    # --- merge on position ---
    merged = truth.merge(pred, on="pos", how="inner")
    if merged.empty:
        raise ValueError("No overlapping positions between truth and predictions.")

    # --- global metrics ---
    allelic_accuracy = (merged["t_pair"] == merged["p_pair"]).mean()
    total_accuracy   = (merged["t_total"] == merged["p_total"]).mean()
    total_mae        = np.abs(merged["t_total"] - merged["p_total"]).mean()

    # labels (ensure stable, meaningful ordering)
    # total CN labels: numeric ascending
    labels_total = sorted(set(merged["t_total"]).union(set(merged["p_total"])))
    
    # pair labels: sort by (total, major, minor) descending major then minor within total
    def _pair_key(s):
        a, b = map(int, s.split(":"))
        return (a + b, a, b)  # total first, then major, then minor
    labels_pair = sorted(set(merged["t_pair"]).union(set(merged["p_pair"])), key=_pair_key)

    # --- per-class tables ---
    per_class_total = _per_class_table(
        merged["t_total"].to_numpy(),
        merged["p_total"].to_numpy(),
        labels_total,
        label_type="totalCN"
    )
    per_class_pair = _per_class_table(
        merged["t_pair"].to_numpy(),
        merged["p_pair"].to_numpy(),
        labels_pair,
        label_type="pair"
    )

    # --- macro averages from the per-class tables (unweighted mean) ---
    per_class_total = per_class_total.query('support > 0')
    precision_total_macro = per_class_total["precision"].mean() if not per_class_total.empty else 0.0
    recall_total_macro    = per_class_total["recall"].mean()    if not per_class_total.empty else 0.0
    f1_total_macro        = per_class_total["F1"].mean()        if not per_class_total.empty else 0.0

    per_class_pair = per_class_pair.query('support > 0')
    precision_pair_macro = per_class_pair["precision"].mean() if not per_class_pair.empty else 0.0
    recall_pair_macro    = per_class_pair["recall"].mean()    if not per_class_pair.empty else 0.0
    f1_pair_macro        = per_class_pair["F1"].mean()        if not per_class_pair.empty else 0.0

    # --- confusion matrices ---
    confusion_total = pd.crosstab(
        merged["t_total"], merged["p_total"],
        rownames=["truth_totalCN"], colnames=["pred_totalCN"]
    ).reindex(index=labels_total, columns=labels_total, fill_value=0)

    confusion_pair = pd.crosstab(
        merged["t_pair"], merged["p_pair"],
        rownames=["truth_pair"], colnames=["pred_pair"]
    ).reindex(index=labels_pair, columns=labels_pair, fill_value=0)

    # --- summary (one row) ---
    summary = pd.DataFrame([{
        "sample": sample_name,
        "n_positions": int(len(merged)),
        "allelic_accuracy": float(allelic_accuracy),
        "total_cn_accuracy": float(total_accuracy),
        "total_cn_mae": float(total_mae),
        "precision_totalCN_macro": float(precision_total_macro),
        "recall_totalCN_macro": float(recall_total_macro),
        "f1_totalCN_macro": float(f1_total_macro),
        "precision_pair_macro": float(precision_pair_macro),
        "recall_pair_macro": float(recall_pair_macro),
        "f1_pair_macro": float(f1_pair_macro),
    }])

    return {
        "summary": summary,
        "per_class_total": per_class_total,
        "per_class_pair": per_class_pair,
        "confusion_total": confusion_total,
        "confusion_pair": confusion_pair,
    }

