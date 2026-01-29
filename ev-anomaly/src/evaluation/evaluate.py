from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from src.evaluation.latency import detection_delay_hours


def evaluate_window_scores(
    ts_end: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    group_id: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    ts_end = np.asarray(ts_end)
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= float(threshold)).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    pr_auc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None

    curves = pd.DataFrame({
        "ts_end": ts_end,
        "y_true": y_true,
        "score": scores,
        "y_pred": y_pred,
    })
    if group_id is not None:
        curves["group_id"] = np.asarray(group_id, dtype=int)
    else:
        curves["group_id"] = -1

    latency = detection_delay_hours(curves)

    metrics = {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "avg_delay_hours": latency["avg_delay_hours"],
        "median_delay_hours": latency["median_delay_hours"],
        "detected_rate": latency["detected_rate"],
    }
    return metrics, curves

def evaluate_edge_pointwise(
    ts: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    group_id: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Point-wise evaluation for Edge detector.
    """
    ts = np.asarray(ts)
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= float(threshold)).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    pr_auc = (
        float(average_precision_score(y_true, scores))
        if len(np.unique(y_true)) > 1 else None
    )

    curves = pd.DataFrame({
        "ts_end": ts,
        "y_true": y_true,
        "score": scores,
        "y_pred": y_pred,
        "group_id": group_id if group_id is not None else -1,
    })

    latency = detection_delay_hours(curves)

    metrics = {
        "edge_threshold": float(threshold),
        "edge_precision": float(p),
        "edge_recall": float(r),
        "edge_f1": float(f1),
        "edge_pr_auc": pr_auc,
        "edge_avg_delay_hours": latency["avg_delay_hours"],
        "edge_median_delay_hours": latency["median_delay_hours"],
        "edge_detected_rate": latency["detected_rate"],
    }
    return metrics, curves