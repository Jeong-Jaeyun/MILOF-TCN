from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
)

from src.evaluation.latency import detection_delay_hours


def _false_alarms_per_day(curves: pd.DataFrame) -> float | None:
    """
    False alarms/day computed on window timeline:
    count windows where y_pred=1 and y_true=0, aggregated per calendar day of ts_end.
    """
    df = curves.copy()
    if df.empty:
        return None
    df["ts_end"] = pd.to_datetime(df["ts_end"])
    df["date"] = df["ts_end"].dt.date
    fa = df[(df["y_pred"] == 1) & (df["y_true"] == 0)].groupby("date").size()
    if fa.empty:
        return 0.0
    return float(fa.mean())


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    # requires both classes present
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


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
    roc_auc = _safe_roc_auc(y_true, scores)

    # MCC / Balanced Acc / Confusion
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    bacc = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    curves = pd.DataFrame({
        "ts_end": ts_end,
        "y_true": y_true,
        "score": scores,
        "y_pred": y_pred,
    })
    curves["group_id"] = np.asarray(group_id, dtype=int) if group_id is not None else -1

    latency = detection_delay_hours(curves)
    fa_per_day = _false_alarms_per_day(curves)

    metrics = {
        "threshold": float(threshold),

        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "balanced_acc": bacc,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "false_alarms_per_day": fa_per_day,

        "avg_delay_hours": latency["avg_delay_hours"],
        "median_delay_hours": latency["median_delay_hours"],
        "detected_rate": latency["detected_rate"],
    }
    return metrics, curves
