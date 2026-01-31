from __future__ import annotations
import numpy as np
import pandas as pd


def aggregate_point_scores_to_windows(
    ts: np.ndarray,
    point_scores: np.ndarray,
    win_len: int,
    stride: int,
    agg: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_end: window end timestamps
      w_scores: window scores
    """
    ts = np.asarray(ts)
    s = np.asarray(point_scores, dtype=float)
    n = len(ts)

    t_list = []
    w_list = []
    for start in range(0, n - win_len + 1, stride):
        end = start + win_len
        seg = s[start:end]
        if agg == "max":
            w = float(np.max(seg))
        elif agg == "mean":
            w = float(np.mean(seg))
        elif agg == "p95":
            w = float(np.quantile(seg, 0.95))
        else:
            raise ValueError(f"Unknown agg: {agg}")
        t_list.append(ts[end - 1])
        w_list.append(w)

    return np.array(t_list), np.array(w_list, dtype=float)


def make_base_labels(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": df["ts"].values,
        "is_anomaly": 0,
        "anomaly_type": "none",
        "group_id": -1
    })
