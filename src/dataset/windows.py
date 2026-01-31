from __future__ import annotations
import numpy as np
import pandas as pd

def make_windows(df: pd.DataFrame, labels: pd.DataFrame, feat_cols: list[str], win_len: int, stride: int, label_mode: str):
    X_list, y_list, t_list = [], [], []
    x = df[feat_cols].to_numpy(dtype=np.float32)
    y = labels["is_anomaly"].to_numpy(dtype=np.int64)

    n = len(df)
    for start in range(0, n - win_len + 1, stride):
        end = start + win_len
        Xw = x[start:end]
        yw = y[start:end]
        if label_mode == "any":
            yw_label = int(yw.max() > 0)
        else:
            yw_label = int(yw.mean() > 0.5)

        X_list.append(Xw)
        y_list.append(yw_label)
        t_list.append(df["ts"].iloc[end-1])

    X = np.stack(X_list, axis=0)  # [N, L, C]
    y = np.array(y_list, dtype=np.int64)
    t = np.array(t_list)
    return X, y, t

def make_windows_with_edge_gate(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    edge_flags: np.ndarray,
    feat_cols: list[str],
    win_len: int,
    stride: int,
    label_mode: str,
    max_edge_ratio: float,
):
    """
    edge_flags: boolean array, True if Edge flagged anomaly at that timestep
    max_edge_ratio: if (sum(edge_flags in window)/win_len) > this, drop window
    """
    X_list, y_list, t_list, keep_idx = [], [], [], []

    x = df[feat_cols].to_numpy(dtype=np.float32)
    y = labels["is_anomaly"].to_numpy(dtype=np.int64)
    e = np.asarray(edge_flags, dtype=bool)

    n = len(df)
    for start in range(0, n - win_len + 1, stride):
        end = start + win_len

        edge_ratio = e[start:end].mean()
        if edge_ratio > max_edge_ratio:
            continue  # drop this window

        Xw = x[start:end]
        yw = y[start:end]

        if label_mode == "any":
            yw_label = int(yw.max() > 0)
        else:
            yw_label = int(yw.mean() > 0.5)

        X_list.append(Xw)
        y_list.append(yw_label)
        t_list.append(df["ts"].iloc[end - 1])
        keep_idx.append((start, end - 1))

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, win_len, x.shape[1]), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    t = np.array(t_list)
    return X, y, t, keep_idx
