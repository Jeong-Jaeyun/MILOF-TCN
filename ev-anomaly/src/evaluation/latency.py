from __future__ import annotations
import numpy as np
import pandas as pd

def detection_delay_hours(curves: pd.DataFrame) -> dict:
    """
    curves columns required:
      ts_end, y_true, y_pred, group_id
    Returns:
      avg_delay_hours, median_delay_hours, detected_rate, per_group (DataFrame)
    """
    df = curves.copy()
    df["ts_end"] = pd.to_datetime(df["ts_end"])
    df = df.sort_values("ts_end").reset_index(drop=True)

    # anomaly event groups only
    gdf = df[df["group_id"] >= 0].copy()
    if gdf.empty:
        return {
            "avg_delay_hours": None,
            "median_delay_hours": None,
            "detected_rate": None,
            "per_group": pd.DataFrame(columns=["group_id", "start_ts", "first_detect_ts", "delay_hours", "detected"]),
        }

    per = []
    for gid, part in gdf.groupby("group_id"):
        part = part.sort_values("ts_end")
        # event start: first window end timestamp that overlaps the event as labeled positive
        start_ts = part.loc[part["y_true"] == 1, "ts_end"].min()
        det_ts = part.loc[(part["y_true"] == 1) & (part["y_pred"] == 1), "ts_end"].min()

        detected = 0
        delay = None
        if pd.notna(det_ts) and pd.notna(start_ts):
            detected = 1
            delay = (det_ts - start_ts).total_seconds() / 3600.0

        per.append([int(gid), start_ts, det_ts, delay, detected])

    per_df = pd.DataFrame(per, columns=["group_id", "start_ts", "first_detect_ts", "delay_hours", "detected"])
    detected_rate = float(per_df["detected"].mean()) if len(per_df) else None
    delays = per_df.loc[per_df["detected"] == 1, "delay_hours"].dropna().to_numpy(dtype=float)

    return {
        "avg_delay_hours": float(np.mean(delays)) if delays.size else None,
        "median_delay_hours": float(np.median(delays)) if delays.size else None,
        "detected_rate": detected_rate,
        "per_group": per_df,
    }
