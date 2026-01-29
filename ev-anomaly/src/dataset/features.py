from __future__ import annotations
import pandas as pd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["ts"])
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

def add_rolling(df: pd.DataFrame, windows_hours: list[int], cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts").reset_index(drop=True)
    out = out.set_index(pd.to_datetime(out["ts"]))
    for w in windows_hours:
        for c in cols:
            out[f"{c}_roll_mean_{w}"] = out[c].rolling(w, min_periods=max(2, w//4)).mean()
            out[f"{c}_roll_std_{w}"]  = out[c].rolling(w, min_periods=max(2, w//4)).std()
    out = out.reset_index(drop=True)
    return out
