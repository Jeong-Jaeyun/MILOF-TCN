from __future__ import annotations
import pandas as pd
import numpy as np

def read_kepco_raw(csv_path: str, date_col: str, mode_col: str, hour_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    # 일자 파싱
    df[date_col] = pd.to_datetime(df[date_col])
    # 숫자 변환(쉼표 대비)
    for c in hour_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df

def wide_to_long(df: pd.DataFrame, date_col: str, mode_col: str, hour_cols: list[str]) -> pd.DataFrame:
    # melt
    m = df.melt(id_vars=[date_col, mode_col], value_vars=hour_cols, var_name="hour_str", value_name="load")
    # "00시" -> 0
    m["hour"] = m["hour_str"].str.replace("시", "", regex=False).astype(int)
    m["ts"] = m[date_col] + pd.to_timedelta(m["hour"], unit="h")
    out = m[["ts", mode_col, "load"]].copy()
    out = out.sort_values("ts").reset_index(drop=True)
    return out

def pivot_modes(long_df: pd.DataFrame, mode_col: str, fast_values: list[str], slow_values: list[str]) -> pd.DataFrame:
    # mode 표준화
    def map_mode(x: str) -> str:
        if x in fast_values:
            return "fast"
        if x in slow_values:
            return "slow"
        return "other"
    long_df = long_df.copy()
    long_df["mode_std"] = long_df[mode_col].astype(str).map(map_mode)
    p = long_df.pivot_table(index="ts", columns="mode_std", values="load", aggfunc="sum").reset_index()
    for c in ["fast", "slow"]:
        if c not in p.columns:
            p[c] = 0.0
    p = p.sort_values("ts").reset_index(drop=True)
    p["total"] = p["fast"].fillna(0) + p["slow"].fillna(0)
    denom = p["total"].replace(0, np.nan)
    p["ratio_fast"] = (p["fast"] / denom).fillna(0.0)
    return p[["ts", "fast", "slow", "total", "ratio_fast"]]
