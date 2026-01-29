from __future__ import annotations
import pandas as pd

def slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    ts = pd.to_datetime(df["ts"])
    mask = (ts >= pd.to_datetime(start)) & (ts <= pd.to_datetime(end) + pd.Timedelta(hours=23))
    return df.loc[mask].copy().reset_index(drop=True)

def make_splits(df: pd.DataFrame, split_cfg: dict) -> dict[str, pd.DataFrame]:
    return {
        "train": slice_by_date(df, split_cfg["train_start"], split_cfg["train_end"]),
        "val":   slice_by_date(df, split_cfg["val_start"],   split_cfg["val_end"]),
        "test":  slice_by_date(df, split_cfg["test_start"],  split_cfg["test_end"]),
    }
