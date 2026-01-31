from __future__ import annotations
import numpy as np
import pandas as pd


class ContextZScoreBaseline:
    """
    Compute contextual z-score using (hour, dow) grouped mean/std from train.
    Score per timestamp:
    score = |x - mu(hour,dow)| / (std(hour,dow) + eps)
    """
    def __init__(self, target_col: str = "total", eps: float = 1e-9):
        self.target_col = target_col
        self.eps = float(eps)
        self.stats_: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> "ContextZScoreBaseline":
        df = train_df.copy()
        if "hour" not in df.columns or "dow" not in df.columns:
            raise ValueError("train_df must contain 'hour' and 'dow' columns. Enable calendar features.")
        g = df.groupby(["hour", "dow"])[self.target_col]
        st = g.agg(["mean", "std"]).reset_index()
        st["std"] = st["std"].fillna(0.0)
        self.stats_ = st
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.stats_ is None:
            raise RuntimeError("Call fit() first.")
        x = df.copy()
        m = x.merge(self.stats_, on=["hour", "dow"], how="left")
        mu = m["mean"].to_numpy(dtype=float)
        sd = m["std"].to_numpy(dtype=float)
        sd = np.where(sd <= 0, 1.0, sd)
        val = m[self.target_col].to_numpy(dtype=float)
        z = np.abs(val - mu) / (sd + self.eps)
        # 점수는 양수, 클수록 이상
        return z
