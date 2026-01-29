from __future__ import annotations
import numpy as np

def threshold_by_quantile(scores: np.ndarray, q: float) -> float:
    scores = np.asarray(scores, dtype=float)
    q = float(q)
    return float(np.quantile(scores, q))

def threshold_by_mean_std(scores: np.ndarray, k: float = 3.0) -> float:
    scores = np.asarray(scores, dtype=float)
    return float(scores.mean() + float(k) * scores.std(ddof=0))
