from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestBaseline:
    """
    Fit IsolationForest on point-wise features.
    Score per timestamp: higher means more anomalous.
    score = -decision_function(x)
    """
    def __init__(
        self,
        feature_cols: list[str],
        n_estimators: int = 200,
        contamination: str | float = "auto",
        random_state: int = 42,
    ):
        self.feature_cols = list(feature_cols)
        self.model = IsolationForest(
            n_estimators=int(n_estimators),
            contamination=contamination,
            random_state=int(random_state),
            n_jobs=-1,
        )

    def fit(self, train_df: pd.DataFrame) -> "IsolationForestBaseline":
        X = train_df[self.feature_cols].to_numpy(dtype=float)
        self.model.fit(X)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=float)
        # decision_function: 클수록 정상 → 부호를 뒤집어 이상 점수로 사용
        s = -self.model.decision_function(X)
        return np.asarray(s, dtype=float)
