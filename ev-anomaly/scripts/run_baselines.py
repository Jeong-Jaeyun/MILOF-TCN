from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.common.io import load_yaml
from src.common.seed import set_seed
from src.common.log import create_run_context, save_config, save_metrics, save_curves_csv
from src.common.metrics import aggregate_point_scores_to_windows
from src.training.thresholds import threshold_by_quantile, threshold_by_mean_std
from src.evaluation.evaluate import evaluate_window_scores
from src.common.plot import save_score_plot, save_pred_plot

from src.models.baseline_stats import ContextZScoreBaseline
from src.models.baseline_iforest import IsolationForestBaseline


def _load_split(processed_dir: str, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(os.path.join(processed_dir, "injected", f"{name}.parquet"))
    lab = pd.read_parquet(os.path.join(processed_dir, "injected", f"{name}_labels.parquet"))
    return df, lab


def _select_feature_cols(df: pd.DataFrame, feat_cfg: dict) -> list[str]:
    # run_milof_tcn.py와 동일한 규칙 (가능한 한 일관성 유지)
    base = list(feat_cfg["use_cols"])
    cols = [c for c in base if c in df.columns]

    if feat_cfg.get("add_calendar", False):
        for c in ["hour", "dow", "is_weekend"]:
            if c in df.columns:
                cols.append(c)

    roll_cfg = feat_cfg.get("rolling", {})
    if roll_cfg.get("enable", False):
        for w in roll_cfg.get("windows_hours", []):
            for base_c in ["total", "fast", "slow", "ratio_fast"]:
                m = f"{base_c}_roll_mean_{w}"
                s = f"{base_c}_roll_std_{w}"
                if m in df.columns:
                    cols.append(m)
                if s in df.columns:
                    cols.append(s)

    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _pick_threshold(val_scores: np.ndarray, th_cfg: dict) -> float:
    method = str(th_cfg["method"])
    if method == "quantile":
        return threshold_by_quantile(val_scores, float(th_cfg["q"]))
    if method == "mean_std":
        return threshold_by_mean_std(val_scores, float(th_cfg.get("k", 3.0)))
    raise ValueError(f"Unknown threshold method: {method}")


def run_stats(cfg: dict, train_df, val_df, test_df, val_lab, test_lab) -> tuple[dict, pd.DataFrame]:
    model = ContextZScoreBaseline(target_col="total").fit(train_df)
    val_point = model.score(val_df)
    test_point = model.score(test_df)

    win_len = int(cfg["windows"]["length"])
    stride = int(cfg["windows"]["stride"])

    tva, val_scores = aggregate_point_scores_to_windows(val_df["ts"].to_numpy(), val_point, win_len, stride, agg="max")
    tte, test_scores = aggregate_point_scores_to_windows(test_df["ts"].to_numpy(), test_point, win_len, stride, agg="max")

    # window labels은 run_milof_tcn과 동일 규칙(“any”)을 쓰는 게 맞다.
    # 여기선 이미 inject_anomalies가 point label을 제공하므로, 윈도우 라벨을 다시 만든다.
    def window_labels(lab_df: pd.DataFrame) -> np.ndarray:
        y = lab_df["is_anomaly"].to_numpy(dtype=int)
        ys = []
        for start in range(0, len(y) - win_len + 1, stride):
            end = start + win_len
            ys.append(int(y[start:end].max() > 0))
        return np.asarray(ys, dtype=int)

    yva = window_labels(val_lab)
    yte = window_labels(test_lab)

    thr = _pick_threshold(val_scores, cfg["threshold"])

    metrics, curves = evaluate_window_scores(
        ts_end=tte,
        y_true=yte,
        scores=test_scores,
        threshold=thr,
        group_id=None,  # baseline은 latency를 꼭 넣고 싶으면 keep_idx 기반으로 group_id 매핑을 추가하면 됨
    )
    metrics["model"] = "stats_context_z"
    return metrics, curves


def run_iforest(cfg: dict, train_df, val_df, test_df, val_lab, test_lab, feat_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    model = IsolationForestBaseline(
        feature_cols=feat_cols,
        n_estimators=200,
        contamination="auto",
        random_state=int(cfg["project"]["seed"]),
    ).fit(train_df)

    val_point = model.score(val_df)
    test_point = model.score(test_df)

    win_len = int(cfg["windows"]["length"])
    stride = int(cfg["windows"]["stride"])

    tva, val_scores = aggregate_point_scores_to_windows(val_df["ts"].to_numpy(), val_point, win_len, stride, agg="max")
    tte, test_scores = aggregate_point_scores_to_windows(test_df["ts"].to_numpy(), test_point, win_len, stride, agg="max")

    def window_labels(lab_df: pd.DataFrame) -> np.ndarray:
        y = lab_df["is_anomaly"].to_numpy(dtype=int)
        ys = []
        for start in range(0, len(y) - win_len + 1, stride):
            end = start + win_len
            ys.append(int(y[start:end].max() > 0))
        return np.asarray(ys, dtype=int)

    yva = window_labels(val_lab)
    yte = window_labels(test_lab)

    thr = _pick_threshold(val_scores, cfg["threshold"])

    metrics, curves = evaluate_window_scores(
        ts_end=tte,
        y_true=yte,
        scores=test_scores,
        threshold=thr,
        group_id=None,
    )
    metrics["model"] = "iforest_point"
    return metrics, curves


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg["project"]["seed"]))

    processed = cfg["paths"]["processed_dir"]
    train_df, train_lab = _load_split(processed, "train")
    val_df, val_lab = _load_split(processed, "val")
    test_df, test_lab = _load_split(processed, "test")

    feat_cols = _select_feature_cols(train_df, cfg["features"])

    # 1) stats baseline
    ctx1 = create_run_context(cfg["paths"]["results_dir"], prefix="baseline_stats")
    save_config(cfg, ctx1)
    m1, c1 = run_stats(cfg, train_df, val_df, test_df, val_lab, test_lab)
    save_metrics(m1, ctx1)
    save_curves_csv(c1, ctx1)
    save_score_plot(c1, os.path.join(ctx1.fig_dir, "score.png"), title="Stats baseline score (test)")
    save_pred_plot(c1, os.path.join(ctx1.fig_dir, "pred.png"), title="Stats baseline pred (test)")
    print("[DONE] stats:", ctx1.run_dir, m1)

    # 2) iforest baseline
    ctx2 = create_run_context(cfg["paths"]["results_dir"], prefix="baseline_iforest")
    save_config(cfg, ctx2)
    m2, c2 = run_iforest(cfg, train_df, val_df, test_df, val_lab, test_lab, feat_cols=feat_cols)
    save_metrics(m2, ctx2)
    save_curves_csv(c2, ctx2)
    save_score_plot(c2, os.path.join(ctx2.fig_dir, "score.png"), title="IForest baseline score (test)")
    save_pred_plot(c2, os.path.join(ctx2.fig_dir, "pred.png"), title="IForest baseline pred (test)")
    print("[DONE] iforest:", ctx2.run_dir, m2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
