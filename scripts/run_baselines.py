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


    yva, keep_idx_va = window_labels_and_keepidx(val_lab, win_len, stride)
    yte, keep_idx_te = window_labels_and_keepidx(test_lab, win_len, stride)
    gte = window_group_id_from_keepidx(test_lab, keep_idx_te)


    thr = _pick_threshold(val_scores, cfg["threshold"])

    metrics, curves = evaluate_window_scores(
        ts_end=tte,
        y_true=yte,
        scores=test_scores,
        threshold=thr,
        group_id=gte,  
    )
    metrics["model"] = "stats_context_z"
    return metrics, curves

def window_labels_and_keepidx(lab_df: pd.DataFrame, win_len: int, stride: int):
    y = lab_df["is_anomaly"].to_numpy(dtype=int)


    y_out, keep_idx = [], []
    for start in range(0, len(y) - win_len + 1, stride):
        end = start + win_len
        y_out.append(int(y[start:end].max() > 0))
        keep_idx.append((start, end - 1))
    return np.asarray(y_out, dtype=int), keep_idx

def window_group_id_from_keepidx(labels_df: pd.DataFrame, keep_idx):
    gid_arr = labels_df["group_id"].to_numpy(dtype=int)
    out = []
    for (s, e) in keep_idx:
        part = gid_arr[s:e+1]
        part = part[part >= 0]
        out.append(int(part[0]) if part.size else -1)
    return np.asarray(out, dtype=int)

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

    yva, keep_idx_va = window_labels_and_keepidx(val_lab, win_len, stride)
    yte, keep_idx_te = window_labels_and_keepidx(test_lab, win_len, stride)

    gte = window_group_id_from_keepidx(test_lab, keep_idx_te)


    thr = _pick_threshold(val_scores, cfg["threshold"])

    metrics, curves = evaluate_window_scores(
        ts_end=tte,
        y_true=yte,
        scores=test_scores,
        threshold=thr,
        group_id=gte,
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
