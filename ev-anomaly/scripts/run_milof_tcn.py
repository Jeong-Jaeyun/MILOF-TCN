from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from src.common.io import load_yaml, make_run_dir, dump_yaml
from src.common.seed import set_seed
from src.dataset.windows import make_windows, make_windows_with_edge_gate
from src.training.train_tcn import train_tcn_ae, score_tcn_ae
from src.training.thresholds import threshold_by_quantile, threshold_by_mean_std
from src.evaluation.evaluate import evaluate_window_scores
from src.common.plot import save_score_plot, save_pred_plot
from src.models.edge_milof import MiLOFEdgeDetector


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


def _fit_edge_threshold(scores: np.ndarray, method: str, q: float = 0.99, k: float = 3.0) -> float:
    if method == "quantile":
        return float(np.quantile(scores, float(q)))
    if method == "mean_std":
        return float(scores.mean() + float(k) * scores.std(ddof=0))
    raise ValueError(f"Unknown edge threshold method: {method}")


def _edge_gate_mask(df: pd.DataFrame, feat_cols: list[str], edge_cfg: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Returns:
      pass_mask: boolean array length=len(df), True means pass to Fog
      info: dict with edge stats
    """
    enabled = bool(edge_cfg.get("enabled", True))
    if not enabled:
        return np.ones(len(df), dtype=bool), {
            "edge_enabled": False,
            "edge_tau": None,
            "edge_pass_rate": 1.0,
            "edge_clusters": None,
        }

    X = df[feat_cols].to_numpy(dtype=float)

    # standardize using train stats (robustness)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma <= 0, 1.0, sigma)

    det = MiLOFEdgeDetector(
        k=int(edge_cfg.get("k", 10)),
        radius_factor=float(edge_cfg.get("radius_factor", 2.5)),
        max_clusters=int(edge_cfg.get("max_clusters", 256)),
        min_clusters_for_scoring=int(edge_cfg.get("min_clusters_for_scoring", 20)),
        decay_older_than=int(edge_cfg.get("decay_older_than", 24 * 14)),
    )
    det.set_standardizer(mu, sigma)

    # warmup + collect scores on the fly
    scores = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        x = X[i]
        s = det.score(x)
        scores[i] = s
        det.partial_fit(x)

    # tau from distribution
    tau_method = str(edge_cfg.get("threshold_method", "quantile"))
    tau_q = float(edge_cfg.get("q", 0.99))
    tau_k = float(edge_cfg.get("k_std", 3.0))
    tau = _fit_edge_threshold(scores, tau_method, q=tau_q, k=tau_k)

    # pass policy:
    #   if edge flags anomaly -> it is handled at Edge (do NOT forward to Fog)
    #   else forward to Fog for deep pattern detection
    is_edge_anom = (scores >= tau)
    pass_mask = ~is_edge_anom

    info = {
        "edge_enabled": True,
        "edge_tau": float(tau),
        "edge_pass_rate": float(pass_mask.mean()) if len(pass_mask) else None,
        "edge_clusters": int(len(det.clusters)),
    }
    return pass_mask, info, scores


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg["project"]["seed"]))

    run_dir = make_run_dir(cfg["paths"]["results_dir"], prefix="milof_tcn")
    dump_yaml(cfg, os.path.join(run_dir, "config_resolved.yaml"))

    processed = cfg["paths"]["processed_dir"]
    train_df, train_lab = _load_split(processed, "train")
    val_df, val_lab = _load_split(processed, "val")
    test_df, test_lab = _load_split(processed, "test")

    feat_cols = _select_feature_cols(train_df, cfg["features"])
    print("[INFO] feature_cols:", feat_cols)

    win_len = int(cfg["windows"]["length"])
    stride = int(cfg["windows"]["stride"])
    label_mode = str(cfg["windows"]["label_mode"])

    # --------------------------
    # Edge gating (train/val/test 각각에 mask 생성)
    # --------------------------
    edge_cfg = cfg.get("edge", {
        "enabled": True,
        "k": 10,
        "radius_factor": 2.5,
        "max_clusters": 256,
        "min_clusters_for_scoring": 20,
        "decay_older_than": 24 * 14,
        "threshold_method": "quantile",
        "q": 0.99,
    })

    tr_mask, tr_edge_info, tr_edge_scores = _edge_gate_mask(train_df, feat_cols, edge_cfg)
    va_mask, va_edge_info, va_edge_scores = _edge_gate_mask(val_df, feat_cols, edge_cfg)
    te_mask, te_edge_info, te_edge_scores = _edge_gate_mask(test_df, feat_cols, edge_cfg)

    # --------------------------
    # Edge flags (point-wise)
    # --------------------------
    tr_edge_flags = (tr_edge_scores >= float(tr_edge_info["edge_tau"]))
    va_edge_flags = (va_edge_scores >= float(va_edge_info["edge_tau"]))
    te_edge_flags = (te_edge_scores >= float(te_edge_info["edge_tau"]))

    # --------------------------
    # Fog window gate (time-axis preserved)
    #   - build windows on original timeline
    #   - drop windows if too many edge anomalies inside
    # --------------------------
    edge_win_cfg = cfg.get("edge_window", {"max_edge_ratio": 0.2})
    max_edge_ratio = float(edge_win_cfg.get("max_edge_ratio", 0.2))

    Xtr, ytr, ttr, _ = make_windows_with_edge_gate(
        train_df, train_lab, tr_edge_flags,
        feat_cols, win_len, stride, label_mode,
        max_edge_ratio=max_edge_ratio
    )
    Xva, yva, tva, _ = make_windows_with_edge_gate(
        val_df, val_lab, va_edge_flags,
        feat_cols, win_len, stride, label_mode,
        max_edge_ratio=max_edge_ratio
    )
    Xte, yte, tte, keep_idx = make_windows_with_edge_gate(
        test_df, test_lab, te_edge_flags,
        feat_cols, win_len, stride, label_mode,
        max_edge_ratio=max_edge_ratio
    )

    def window_group_id_from_keepidx(labels_df: pd.DataFrame, keep_idx: list[tuple[int,int]]) -> np.ndarray:
        gid_arr = labels_df["group_id"].to_numpy(dtype=int)
        out = []
        for (s, e) in keep_idx:
            part = gid_arr[s:e+1]
            part = part[part >= 0]
            out.append(int(part[0]) if part.size else -1)
        return np.asarray(out, dtype=int)


    gte = window_group_id_from_keepidx(test_lab, keep_idx) if len(keep_idx) else np.array([], dtype=int)

    # --------------------------
    # Train Fog: TCN-AE
    # --------------------------
    device = "cuda:0"
    model_path = os.path.join(run_dir, "artifacts", "tcn_ae.pt")

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise RuntimeError(
            "Edge gating filtered too much data, resulting in empty windows. "
            "Decrease edge threshold strictness (q higher pass rate) or disable edge for now."
        )

    tr_res = train_tcn_ae(
        X_train=Xtr,
        X_val=Xva,
        n_features=int(Xtr.shape[-1]),
        channels=list(cfg["tcn"]["channels"]),
        kernel_size=int(cfg["tcn"]["kernel_size"]),
        dropout=float(cfg["tcn"]["dropout"]),
        lr=float(cfg["tcn"]["lr"]),
        epochs=int(cfg["tcn"]["epochs"]),
        batch_size=int(cfg["tcn"]["batch_size"]),
        device=device,
        save_path=model_path,
    )

    # Score
    val_scores = score_tcn_ae(tr_res.model, Xva, batch_size=int(cfg["tcn"]["batch_size"]), device=device)
    test_scores = score_tcn_ae(tr_res.model, Xte, batch_size=int(cfg["tcn"]["batch_size"]), device=device)

    # Threshold (Fog)
    th_cfg = cfg["threshold"]
    method = str(th_cfg["method"])
    if method == "quantile":
        thr = threshold_by_quantile(val_scores, float(th_cfg["q"]))
    elif method == "mean_std":
        thr = threshold_by_mean_std(val_scores, float(th_cfg.get("k", 3.0)))
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    # Evaluate
    metrics, curves = evaluate_window_scores(
        ts_end=tte,
        y_true=yte,
        scores=test_scores,
        threshold=thr,
        group_id=gte,
    )

    # Attach Edge stats
    metrics.update({
        "edge_enabled": True,
        "edge_pass_rate_train": tr_edge_info.get("edge_pass_rate"),
        "edge_pass_rate_val": va_edge_info.get("edge_pass_rate"),
        "edge_pass_rate_test": te_edge_info.get("edge_pass_rate"),
        "edge_tau_train": tr_edge_info.get("edge_tau"),
        "edge_tau_val": va_edge_info.get("edge_tau"),
        "edge_tau_test": te_edge_info.get("edge_tau"),
        "edge_clusters_train": tr_edge_info.get("edge_clusters"),
        "edge_clusters_val": va_edge_info.get("edge_clusters"),
        "edge_clusters_test": te_edge_info.get("edge_clusters"),
    })

    # Save outputs
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    curves.to_csv(os.path.join(run_dir, "curves.csv"), index=False, encoding="utf-8-sig")

    save_score_plot(curves, os.path.join(run_dir, "figures", "score.png"), title="MiLOF->TCN-AE Score (test)")
    save_pred_plot(curves, os.path.join(run_dir, "figures", "pred.png"), title="MiLOF->TCN-AE Predictions (test)")

    print("[DONE] run_dir:", run_dir)
    print("[METRICS]", metrics)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
