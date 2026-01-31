from __future__ import annotations
import os
import json
import matplotlib.pyplot as plt
import pandas as pd


def collect_metrics(results_root: str) -> pd.DataFrame:
    rows = []
    for d in os.listdir(results_root):
        run_dir = os.path.join(results_root, d)
        if not os.path.isdir(run_dir):
            continue
        mpath = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath, "r", encoding="utf-8") as f:
            m = json.load(f)

        rows.append({
            "run": d,
            "model": m.get("model", "milof_tcn" if d.startswith("milof_tcn_") else "unknown"),
            "f1": m.get("f1"),
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
            "false_alarms_per_day": m.get("false_alarms_per_day"),
            "avg_delay_hours": m.get("avg_delay_hours"),
            "edge_pass_rate_test": m.get("edge_pass_rate_test"),
            "fog_window_keep_rate_test": m.get("fog_window_keep_rate_test"),
        })
    return pd.DataFrame(rows)


def main(results_root: str):
    df = collect_metrics(results_root)
    if df.empty:
        raise RuntimeError("No metrics.json found under results root.")

    # Fog load proxy: window_keep_rate_test (lower is better efficiency)
    x = df["fog_window_keep_rate_test"].astype(float)
    y = df["f1"].astype(float)

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Fog window keep rate (test)")
    plt.ylabel("F1 (test)")
    plt.title("Trade-off: F1 vs Fog load")
    plt.tight_layout()

    out_path = os.path.join(results_root, "tradeoff_f1_vs_fogload.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[DONE]", out_path)

    # also save table
    out_csv = os.path.join(results_root, "tradeoff_table.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[DONE]", out_csv)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    args = ap.parse_args()
    main(args.results_root)
