from __future__ import annotations
import os
import matplotlib.pyplot as plt
import pandas as pd


def save_score_plot(curves: pd.DataFrame, out_path: str, title: str = "Anomaly Score"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = curves.copy()
    df["ts_end"] = pd.to_datetime(df["ts_end"])
    df = df.sort_values("ts_end")

    plt.figure()
    plt.plot(df["ts_end"], df["score"])
    # true anomaly highlight (binary)
    if "y_true" in df.columns:
        ya = df["y_true"].to_numpy()
        # mark anomaly windows
        plt.scatter(df.loc[ya == 1, "ts_end"], df.loc[ya == 1, "score"], s=10)

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pred_plot(curves: pd.DataFrame, out_path: str, title: str = "Predictions"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = curves.copy()
    df["ts_end"] = pd.to_datetime(df["ts_end"])
    df = df.sort_values("ts_end")

    plt.figure()
    plt.plot(df["ts_end"], df["y_true"], label="y_true")
    plt.plot(df["ts_end"], df["y_pred"], label="y_pred")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
