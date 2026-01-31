from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt


def pick_representative_groups(curves: pd.DataFrame, labels: pd.DataFrame, topk: int = 1):
    """
    Pick representative group_id per anomaly_type.
    """
    out = {}
    for a_type in labels["anomaly_type"].unique():
        if a_type == "none":
            continue
        gids = labels.loc[labels["anomaly_type"] == a_type, "group_id"].unique()
        if len(gids) == 0:
            continue
        out[a_type] = gids[:topk]
    return out


def plot_group(
    df: pd.DataFrame,
    curves: pd.DataFrame,
    gid: int,
    out_path: str,
    title: str,
):
    df = df.copy()
    curves = curves.copy()

    df["ts"] = pd.to_datetime(df["ts"])
    curves["ts_end"] = pd.to_datetime(curves["ts_end"])

    # select region
    idx = df.index[df["group_id"] == gid]
    if len(idx) == 0:
        return
    s, e = max(0, idx.min() - 48), min(len(df) - 1, idx.max() + 48)

    sub = df.iloc[s:e+1]
    sub_curves = curves[(curves["ts_end"] >= sub["ts"].iloc[0]) &
                         (curves["ts_end"] <= sub["ts"].iloc[-1])]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(sub["ts"], sub["total"], label="Load", color="black")
    ax1.set_ylabel("Load")

    ax2 = ax1.twinx()
    ax2.plot(sub_curves["ts_end"], sub_curves["score"], label="Score", color="tab:red")
    ax2.axhline(sub_curves["score"].median(), linestyle="--", color="gray", alpha=0.5)
    ax2.set_ylabel("Anomaly score")

    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(run_dir: str, processed_dir: str):
    curves = pd.read_csv(os.path.join(run_dir, "curves.csv"))
    df = pd.read_parquet(os.path.join(processed_dir, "injected", "test.parquet"))
    lab = pd.read_parquet(os.path.join(processed_dir, "injected", "test_labels.parquet"))

    rep = pick_representative_groups(curves, lab, topk=1)
    fig_dir = os.path.join(run_dir, "figures", "case_studies")
    os.makedirs(fig_dir, exist_ok=True)

    for a_type, gids in rep.items():
        for gid in gids:
            out = os.path.join(fig_dir, f"case_{a_type}_gid{gid}.png")
            plot_group(
                df, curves, gid, out,
                title=f"Case study: {a_type} anomaly (group {gid})"
            )
            print("[DONE]", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--processed_dir", required=True)
    args = ap.parse_args()
    main(args.run_dir, args.processed_dir)
