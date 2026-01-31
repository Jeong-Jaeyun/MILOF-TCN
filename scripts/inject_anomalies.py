import os
import sys
import pandas as pd

# allow running as a script without PYTHONPATH set
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.io import load_yaml, ensure_dir
from src.common.seed import set_seed
from src.dataset.split import make_splits
from src.dataset.injection import inject_anomalies

def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg["project"]["seed"]))

    processed = cfg["paths"]["processed_dir"]
    df = pd.read_parquet(os.path.join(processed, "timeseries_long.parquet"))

    splits = make_splits(df, cfg["split"])
    target_split = cfg["injection"]["target"]

    ensure_dir(os.path.join(processed, "injected"))

    # 기본 labels(없으면 전부 0)
    base_labels = {}
    for k, sdf in splits.items():
        base_labels[k] = pd.DataFrame({
            "ts": sdf["ts"].values,
            "is_anomaly": 0,
            "anomaly_type": "none",
            "group_id": -1
        })

    if cfg["injection"]["enabled"]:
        inj_df, inj_labels = inject_anomalies(splits[target_split], cfg["injection"], target_col="total")
        splits[target_split] = inj_df
        base_labels[target_split] = inj_labels

    # 저장
    for k in ["train", "val", "test"]:
        splits[k].to_parquet(os.path.join(processed, "injected", f"{k}.parquet"), index=False)
        base_labels[k].to_parquet(os.path.join(processed, "injected", f"{k}_labels.parquet"), index=False)

    print("[OK] splits saved under:", os.path.join(processed, "injected"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
