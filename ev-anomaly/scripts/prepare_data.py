import os
import pandas as pd
from src.common.io import load_yaml, ensure_dir
from src.common.seed import set_seed
from src.dataset.kepco_preprocess import read_kepco_raw, wide_to_long, pivot_modes
from src.dataset.features import add_calendar_features, add_rolling

def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg["project"]["seed"]))

    ensure_dir(cfg["paths"]["processed_dir"])

    raw = read_kepco_raw(
        cfg["paths"]["raw_csv"],
        cfg["data"]["date_col"],
        cfg["data"]["mode_col"],
        cfg["data"]["hour_cols"],
    )
    long_df = wide_to_long(raw, cfg["data"]["date_col"], cfg["data"]["mode_col"], cfg["data"]["hour_cols"])
    ts_df = pivot_modes(long_df, cfg["data"]["mode_col"], cfg["data"]["mode_fast_values"], cfg["data"]["mode_slow_values"])

    if cfg["features"]["add_calendar"]:
        ts_df = add_calendar_features(ts_df)

    if cfg["features"]["rolling"]["enable"]:
        w = cfg["features"]["rolling"]["windows_hours"]
        cols = ["total", "fast", "slow", "ratio_fast"]
        ts_df = add_rolling(ts_df, w, cols)

    out_path = os.path.join(cfg["paths"]["processed_dir"], "timeseries_long.parquet")
    ts_df.to_parquet(out_path, index=False)
    print(f"[OK] saved: {out_path}  rows={len(ts_df)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
