from __future__ import annotations
import os
import json
import itertools
import subprocess
import pandas as pd

from src.common.io import load_yaml, ensure_dir


def _set_nested(cfg: dict, key_path: str, value):
    keys = key_path.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def main(base_cfg_path: str):
    base = load_yaml(base_cfg_path)
    results_root = base["paths"]["results_dir"]
    ensure_dir(results_root)

    sweep_dir = os.path.join(results_root, "sweep_runs")
    ensure_dir(sweep_dir)

    # grid 설정
    grid = {
        "edge.q": [0.99, 0.995, 0.999],
        "edge_window.max_edge_ratio": [0.1, 0.2, 0.3],
        "windows.length": [72, 168],
        "threshold.q": [0.99, 0.995],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rows = []
    for i, combo in enumerate(combos, 1):
        cfg = load_yaml(base_cfg_path)
        tag_parts = []
        for k, v in zip(keys, combo):
            _set_nested(cfg, k, v)
            tag_parts.append(f"{k.replace('.','_')}={v}")
        tag = "__".join(tag_parts)

        cfg_path = os.path.join(sweep_dir, f"cfg_{i:03d}.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        # run_milof_tcn.py가 results_dir 아래에 exp 폴더 생성
        cmd = ["python", "scripts/run_milof_tcn.py", "--config", cfg_path]
        print(f"[SWEEP] {i}/{len(combos)}  {tag}")
        subprocess.run(cmd, check=True)

        # 가장 최근 생성된 milof_tcn_ 사용
        dirs = [d for d in os.listdir(results_root) if d.startswith("milof_tcn_")]
        dirs = sorted(dirs, key=lambda x: os.path.getmtime(os.path.join(results_root, x)))
        run_dir = os.path.join(results_root, dirs[-1])

        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        row = {"tag": tag, "run_dir": run_dir}
        # configs
        for k, v in zip(keys, combo):
            row[k] = v
        # key metrics
        for kk in [
            "f1", "pr_auc", "roc_auc", "mcc", "balanced_acc",
            "avg_delay_hours", "false_alarms_per_day",
            "edge_pass_rate_test", "fog_window_keep_rate_test"
        ]:
            row[kk] = m.get(kk)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(results_root, "sweep_summary.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[DONE] sweep summary:", out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
