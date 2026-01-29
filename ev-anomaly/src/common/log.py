from __future__ import annotations
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.common.io import ensure_dir, dump_yaml


@dataclass
class RunContext:
    run_dir: str
    fig_dir: str
    art_dir: str


def create_run_context(results_root: str, prefix: str) -> RunContext:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"{prefix}_{ts}")
    fig_dir = os.path.join(run_dir, "figures")
    art_dir = os.path.join(run_dir, "artifacts")
    ensure_dir(run_dir)
    ensure_dir(fig_dir)
    ensure_dir(art_dir)
    return RunContext(run_dir=run_dir, fig_dir=fig_dir, art_dir=art_dir)


def save_config(cfg: dict, ctx: RunContext) -> None:
    dump_yaml(cfg, os.path.join(ctx.run_dir, "config_resolved.yaml"))


def save_metrics(metrics: dict, ctx: RunContext, filename: str = "metrics.json") -> None:
    path = os.path.join(ctx.run_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_curves_csv(curves_df, ctx: RunContext, filename: str = "curves.csv") -> None:
    import pandas as pd
    assert isinstance(curves_df, pd.DataFrame)
    curves_df.to_csv(os.path.join(ctx.run_dir, filename), index=False, encoding="utf-8-sig")


def log_kv(prefix: str, d: dict[str, Any]) -> None:
    items = ", ".join([f"{k}={v}" for k, v in d.items()])
    print(f"[{prefix}] {items}")
