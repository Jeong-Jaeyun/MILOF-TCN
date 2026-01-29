from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from datetime import datetime

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_run_dir(results_root: str, prefix: str = "exp") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"{prefix}_{ts}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "figures"))
    ensure_dir(os.path.join(run_dir, "artifacts"))
    return run_dir

def dump_yaml(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
