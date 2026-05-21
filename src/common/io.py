from __future__ import annotations
import os
import re
import yaml
from dataclasses import dataclass
from datetime import datetime

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _sanitize_tag(tag: str | None) -> str:
    if not tag:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(tag)).strip("-")

def make_run_dir(results_root: str, prefix: str = "exp", tag: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = _sanitize_tag(tag)
    name = f"{prefix}_{ts}" if not tag_part else f"{prefix}_{tag_part}_{ts}"
    run_dir = os.path.join(results_root, name)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "figures"))
    ensure_dir(os.path.join(run_dir, "artifacts"))
    return run_dir

def dump_yaml(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
