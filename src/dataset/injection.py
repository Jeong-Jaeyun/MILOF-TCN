from __future__ import annotations
import numpy as np
import pandas as pd

def _choice(rng, arr):
    return arr[rng.integers(0, len(arr))]

def _apply_missing_block(out: pd.DataFrame, idx: list[int], cols: list[str], mode: str):
    # mode: "ffill", "bfill", "zero", "interp"
    for c in cols:
        out.loc[idx, c] = np.nan
    if mode == "zero":
        out[cols] = out[cols].fillna(0.0)
    elif mode == "ffill":
        out[cols] = out[cols].ffill()
    elif mode == "bfill":
        out[cols] = out[cols].bfill()
    elif mode == "interp":
        out[cols] = out[cols].interpolate(limit_direction="both")
    else:
        raise ValueError(f"Unknown missing mode: {mode}")


def inject_anomalies(df: pd.DataFrame, cfg: dict, target_col: str = "total") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_inj: injected dataframe
      labels: columns = [ts, is_anomaly, anomaly_type, group_id]
    """
    rng = np.random.default_rng(int(cfg["seed"]))
    rate = float(cfg["rate"])
    types = cfg["types"]

    out = df.copy().reset_index(drop=True)
    n = len(out)
    labels = pd.DataFrame({
        "ts": out["ts"],
        "is_anomaly": np.zeros(n, dtype=int),
        "anomaly_type": ["none"] * n,
        "group_id": [-1] * n
    })

    # 각 타입의 확률을 합 1로 정규화
    type_names = list(types.keys())
    probs = np.array([types[t].get("prob", 0.0) for t in type_names], dtype=float)
    probs = probs / probs.sum()

    budget = int(np.floor(n * rate))
    if budget <= 0:
        return out, labels

    # anomaly event 단위로 주입 (collective는 여러 포인트를 한 그룹으로 묶음)
    group_id = 0
    used = set()

    def mark(idx_list, a_type: str, gid: int):
        labels.loc[idx_list, "is_anomaly"] = 1
        labels.loc[idx_list, "anomaly_type"] = a_type
        labels.loc[idx_list, "group_id"] = gid

    max_attempts = n * 10  # avoid infinite loops when overlaps exhaust budget
    attempts = 0

    while len(used) < budget and attempts < max_attempts:
        attempts += 1
        remaining = budget - len(used)
        if remaining <= 0:
            break
        a_type = rng.choice(type_names, p=probs)

        if a_type in ("point_spike", "point_drop"):
            i = int(rng.integers(0, n))
            if i in used:
                continue
            k_list = types[a_type]["strength"]
            k = float(_choice(rng, k_list))
            if a_type == "point_spike":
                out.loc[i, target_col] = out.loc[i, target_col] * k
            else:
                out.loc[i, target_col] = out.loc[i, target_col] * k
            used.add(i)
            mark([i], a_type, group_id)
            group_id += 1

        elif a_type == "collective_shift":
            dur = min(int(_choice(rng, types[a_type]["duration_hours"])), remaining)
            if dur <= 0:
                break
            start = int(rng.integers(0, max(1, n - dur)))
            idx = [j for j in range(start, start + dur) if j not in used][:remaining]
            if not idx:
                continue
            add_ratio = float(_choice(rng, types[a_type]["base_add_ratio"]))
            # base 증가: total에 (평균 기반) offset을 더함
            base = float(out[target_col].iloc[max(0, start-24):start].mean() if start > 0 else out[target_col].mean())
            out.loc[idx, target_col] = out.loc[idx, target_col] + base * add_ratio
            for i in idx: used.add(i)
            mark(idx, a_type, group_id)
            group_id += 1

        elif a_type == "peak_shift":
            dur = min(int(_choice(rng, types[a_type]["duration_hours"])), remaining)
            if dur <= 0:
                break
            shift = int(_choice(rng, types[a_type]["shift_hours"]))
            start = int(rng.integers(0, max(1, n - dur)))
            idx = [j for j in range(start, start + dur) if j not in used][:remaining]
            if not idx:
                continue
            # 하루(24h) 패턴을 shift해서 붙여넣기: 해당 구간을 24h 블록으로 가정
            # 단순화: 구간 내 값을 shift된 인덱스에서 가져오되 범위 밖은 원값 유지
            arr = out[target_col].to_numpy().copy()
            for j in idx:
                src = j + shift
                if 0 <= src < n:
                    out.loc[j, target_col] = arr[src]
            for i in idx: used.add(i)
            mark(idx, a_type, group_id)
            group_id += 1
        
        elif a_type == "drift_linear":
            # linear upward/downward drift over duration
            dur = min(int(_choice(rng, types[a_type]["duration_hours"])), remaining)
            if dur <= 0:
                break
            start = int(rng.integers(0, max(1, n - dur)))
            idx = [j for j in range(start, start + dur) if j not in used][:remaining]
            if not idx:
                continue
            # ratio range e.g., [0.05, 0.2]
            end_ratio = float(_choice(rng, types[a_type]["end_ratio"]))
            direction = str(_choice(rng, types[a_type].get("direction", ["up"])))
            sign = 1.0 if direction == "up" else -1.0

            base = float(out[target_col].iloc[max(0, start-24):start].mean() if start > 0 else out[target_col].mean())
            # add linearly increasing offset
            for t, ii in enumerate(idx):
                frac = (t + 1) / dur
                out.loc[ii, target_col] = out.loc[ii, target_col] + sign * base * end_ratio * frac

            for i in idx: used.add(i)
            mark(idx, a_type, group_id); group_id += 1

        elif a_type == "missing_blocks":
            # missing block then imputation
            dur = min(int(_choice(rng, types[a_type]["duration_hours"])), remaining)
            if dur <= 0:
                break
            start = int(rng.integers(0, max(1, n - dur)))
            idx = [j for j in range(start, start + dur) if j not in used][:remaining]
            if not idx:
                continue

            cols = list(types[a_type].get("cols", [target_col]))
            mode = str(_choice(rng, types[a_type].get("impute", ["ffill"])))
            _apply_missing_block(out, idx, cols, mode)

            for i in idx: used.add(i)
            mark(idx, a_type, group_id); group_id += 1

        elif a_type == "gaussian_noise":
            # add observation noise to target_col (and optionally other cols)
            dur = min(int(_choice(rng, types[a_type]["duration_hours"])), remaining)
            if dur <= 0:
                break
            start = int(rng.integers(0, max(1, n - dur)))
            idx = [j for j in range(start, start + dur) if j not in used][:remaining]
            if not idx:
                continue

            sigma_ratio = float(_choice(rng, types[a_type]["sigma_ratio"]))
            cols = list(types[a_type].get("cols", [target_col]))

            for c in cols:
                base = float(out[c].iloc[max(0, start-24):start].std(ddof=0) if start > 0 else out[c].std(ddof=0))
                noise = rng.normal(loc=0.0, scale=max(1e-9, base * sigma_ratio), size=len(idx))
                out.loc[idx, c] = out.loc[idx, c].to_numpy(dtype=float) + noise

            for i in idx: used.add(i)
            mark(idx, a_type, group_id); group_id += 1


        else:
            # 확장 대비
            continue

        if len(used) >= budget:
            break

    # fast/slow 일관성 유지
    if "fast" in out.columns and "slow" in out.columns and "ratio_fast" in out.columns:
        r = out["ratio_fast"].clip(0, 1)
        out["fast"] = out[target_col] * r
        out["slow"] = out[target_col] * (1 - r)

    return out, labels
