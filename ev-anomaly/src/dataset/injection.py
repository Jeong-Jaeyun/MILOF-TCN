from __future__ import annotations
import numpy as np
import pandas as pd

def _choice(rng, arr):
    return arr[rng.integers(0, len(arr))]

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

    while len(used) < budget:
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
            dur = int(_choice(rng, types[a_type]["duration_hours"]))
            start = int(rng.integers(0, max(1, n - dur)))
            idx = list(range(start, start + dur))
            if any(i in used for i in idx):
                continue
            add_ratio = float(_choice(rng, types[a_type]["base_add_ratio"]))
            # base 증가: total에 (평균 기반) offset을 더함
            base = float(out[target_col].iloc[max(0, start-24):start].mean() if start > 0 else out[target_col].mean())
            out.loc[idx, target_col] = out.loc[idx, target_col] + base * add_ratio
            for i in idx: used.add(i)
            mark(idx, a_type, group_id)
            group_id += 1

        elif a_type == "peak_shift":
            dur = int(_choice(rng, types[a_type]["duration_hours"]))
            shift = int(_choice(rng, types[a_type]["shift_hours"]))
            start = int(rng.integers(0, max(1, n - dur)))
            idx = list(range(start, start + dur))
            if any(i in used for i in idx):
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

        else:
            # 확장 대비
            continue

        if len(used) >= budget:
            break

    # fast/slow 일관성 유지: total만 바꾸면 ratio가 깨질 수 있으니, 단순 비례로 분해
    if "fast" in out.columns and "slow" in out.columns and "ratio_fast" in out.columns:
        r = out["ratio_fast"].clip(0, 1)
        out["fast"] = out[target_col] * r
        out["slow"] = out[target_col] * (1 - r)

    return out, labels
