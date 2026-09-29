"""按位置和唯一指纹构造 MetaLoc 式 Support/Query episode。"""
from collections import defaultdict

import numpy as np
import torch


def prepare_floor(raw, device):
    # 同一位置的完全相同 RSSI/坐标只保留一个候选索引。
    by_position = defaultdict(dict)
    for i, (xy, group) in enumerate(zip(raw["xy"], raw["groups"])):
        by_position[tuple(xy)].setdefault(group, i)
    groups = {xy: np.array(list(items.values()), dtype=np.int64)
              for xy, items in sorted(by_position.items())}
    rss = raw["rssi"].astype(np.float32)
    rss[rss == 100] = -110
    x = torch.tensor(np.clip((rss + 110) / 110, 0, 1), device=device)
    # UJI 坐标含大数值；先在 float64 中减原点，再转网络的 float32 标签。
    xy = torch.tensor(raw["xy"], dtype=torch.float64, device=device)
    return {"x": x, "xy": xy, "positions": groups, "row_ids": raw["row_ids"],
            "group_ids": raw["groups"], "position_keys": [tuple(p) for p in raw["xy"]]}


def sample_episode(floor, rng, n_positions, n_support, n_query, unseen_query=False):
    """每个位置选不同观测组；源域可要求 Query 来自另外的位置。"""
    eligible = [pos for pos, rows in floor["positions"].items()
                if len(rows) >= n_support + n_query]
    count = n_positions * (2 if unseen_query else 1)
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} positions have enough unique scans")
    chosen = rng.choice(len(eligible), count, replace=False)
    support, query = [], []
    for j in chosen[:n_positions]:
        rows = rng.choice(floor["positions"][eligible[j]], n_support + (0 if unseen_query else n_query), replace=False)
        support.extend(rows[:n_support].tolist())
        if not unseen_query:
            query.extend(rows[n_support:].tolist())
    if unseen_query:
        for j in chosen[n_positions:]:
            query.extend(rng.choice(floor["positions"][eligible[j]], n_query, replace=False).tolist())
    return np.array(support), np.array(query)


def target_split(floor, rng, n_positions, n_support, n_query):
    """固定目标锚点，另取全部未采集位置作为主评价集。"""
    support, matched = sample_episode(floor, rng, n_positions, n_support, n_query)
    selected = {floor["position_keys"][i] for i in support}
    unseen = np.concatenate([rows for pos, rows in floor["positions"].items()
                             if pos not in selected])
    assert not (set(floor["group_ids"][support]) & set(floor["group_ids"][matched]))
    assert not (set(floor["group_ids"][support]) & set(floor["group_ids"][unseen]))
    return support, matched, unseen


def episode_tensors(floor, support, query, coordinate_scale):
    origin = floor["xy"][support].mean(0)
    sy = ((floor["xy"][support] - origin) / coordinate_scale).float()
    qy = ((floor["xy"][query] - origin) / coordinate_scale).float()
    return floor["x"][support], sy, floor["x"][query], qy
