"""UJI楼层任务、位置级少样本抽取与固定RSSI数值变换。"""
import csv
from collections import defaultdict

import numpy as np


def load(path):
    records = defaultdict(list)
    with open(path, newline='') as f:
        for row_id, row in enumerate(csv.DictReader(f), 1):
            name = f'B{row["BUILDINGID"]}F{row["FLOOR"]}'
            rss = [int(row[f'WAP{i:03}']) for i in range(1, 521)]
            records[name].append((row_id, rss, [float(row['LONGITUDE']), float(row['LATITUDE'])]))
    floors = {}
    for name, rows in sorted(records.items()):
        ids = np.array([r[0] for r in rows])
        rss = np.array([r[1] for r in rows], dtype=np.int16)
        xy = np.array([r[2] for r in rows], dtype=np.float64)
        positions, pos = np.unique(xy, axis=0, return_inverse=True)
        pools = []
        for j in range(len(positions)):
            idx = np.flatnonzero(pos == j)
            # 相同位置的完全相同RSSI只提供一个候选，避免重复记录占据采集预算。
            _, unique = np.unique(rss[idx], axis=0, return_index=True)
            pools.append(idx[np.sort(unique)])
        x = np.where(rss == 100, 0., (rss.astype(float) + 105.) / 105.)
        assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
        floors[name] = dict(ids=ids, x=x, xy=xy, pos=pos, pools=pools)
    return floors


def episode(floor, rng, k, r, query_positions=None):
    """按位置均匀抽K点、每点r个不同指纹；Query与Support坐标完全分离。"""
    eligible = np.array([j for j, pool in enumerate(floor['pools']) if len(pool) >= r])
    selected = rng.choice(eligible, k, replace=False)
    support = np.concatenate([rng.choice(floor['pools'][j], r, replace=False) for j in selected])
    if query_positions is None:
        query = np.flatnonzero(~np.isin(floor['pos'], selected))
    else:
        available = eligible[~np.isin(eligible, selected)]
        chosen = rng.choice(available, query_positions, replace=False)
        query = np.concatenate([rng.choice(floor['pools'][j], r, replace=False) for j in chosen])
    return support, query


def coordinate_scale(floors, names):
    """仅以源楼层内部坐标离散度设定统一米制缩放，不使用目标统计。"""
    centered = [floors[n]['xy'] - floors[n]['xy'].mean(0) for n in names]
    return float(np.sqrt(np.mean(np.concatenate(centered) ** 2)))
