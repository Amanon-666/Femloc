"""Read UJI scans and build source-floor and target-support views."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass

import numpy as np
import torch


WAPS = [f"WAP{i:03d}" for i in range(1, 521)]
MAX_APS = 51


@dataclass
class Scans:
    raw: np.ndarray
    xy: np.ndarray
    building: np.ndarray
    floor: np.ndarray
    row_id: np.ndarray
    user: np.ndarray
    phone: np.ndarray
    timestamp: np.ndarray

    def take(self, indices: np.ndarray) -> "Scans":
        return Scans(*(getattr(self, field)[indices] for field in self.__dataclass_fields__))

    def floor_indices(self, name: str) -> np.ndarray:
        building, floor = parse_floor(name)
        return np.flatnonzero((self.building == building) & (self.floor == floor))


def parse_floor(name: str) -> tuple[int, int]:
    building, floor = name.split("F")
    return int(building[1:]), int(floor)


def read_csv(path: str) -> Scans:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    return Scans(
        raw=np.asarray([[float(row[wap]) for wap in WAPS] for row in rows], dtype=np.float32),
        xy=np.asarray([[float(row["LONGITUDE"]), float(row["LATITUDE"])] for row in rows], dtype=np.float32),
        building=np.asarray([int(row["BUILDINGID"]) for row in rows], dtype=np.int16),
        floor=np.asarray([int(row["FLOOR"]) for row in rows], dtype=np.int16),
        row_id=np.arange(len(rows), dtype=np.int32),
        user=np.asarray([int(row["USERID"]) for row in rows], dtype=np.int16),
        phone=np.asarray([int(row["PHONEID"]) for row in rows], dtype=np.int16),
        timestamp=np.asarray([int(row["TIMESTAMP"]) for row in rows], dtype=np.int64),
    )


def source_floors(scans: Scans, target: str) -> list[str]:
    building, target_floor = parse_floor(target)
    floors = sorted(set(scans.floor[scans.building == building].tolist()) - {target_floor})
    return [f"B{building}F{floor}" for floor in floors]


def dense(raw: np.ndarray) -> np.ndarray:
    present = raw != 100
    dbm = np.where(present, raw, -110)
    return np.clip((dbm + 110) / 110, 0, 1).astype(np.float32)


def tokens(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.full((len(raw), MAX_APS), 520, dtype=np.int64)
    values = np.zeros((len(raw), MAX_APS), dtype=np.float32)
    mask = np.zeros((len(raw), MAX_APS), dtype=np.bool_)
    for row_index, row in enumerate(raw):
        visible = np.flatnonzero(row != 100)
        if len(visible) == 0:
            mask[row_index, 0] = True
            continue
        if len(visible) > MAX_APS:
            raise ValueError(f"Scan {row_index} has {len(visible)} APs, above MAX_APS={MAX_APS}")
        n = len(visible)
        ids[row_index, :n] = visible
        values[row_index, :n] = np.clip((row[visible] + 110) / 110, 0, 1)
        mask[row_index, :n] = True
    return ids, values, mask


def tensor_batch(scans: Scans, indices: np.ndarray, origin: np.ndarray, device: torch.device,
                 known_aps: np.ndarray) -> dict:
    raw = scans.raw[indices].copy()
    raw[:, ~known_aps] = 100
    ap_ids, values, mask = tokens(raw)
    return {
        "dense": torch.from_numpy(dense(raw)).to(device),
        "ap_ids": torch.from_numpy(ap_ids).to(device),
        "ap_values": torch.from_numpy(values).to(device),
        "ap_mask": torch.from_numpy(mask).to(device),
        "xy": torch.from_numpy(((scans.xy[indices] - origin) / 100).astype(np.float32)).to(device),
    }


def support_indices(scans: Scans, target: str, positions: int, repeats: int, seed: int) -> np.ndarray:
    target_indices = scans.floor_indices(target)
    groups: dict[tuple[float, float], list[int]] = {}
    for index in target_indices:
        key = tuple(scans.xy[index].tolist())
        groups.setdefault(key, []).append(int(index))
    eligible = [(key, group) for key, group in groups.items() if len(group) >= repeats]
    # Hash ordering makes support independent of Python's hash seed and method order.
    eligible.sort(key=lambda item: hashlib.sha256(f"{seed}:{target}:{item[0]}".encode()).hexdigest())
    if len(eligible) < positions:
        raise ValueError(f"{target} has only {len(eligible)} positions with {repeats} scans")
    selected = []
    for key, group in eligible[:positions]:
        ordered = sorted(group, key=lambda index: hashlib.sha256(f"{seed}:{target}:{key}:{index}".encode()).hexdigest())
        selected.extend(ordered[:repeats])
    return np.asarray(selected, dtype=np.int32)
