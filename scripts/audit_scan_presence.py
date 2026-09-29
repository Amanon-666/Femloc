"""统计同条件短时间相邻扫描的 AP 检出与消失频率。"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BINS = [(-60, 0), (-70, -60), (-80, -70), (-90, -80), (-105, -90)]
META_COLS = (520, 521, 522, 523, 524, 525, 526, 527)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    with open(args.data, newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            key = tuple(row[i] for i in META_COLS)
            groups[key].append((int(row[528]), np.asarray(row[:520], dtype=np.int16)))
    n = np.zeros(len(BINS), dtype=np.int64)
    missing = np.zeros(len(BINS), dtype=np.int64)
    pairs = 0
    for scans in groups.values():
        scans.sort(key=lambda value: value[0])
        for (t1, a), (t2, b) in zip(scans, scans[1:]):
            if t2 - t1 > 10:
                continue
            pairs += 1
            for observed, other in ((a, b), (b, a)):
                for index, (left, right) in enumerate(BINS):
                    selected = (observed >= left) & (observed < right)
                    n[index] += int(selected.sum())
                    missing[index] += int((selected & (other == 100)).sum())
    output = {
        "rows_grouped": int(sum(len(v) for v in groups.values())),
        "groups": len(groups),
        "consecutive_pairs_within_10s": pairs,
        "bins": [{"rssi_dbm": [left, right], "observed_ap_events": int(n[i]),
                  "next_scan_missing": int(missing[i]),
                  "fraction_missing": float(missing[i] / n[i])}
                 for i, (left, right) in enumerate(BINS)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
