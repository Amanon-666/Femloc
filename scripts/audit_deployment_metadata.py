"""Read-only protocol audit. Select metadata columns; never interpret WAP values.

Run with trainingData.csv validationData.csv. Full bytes are hashed for provenance.
The temporal half split below is a feasibility diagnostic, not a selected protocol.
"""
import collections
import csv
import datetime
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path

FIELDS = ("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "PHONEID", "USERID", "TIMESTAMP")


def utc(timestamp):
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat()


def position(r):
    return (r["LONGITUDE"], r["LATITUDE"])


def describe(rows):
    positions = collections.Counter(position(r) for r in rows)
    times = [r["TIMESTAMP"] for r in rows]
    by_phone = collections.defaultdict(list)
    for r in rows:
        by_phone[r["PHONEID"]].append(r)
    unique_times = sorted(set(times))
    cutoff = unique_times[len(unique_times) // 2]
    early = [r for r in rows if r["TIMESTAMP"] < cutoff]
    late = [r for r in rows if r["TIMESTAMP"] >= cutoff]
    early_pos = {position(r) for r in early}
    late_pos = {position(r) for r in late}
    shared = early_pos & late_pos
    return {
        "rows": len(rows), "positions_exact": len(positions),
        "scans_per_position_min_median_max": [min(positions.values()), statistics.median(positions.values()), max(positions.values())],
        "time_min_max_utc": [utc(min(times)), utc(max(times))],
        "unique_timestamps": len(unique_times),
        "utc_day_counts": dict(sorted(collections.Counter(utc(t)[:10] for t in times).items())),
        "user_counts": dict(sorted(collections.Counter(r["USERID"] for r in rows).items())),
        "phones": {phone: {"rows": len(rr), "positions": len({position(r) for r in rr}),
                           "time_min_max_utc": [utc(min(r["TIMESTAMP"] for r in rr)), utc(max(r["TIMESTAMP"] for r in rr))]}
                   for phone, rr in sorted(by_phone.items())},
        "chronological_half_unique_timestamps_diagnostic": {
            "cutoff_utc": utc(cutoff), "early_rows": len(early), "late_rows": len(late),
            "cross_boundary_positions": len(shared),
            "early_after_purging_shared_positions": sum(position(r) not in shared for r in early),
            "late_after_purging_shared_positions": sum(position(r) not in shared for r in late),
            "early_phones": sorted({r["PHONEID"] for r in early}),
            "late_phones": sorted({r["PHONEID"] for r in late}),
        },
    }


def load(path):
    floors = collections.defaultdict(list)
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        indices = [columns.index(name) for name in FIELDS]
        for raw in reader:
            row = dict(zip(FIELDS, (raw[i] for i in indices)))
            for name in ["LONGITUDE", "LATITUDE"]:
                row[name] = float(row[name])
            row["TIMESTAMP"] = int(row["TIMESTAMP"])
            floors[f'B{row["BUILDINGID"]}F{row["FLOOR"]}'].append(row)
    return floors


def main(train_path, later_path):
    report, raw_sets = {}, []
    for name, path in [("training", train_path), ("later_validation", later_path)]:
        floors = load(path)
        raw_sets.append(floors)
        report[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "rows_total": sum(len(rows) for rows in floors.values()),
                        "floors": {floor: describe(rows) for floor, rows in sorted(floors.items())}}
    training, later = raw_sets
    report["overlap_by_floor"] = {
        floor: {"exact_positions": len({position(r) for r in rows} & {position(r) for r in training[floor]}),
                "phones": sorted({r["PHONEID"] for r in rows} & {r["PHONEID"] for r in training[floor]})}
        for floor, rows in sorted(later.items())}
    report["permissions"] = {"columns_interpreted": list(FIELDS), "rssi_statistics": False,
                              "localization_metrics": False, "training": False,
                              "note": "protocol metadata audit before method freeze; all floors inspected"}
    report["execution"] = {"system": platform.system(), "host": platform.node(), "python": platform.python_version()}
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_deployment_metadata.py trainingData.csv validationData.csv")
    print(json.dumps(main(*(Path(p) for p in sys.argv[1:])), indent=2, ensure_ascii=False, allow_nan=False))
