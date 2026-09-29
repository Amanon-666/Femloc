"""统计原始 UJI 文件的楼层、坐标位置、AP 与重复记录，不执行清洗或划分。"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median


def audit(path):
    """保留所有原始记录，分别报告每个 Building-Floor 的观测数量。"""
    floors = defaultdict(list)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        waps = [name for name in names if name.startswith("WAP")]
        for row in reader:
            floors[(int(row["BUILDINGID"]), int(row["FLOOR"]))].append(row)
    summaries = []
    for (building, floor), rows in sorted(floors.items()):
        observations = [[int(row[name]) for name in waps] for row in rows]
        visible = [v for scan in observations for v in scan if v != 100]
        aps = [name for i, name in enumerate(waps) if any(scan[i] != 100 for scan in observations)]
        summaries.append({
            "floor": f"B{building}F{floor}", "rows": len(rows),
            "xy_positions": len({(Decimal(row["LONGITUDE"]), Decimal(row["LATITUDE"])) for row in rows}),
            "space_relative_pairs": len({(row["SPACEID"], row["RELATIVEPOSITION"]) for row in rows}),
            "ap_count": len(aps), "ap_columns": aps,
            "observed_rssi_min": min(visible) if visible else None,
            "observed_rssi_max": max(visible) if visible else None,
            "all_missing_rows": sum(all(v == 100 for v in scan) for scan in observations),
            "exact_duplicate_rows": len(rows) - len({tuple(row[name] for name in names) for row in rows}),
        })
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "columns": len(names), "wap_columns": len(waps),
            "rows": sum(item["rows"] for item in summaries), "floors": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [audit(args.raw_directory / name) for name in ("trainingData.csv", "validationData.csv")]
    illustrated_targets = {"B0F3", "B1F3", "B2F4"}
    source = [floor for floor in files[0]["floors"] if floor["floor"] not in illustrated_targets]
    result = {"files": files,
              "scope": "Raw file inventory only; no cleaning, split, fitting or training.",
              "training_file_illustrated_ten_floor_ap_median": median(floor["ap_count"] for floor in source),
              "median_note": "Training file only, retain every observed AP, exclude Table III illustrated test floors; not yet a selected reproduction protocol.",
              "training_status": "not_started", "actual_source_training_coverage": 0,
              "support_query_counts": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"file_rows": {f["file"]: f["rows"] for f in files},
                      "source_ap_median": result["training_file_illustrated_ten_floor_ap_median"]}))


if __name__ == "__main__":
    main()
