"""核验完整EXP1结果并生成MI/RI误差、步数和源数据覆盖报告。"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, default=Path("docs/TRAINING_RESULTS.md"))
    args = parser.parse_args()
    root = args.output
    config = json.loads((root / "config.json").read_text())
    selection = json.loads((root / "ae_selection.json").read_text())
    by_key, coverage, thresholds = defaultdict(dict), [], []
    for seed in config["seeds"]:
        folder = root / f"seed_{seed}"
        completed = json.loads((folder / "completed.json").read_text())
        assert completed["rounds"] == config["meta"]["rounds"]
        assert completed["adapt_steps"] == config["adapt"]["steps"]
        curves = json.loads((folder / "curves.json").read_text())
        for name in config["targets"]:
            for method in ("MI", "RI"):
                curve = curves[f"{name}/{method}"]
                assert [row["step"] for row in curve] == list(range(config["adapt"]["steps"] + 1))
                assert all(np.isfinite(row["mde"]) for row in curve)
                for step in config["adapt"]["report_steps"]:
                    by_key[(name, step)][(seed, method)] = curve[step]["mde"]
        rows = list(csv.DictReader((folder / "source_coverage.csv").open()))
        covered = sum(int(row["inner_visits"]) + int(row["outer_visits"]) > 0 for row in rows)
        coverage.append((seed, covered, len(rows)))
        thresholds.append((seed, json.loads((folder / "thresholds.json").read_text())))
    lines = ["# FeMLoc EXP1 独立重实现：训练结果", "",
             f"完成 {len(config['seeds'])} 次完整meta训练，每次10个源楼层、{config['meta']['rounds']}轮；"
             f"3个目标×MI/RI×{len(config['seeds'])}重复，共{6*len(config['seeds'])}条适应曲线，每条{config['adapt']['steps']}步。", "",
             f"AE source-only选择：lr={selection['lr']}，epochs={selection['epochs']}；"
             f"源楼层平均留出MSE/均值预测MSE={selection['selection_score']:.4f}。", "",
             "## 逐楼层MDE（米，重复均值±样本标准差）", "",
             "|楼层|步数|RI|MI|配对 MI−RI|", "|---|---:|---:|---:|---:|"]
    for (name, step), values in sorted(by_key.items()):
        ri = [values[(s, "RI")] for s in config["seeds"]]
        mi = [values[(s, "MI")] for s in config["seeds"]]
        delta = [a-b for a,b in zip(mi,ri)]
        formatted = [f"{mean(v):.3f} ± {stdev(v):.3f}" for v in (ri,mi,delta)]
        lines.append(f"|{name}|{step}|" + "|".join(formatted) + "|")
    lines += ["", "负的MI−RI表示MI误差更小；不挑选最好seed或最好步数。三个seed不足以证明普遍优势。", "",
              "## Source数据覆盖", "", "|seed|实际参与定位训练或outer监督的行|允许行数|", "|---|---:|---:|"]
    lines += [f"|{seed}|{seen}|{total}|" for seed, seen, total in coverage]
    lines += ["", "## 首次达到论文误差阈值的步数", "", "|seed|楼层|阈值(m)|RI|MI|", "|---|---|---:|---:|---:|"]
    for seed, floors in thresholds:
        for name, methods in floors.items():
            for threshold, mi in methods["MI"].items():
                ri = methods["RI"][threshold]
                lines.append(f"|{seed}|{name}|{threshold}|{ri if ri is not None else '未达到'}|{mi if mi is not None else '未达到'}|")
    lines += ["", "## 解释范围", "",
              "- 使用trainingData内重复观测组隔离的扫描级划分；相同物理位置可同时出现在Support和Query。",
              "- 目标使用约4/5观测组，具体扫描量由组大小决定；这不是10位置×3扫描的few-shot实验。",
              "- 三建筑各留一层，source仍含三个建筑；这是跨楼层重实现，不是整栋建筑未见的测试。",
              "- 官方validationData未参与训练、AE选择或评价；用户、设备、时间影响没有被排除。",
              "- AE激活/停止、分组拆分、优化器状态、坐标平移为本项目补充；不能声称精确还原作者所有设置。",
              "- 步数从定位适应开始计；AE阶段单独报告，不能把步数下降等同端到端耗时下降。", ""]
    args.report.write_text("\n".join(lines))
    print(args.report)


if __name__ == "__main__":
    main()
