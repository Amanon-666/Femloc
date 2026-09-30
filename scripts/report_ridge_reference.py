"""将完成的配对结果整理成可直接阅读的训练与定位报告。"""
from pathlib import Path
from statistics import mean


def write_report(root, results, summary, config):
    methods = ("Meta-Ridge", "Sup-Ridge", "RI-Ridge", "Raw-Ridge", "Support-WKNN", "Old-Map")
    lines = ["# Support 闭式适应：第一版参考结果", "",
             "本轮定位任务为新楼层少量标定：10 个位置 × 每位置 3 条扫描，目标端 0 次梯度更新。"
             "所有方法共用 5 组 Support/Query；有训练的方法各 3 个训练种子。", "",
             "## 历史楼层上的独立选择", "",
             "开发训练 7 层，整层留出 B0F2、B1F2、B2F3；选择指标为三层等权 MDE。"
             "最终模型再用全部 10 个非目标楼层训练。", "",
             "|方法|源更新步数|λ|开发 MDE (m)|", "|---|---:|---:|---:|"]
    for method, pick in summary["selection"].items():
        lines.append(f"|{method}|{pick['source_steps']}|{pick['lambda']}|{pick['mean_mde_m']:.3f}|")
    for scope, title in (("unseen_position", "同日未标定位置"),
                          ("official_validation", "官方 validation")):
        lines += ["", f"## {title}", "", "单位为米，越低越好。先平均 3 个训练种子，再计算 5 组标定划分的均值 ± 样本标准差。", "",
                  "|目标|" + "|".join(methods) + "|", "|---|" + "---:|" * len(methods)]
        for target in config["targets"]:
            entries = {r["method"]: r for r in summary["results"] if r.get("method")
                       and r["target"] == target and r["scope"] == scope}
            cells = [f"{entries[m]['mean_mde_m']:.2f} ± {entries[m]['support_split_sd_m']:.2f}" for m in methods]
            lines.append(f"|{target}|" + "|".join(cells) + "|")
        macros = {m: mean(r["mean_mde_m"] for r in summary["results"] if r.get("method") == m
                         and r["scope"] == scope) for m in methods}
        lines.append("|三建筑等权均值|" + "|".join(f"{macros[m]:.2f}" for m in methods) + "|")
        lines += ["", "|目标|Meta − Sup (m)|Meta − Old-Map (m)|", "|---|---:|---:|"]
        for target in config["targets"]:
            entries = {r["comparison"]: r for r in summary["results"] if r.get("comparison")
                       and r["target"] == target and r["scope"] == scope}
            cells = [f"{entries[f'Meta-Ridge - {m}']['mean_delta_m']:.2f} ± "
                     f"{entries[f'Meta-Ridge - {m}']['support_split_sd_m']:.2f}" for m in ("Sup-Ridge", "Old-Map")]
            lines.append(f"|{target}|" + "|".join(cells) + "|")
    lines += ["", "## 标定点拟合与适应计算", "",
              "|方法|Support 平均误差 (m)|Support 求解耗时中位数 (ms)|", "|---|---:|---:|"]
    import numpy as np
    for method in methods:
        rows = [r for r in results if r["method"] == method and r["scope"] == "support_fit"]
        times = [r["adaptation_ms"] for r in rows if r["adaptation_ms"] is not None]
        text = f"{float(np.median(times)):.3f}" if times else "未计时"
        lines.append(f"|{method}|{mean(r['mde_m'] for r in rows):.3f}|{text}|")
    lines += ["", "求解耗时包含 GPU 上 Support 特征提取与 30×30 线性求解，不含数据读取与 Query 预测。"
              "闭式求解仍然是有监督适应；0 梯度步不等于 0 适应，也不能直接与 MAML 的梯度步数比较。", "",
              "## 结果边界", "",
              "- 少标定方法不持有目标层完整地图。Old-Map 使用相邻源楼层全部已标定位置与绝对坐标，属于额外地图信息对照；其 k=8、β=10 来自既有历史地图选择，不使用本轮目标结果改参。",
              "- 这是跨新楼层定位；源数据包含同建筑其他楼层，不是整栋建筑留出。",
              "- trainingData 的未标定位置测试和官方 validation 分开报告；后者是原发布数据划分的混合变化，没有隔离设备、用户、时间。",
              "- 三个目标层与历史开发层已有前期查看记录；新分支与新 Support 划分不能将其变成盲测。",
              "- 标准差反映本次 5 组标定位置，先平均了 3 个训练种子；15 个组合不能当作 15 个独立环境。",
              "- 本方法是 R2-D2 思想的 UJI 坐标回归改造，未复现其图像分类网络、准确率或分类校准参数，也不是 FeMLoc 的精确复现。", "",
              f"详细数值：`{root}/results.json`、`{root}/summary.json`；固定划分：`{root}/manifest.json`。", ""]
    Path("docs/RIDGE_REFERENCE_RESULTS.md").write_text("\n".join(lines))
