# UJI 官方 validationData.csv 外部检查

## 为什么补做

此前 `docs/ADJACENT_FLOOR_RESULTS.md` 的 B0F3/B1F3/B2F4 误差全部来自 `trainingData.csv` 内的自构位置隔离 Query，没有使用官方 `validationData.csv`。那些数字只证明内部划分可运行，不能直接代表官方测试或支持“稳定接近 FeMLoc”的说法。用户指出此问题后，冻结旧楼层地图、源端选出的 beta=10、全部三个目标 manifest，以及源端选出的两种校正参数，在官方 validation 上重新评估。未用 validation 坐标训练、选择参数或做校正。

## 输入权限和测试样本

每个目标的相邻旧楼层完整地图仅取 `trainingData.csv`：B0F2→B0F3、B1F2→B1F3、B2F3→B2F4。零标注方法直接定位相应 `validationData.csv` 的扫描。两种少样本校正方法只读取该目标楼层 `trainingData.csv` 固定 manifest 中的 10 位置×3 扫描及其坐标；adapt 函数中的 validation 坐标显式置零，真正坐标只用于最终误差。三个 seed 对应三组 Support，所有方法测试同一批官方 validation 扫描。

| 目标楼层 | validation 扫描数 | 唯一坐标数 | 与 trainingData 全部坐标的精确重合数 | 与选中 Support 坐标重合数 |
|---|---:|---:|---:|---:|
| B0F3 | 85 | 85 | 0 | 0 |
| B1F3 | 47 | 46 | 1 | 0 |
| B2F4 | 39 | 39 | 0 | 0 |

三个目标楼层的 validation 用户均与训练用户无重叠；手机型号分别有 1、0、0 个与训练数据重叠。训练数据采至 2013 年 6 月，validation 从 2013 年 9 月下旬或 10 月初开始。因此这个外部检查混合了位置、时间、用户和设备变化，不能把变化单独归因于跨楼层适应。

## 结果

平均二维坐标误差，米；少样本方法为三个固定 Support 划分的均值 ± 样本标准差。零标注方法对三个划分是同一个预测结果，不把重复数字当成独立运行。

| 方法 | B0F3 | B1F3 | B2F4 |
|---|---:|---:|---:|
| 内部自构 Query：零标注软地图（此前报告） | 5.76 | 13.16 | 11.76 |
| **官方 validation：零标注软地图** | **7.12** | **12.24** | **15.28** |
| 官方 validation：硬最近旧地图 | 8.00 | 13.82 | 20.09 |
| 官方 validation：软地图 + 10 锚点坐标核 | 6.93 ± 0.16 | 12.41 ± 0.90 | 15.06 ± 0.39 |
| 官方 validation：软地图 + 10 锚点 RSSI 核 | 7.12 ± 0.19 | 12.46 ± 1.40 | 14.73 ± 0.39 |

按三目标楼层等权，零标注软地图从内部 10.23 米变为官方 validation 11.54 米；B0、B2 明显变差，B1 略好。锚点校正最多只带来局部小幅收益，B1 两种校正都比零标注旧地图略差。官方 validation 尤其 B2 只有 39 条扫描，单层误差的外推需要谨慎。

**更正结论：** 此前“数值接近 FeMLoc 表 III”的判断只适用于内部自构 Query。官方 validation 下三个目标结果未稳定保持这一数值关系，不能称为稳定复现 FeMLoc，也不能称为完成 few-shot 快速适应。FeMLoc 原文是否使用相同划分和完整相邻旧地图权限仍未对齐；仅凭数值不能比较方法优劣。

## 复现

先按 `docs/ADJACENT_FLOOR_RESULTS.md` 生成源端参数选择输出，然后在本分支根目录运行：

```bash
../FeMLoc-Reproduction/env/bin/python -m scripts.evaluate_official_validation --output outputs/official_validation_v1
```

逐 seed 数字在 `outputs/official_validation_v1/results.json`，此输出不提交 Git。
