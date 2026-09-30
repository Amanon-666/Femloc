# 相邻楼层地图与 10×3 少样本适应：第一轮方法审计

> **后续更正：** 本文表格来自 `trainingData.csv` 内部自构 Query。官方 `validationData.csv` 的独立检查已完成，B0F3/B1F3/B2F4 零标注软地图误差为 **7.12/12.24/15.28 米**，详见 [`OFFICIAL_VALIDATION.md`](OFFICIAL_VALIDATION.md)。此前“数值接近 FeMLoc”的表述不能外推到官方 validation。

## 实验问题和权限

三个既定目标 B0F3/B1F3/B2F4，各沿用 `metaloc-few-shot` 分支 seeds 0/1/2 的固定 manifest：每层 10 个目标位置，每处 3 条扫描作 Support；不同位置的扫描作 Query。位置集合互斥。RSSI 520 维按 `100→-110` 后 `(r+110)/110` 截断到 [0,1]。完整相邻**旧**楼层的 RSSI 和坐标可建 radio map；目标 Query 坐标只用于最终误差。`validationData.csv` 不参与。所有方法目标梯度步数均为 0，符合“不超过 50 步”，但没有验证 FeMLoc 式 1–50 步梯度适应速度。

论文原文 [FeMLoc](https://arxiv.org/html/2405.11079v1) 表 III 的 B0F3/B1F3/B2F4 数值是 **100 次目标更新**后的 5.1/13.1/12.2 米；原文“10:3”说的是训练与测试任务数的划分，未给出与这里相同的 30 条标注、位置隔离 Query 和相邻旧楼层完整地图协议。因此以下只能做数值参照，不能宣称复现了论文或具有同等数据成本。

## 方法决策和结果

1. 以旧楼层每个坐标位置的平均指纹建图。固定 520 AP 等权的硬最近地图误差为 **7.46/15.18/14.74 米**。旧楼层七对转移等权选择 RBF 软匹配的温度 `beta∈{1,2,5,10,20,50,100}`，选中 `beta=10`（旧转移均值 14.55 米）。这一步没有目标标注。
2. 为检验可学习 RSSI 距离，训练 520 个 AP 权重和 1 个温度：已见转移的新位置验证在 1000→5000 源步持续下降，但目标误差反而恶化。改用**整对楼层转移留出**，连被留楼层作为其他转移下层的情况也移除。候选 0,100,…,3000 源步中，等权旧转移误差最低的是 **0 步：14.55 米**；1000 步 15.61 米，3000 步 16.39 米。说明原来的位置留出验证了同一转移内插值，未验证跨楼层转移能力。因而最终简化为不训练 AP 权重的软地图。
3. 为检验 10×3 标注是否增加信息，分别测试旧楼层 episode 选择的预测坐标核残差校正、RSSI 核残差校正。两者都只使用 30 条 Support，不更新模型权重。三层收益不一致，不能作为可靠的少样本收益结论。

表中为三个固定 manifest 的**扫描级平均定位误差**，形式是均值 ± 三个 seed 的样本标准差，单位米。一个 seed 是一组 support/query 划分，不是论文中的独立随机训练重复。所有方法的 Query 完全相同。

| 方法 | B0F3 | B1F3 | B2F4 |
|---|---:|---:|---:|
| 旧地图硬最近点（0 标注） | 7.46 ± 0.42 | 15.18 ± 1.80 | 14.74 ± 0.69 |
| 旧地图软匹配，源选 β=10（0 标注） | **5.76 ± 0.22** | **13.16 ± 1.26** | **11.76 ± 0.58** |
| 软地图 + 10 锚点预测坐标核 | 5.64 ± 0.11 | 13.84 ± 2.21 | 11.56 ± 0.30 |
| 软地图 + 10 锚点 RSSI 核 | 5.82 ± 0.07 | 14.15 ± 3.17 | 11.13 ± 0.69 |
| FeMLoc 表 III，100 次目标更新（仅参照） | 5.1 | 13.1 | 12.2 |

旧楼层选择 RSSI 核 `(gamma,lambda)=(2,0.3)`，源端七对平均误差约从 14.2 降至 12.9–13.2 米；但 B1F3 三个 seed 的目标误差从软地图的 `[12.54,14.61,12.34]` 变为 `[17.22,14.34,10.89]` 米，随机抽到的十个锚点对结论影响很大。B1F3 Support 到 Query 最近真实位置的平均距离约 23.3 米，明显高于 B0F3 的 10.8 米、B2F4 的 8.0 米。这一距离描述覆盖情况，不是定位误差下界。

## 结论与下一步

本轮找到一个直接、快速、强于先前大网络结果的**相邻旧楼层地图基线**。三目标的数值接近 FeMLoc 表格，但主要收益来自完整旧楼层地图和软平均，**不是 30 条新楼层标注的快速适应**。七旧转移上学习 AP 身份权重还损害了整转移泛化，直接增加训练步数不成立。目标锚点残差校正对 B1 不稳定；继续给单层加阈值、门控或挑好 seed 会污染研究问题，当前停止此类局部修改。

下一阶段应先重新定义何时 10 个标注点应当提供可辨认的新信息：相邻楼层完整地图的可用性、目标锚点覆盖范围、是否允许在标注前查看目标无标签扫描。随后在**新协议**下统一比较旧地图、目标只有 10×3 的局部定位与 FeMLoc/MetaLoc 风格方法。若目标是跨建筑或没有相邻地图的楼层，本方法不适用，应另开主方法，不能把这里的数值外推。

## 复现

从本分支根目录、用相邻 `FeMLoc-Reproduction/env/bin/python`：

```bash
../FeMLoc-Reproduction/env/bin/python -m scripts.evaluate_adjacent_map --output outputs/adjacent_map_v1
../FeMLoc-Reproduction/env/bin/python -m scripts.learn_cross_floor_metric --output outputs/cross_floor_metric_v1
../FeMLoc-Reproduction/env/bin/python -m scripts.learn_cross_floor_metric --source-steps 5000 --output outputs/cross_floor_metric_source_convergence
../FeMLoc-Reproduction/env/bin/python -m scripts.held_transfer_select --output outputs/held_transfer_v1
../FeMLoc-Reproduction/env/bin/python -m scripts.evaluate_softmap --output outputs/softmap_v1
../FeMLoc-Reproduction/env/bin/python -m scripts.evaluate_support_rssi --output outputs/support_rssi_v1
```

具体 row ids 见原始固定 manifest；本分支 `outputs/` 中保留每次的源选择与目标逐 seed 结果，不纳入 Git。
