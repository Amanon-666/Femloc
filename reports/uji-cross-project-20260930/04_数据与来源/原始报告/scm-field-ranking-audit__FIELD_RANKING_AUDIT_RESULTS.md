# 冻结 SCM v1 的机制审计结果

本轮没有修改定位方法，也没有依据最终目标楼层调参。SCM 固定为旧图加 10 个目标位置、每处 3 次扫描估得的空间 RSSI 残差场，核长度 60 m、噪声比 0.3；旧图和 SCM 统一用 WKNN `k=8, β=10`。在新的独立工作树 `scm-field-ranking-audit` 执行。详细计算、距离分组和数据权限见 [审计设计](FIELD_RANKING_AUDIT_DESIGN.md)，数值见 `docs/evidence/*.json`。

## 历史相邻楼层：信号场确实有用，但效果依赖距离

七对历史相邻楼层各抽 12 组 10 位置×3 扫描 Support，Query 为同层其它位置、每组随机至多 300 扫描。这个随机种子没有用于旧版 SCM 参数选择；楼层本身曾用于选参，故不是独立外部测试。均方根信号误差在 Query **真实坐标处离线计算**，定位器推理时不能看到该坐标。下表的全维 RMSE 先把缺失 `100` 编为 −110；它是模型编码误差，不是 520 个真实 dBm 测量的物理误差。仅对实际检出的 AP 计算时，七对等权 RSSI RMSE 另为 20.93→15.26 dB。

|汇总方式|旧图 RSSI RMSE|SCM RSSI RMSE|旧图 MDE|SCM MDE|
|---|---:|---:|---:|---:|
|七对等权|5.09 dB|3.88 dB|14.60 m|9.20 m|
|建筑等权|5.06 dB|3.87 dB|14.55 m|9.24 m|

七对的信号 RMSE 与 MDE 均下降，说明 v1 的“旧图 + 稀疏目标锚点”不是仅凭增加候选节点碰巧生效。建筑等权后的距离分组如下；分组是 Query 到最近 Support 位置的水平距离，`far` 组一些楼层样本很少，不能单独当稳定结论。

|距离|旧图→SCM RSSI RMSE|旧图→SCM MDE|定位改善|
|---|---:|---:|---:|
|<20 m|5.04→3.65 dB|14.39→8.10 m|6.29 m|
|20–40 m|4.95→4.18 dB|16.12→12.54 m|3.58 m|
|≥40 m|5.05→4.67 dB|16.77→14.79 m|1.98 m|

信号修正随离锚点距离增加而减弱，定位收益也明显变小。用源端整体平均误差选择一个长度 60 m 的核，无法保证未采集的远处位置有同样收益。

## 既有目标楼层：信号更准不等于定位更准

以下三层的 Support 均从既有 `seed_0/1/2` manifest 读取。表内对三个 Support seed 取平均；官方 validation 分别有 85、47、39 条扫描。`same_day` 是 trainingData 中与 Support **位置互斥**的 Query；`validation` 同时混有采集时间、设备、用户和位置分布变化。

|目标与 Query|到最近锚点均距|旧图→SCM RSSI RMSE|旧图→SCM MDE|
|---|---:|---:|---:|
|B0F3，同日异位置|10.77 m|4.33→3.19 dB|5.80→5.23 m|
|B0F3，validation|10.20 m|4.84→3.96 dB|7.17→6.88 m|
|B1F3，同日异位置|23.30 m|3.66→2.77 dB|12.47→12.90 m|
|B1F3，validation|19.52 m|4.36→3.52 dB|11.72→10.78 m|
|B2F4，同日异位置|7.95 m|5.22→3.93 dB|11.99→7.98 m|
|B2F4，validation|14.51 m|5.29→4.46 dB|15.40→16.15 m|

最能区分假设的是 B1F3 同日异位置与 B2F4 validation：全维编码误差改善，坐标误差却增加。即使只看 B2F4 validation 实际检出的 AP，RSSI RMSE 也由 23.15 降到 16.49 dB，仍不能保证定位改善。B2F4 validation 的近锚点组平均约 29/39 条扫描，定位改善 1.23 m；20–40 m 组约 6 条，退化 6.68 m；≥40 m 组约 4 条，退化 10.26 m。后两组很小，不能据此拟合新规则；它们解释了为何均值可能由少量位置改变。B2F4 validation 到旧图最近坐标也从同日 Query 的平均 0.46 m 增至 1.97 m，说明几何覆盖也变了。时间、设备和用户效应仍纠缠在一起，不能单独归因。

因此，**第一条假设“少量锚点可在历史相邻楼层改善信号预测”有证据；第二条“信号 RMSE 更低就能在新位置得到更低 MDE”不成立。** v1 应保留作强基线，不能只沿着更复杂的 RSSI 重建器迭代。先前 v2 的锚点留一选参与 v3 的分层场/Tobit 匹配在源端都低于冻结 v1，也支持暂时停止该系列扩展；那些试验在其它工作树，本分支不修改它们。

## 候选排序：旧图有空间点，但无线排名仍可能错

又做了一项在看过上述结果后追加的解释性审计：对每条 Query 找旧图中物理距离最近的点，比较该点在旧图与 SCM **同一批旧图坐标**中的无线距离排名。目标坐标只用于离线审计；没有据此修改定位方法。完整结果见 `docs/evidence/candidate_rank_audit.json`。

B2F4 官方 validation 的约 **94.9%** 扫描在旧图中 5 m 内有空间候选；故主要问题并非旧图根本没有该处坐标。然而物理最近候选进入无线前 8 的比例，旧图只有 **17.9%**，SCM 提到 **36.8%**。近锚点组排名改善（平均约 19.7→13.8），20–40 m 与 ≥40 m 组反而恶化（约 24.2→30.4、28.6→42.1）。整层平均排名改善而最终 MDE 退化，说明单个“正确候选”的排名也不能替代**整组候选位置及其权重**的定位损失。

B1F3 同日异位置另显示一个独立因素：只校准旧图、不把 Support 原型另作候选时，MDE 为 **12.24 m**；拼入 10 个 Support 候选后为 **12.90 m**，旧图为 **12.47 m**。加入已知点并非免费增益，可能改变远处 Query 的前 8 个候选。这是事后机制解释，不能据此把“删去锚点候选”宣布为经过独立验证的新方法。

## `100` 哨兵值的含义不能简化成一个固定 RSSI 阈值

在 19,937 条 trainingData 中，将 Building/Floor、坐标、SpaceID、RelativePosition、UserID、PhoneID 完全相同的扫描分组，取时间相邻且相差不超过 10 秒的 17,819 对扫描。某 AP 此次实际读到以下 RSSI 时，下一次为 `100` 的描述性频率是：

|本次 RSSI|AP 观测事件|下一次缺失比例|
|---|---:|---:|
|−60～0 dBm|62,594|0.80%|
|−70～−60 dBm|84,158|1.53%|
|−80～−70 dBm|152,014|4.63%|
|−90～−80 dBm|252,734|14.06%|
|−105～−90 dBm|88,480|35.88%|

这些事件不是独立样本，也不能说明消失究竟由无线波动、人体遮挡、扫描机制还是姿态引起。但很强的 AP 也会短时消失，`100` 是**未检出状态**，不能当作精确的 −110 dBm 读数。现有 SCM 的数值替换在工程上有效，却不是忠实的观测概率模型；v3 用单一删失阈值替代也未得到源端收益。

## 下一步的完整方法方向

若继续开发，应把观测拆成两个通道：`z_a` 表示是否看到 AP，`r_a | z_a=1` 表示看到时的强度。旧楼层全量地图提供每个候选位置的检出频率和条件强度；目标十个锚点只更新这两种空间关系。Query 直接按“该位置出现/消失这些 AP 的可能性 + 出现时强度的可能性”给候选**位置**打分；源端训练或选择必须以整组候选加权后的坐标误差为目标，而非仅优化 520 维信号重建或单个候选排名。可参考 GUFU 显式保留 AP 观测关系的思想，但这里仍是有完整相邻旧图的少样本新楼层定位，不是 GUFU 的同场地长期地图更新。

实现这个方向之前必须先把旧地图的检出频率估计、未出现 AP 的处理、条件 RSSI 噪声和候选分数的校准完整定义，再在历史楼层做留整栋建筑选择；不能看 B2F4 validation 再添距离门控。由于目标三层已多次进入探索，后续对其结果只能作为探索性比较。真正的独立验证需要新的楼层/建筑或其它数据集。

## 复现

在服务器工作树 `/home/panyushuo/projects/panyushuo/FeMLoc-SCM-Field-Ranking-Audit`，使用已有 FeMLoc 环境执行四个脚本：

```bash
OPENBLAS_NUM_THREADS=1 ../FeMLoc-Reproduction/env/bin/python -m scripts.audit_field_vs_ranking --output outputs/field_ranking_audit.json
OPENBLAS_NUM_THREADS=1 ../FeMLoc-Reproduction/env/bin/python -m scripts.audit_target_shift --output outputs/target_shift_audit.json
OPENBLAS_NUM_THREADS=1 ../FeMLoc-Reproduction/env/bin/python -m scripts.audit_candidate_ranks --output outputs/candidate_rank_audit.json
OPENBLAS_NUM_THREADS=1 ../FeMLoc-Reproduction/env/bin/python -m scripts.audit_scan_presence --data /home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv --output outputs/scan_presence_audit.json
```

脚本语法检查通过，四个输出已实际生成并复制到 `docs/evidence/`。本轮没有训练新模型，也没有声称得到超过现有 SCM 的新定位成绩。
