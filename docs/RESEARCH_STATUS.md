# 2026-09-30：地图学习第一轮

两组实验全部结束：各 12 个模型，每模型 600 次历史源端更新；共 24 个模型。训练、测试、汇总自动顺序执行，失败会留下 `.failed`。没有待运行的本分支实验。

## 最显著结果

主结果是七个历史上层的官方 validation。每层 20 组相同 10×3 Support，学习按三折整栋建筑留出，各三模型种子；先层内平均，再建筑等权。

|方法|扫描 MDE（米）|代表什么|
|---|---:|---|
|零标注相邻旧图|20.67|旧图本身有价值，但信号跨层失配明显|
|SCM-T，30 扫描，不训练网络|10.98|当前最好的固定地图适应参照|
|拆分检出/强度，冻结匹配|15.72|表达更细不等于定位更好|
|拆分检出/强度，学习匹配|12.66 ± 0.13|训练能改善同一表达，七层都改善；仍逊于 SCM-T|
|保持 SCM-T 地图，学习匹配|11.23 ± 0.16|只两层改善，其余五层退化，没有稳定附加收益|

± 仅为三次模型初始化的样本标准差。数据集只有三个建筑，不能解释为跨场地置信区间；历史全局核、SCM-T 参数曾在全部源建筑选择，整栋留出隔离的是本轮网络拟合，不是全部历史设计。

同批次异位置的建筑等权 MDE：SCM-T 8.20，双通道学习 8.53，SCM-T 学习 8.45。新匹配器在未经过官方 validation 的同批次留出建筑上也没有超过固定 SCM-T，不能把退化唯一归因于验证集的时间/设备差异。

源端新 episode 的训练曲线（全源拟合，三模型均值）：双通道 10.24→8.70，SCM-T 学习 8.53→8.26（0→600 步）。源码、非零有限梯度和权重检查证明更新确实发生；问题是改进没有稳定迁移到整栋留出。

## 瓶颈与下一步判断

已经有证据的瓶颈：重建更细的观测表示不保证候选位置排序更准；训练本建筑可用的 AP 证据权重，不保证在另栋建筑仍有用。本轮没有证明哪一个 AP、用户、设备或字段是唯一原因。

具体风险是三个锚点扫描不足以精确估计局部检出率/条件强度，独立通道空间外推会提供不同于 SCM-T 编码场的候选分布；可靠度网络只能改投票力度，不能改错了的地图关系。这是结构推导，尚未单独因果验证。

当前保留 SCM-T 为主参照，两个学习版本保留为负结果。后续若继续，应明确学习“少量锚点的变化沿同 AP 及相邻位置关系怎样传播到未测处”，即改变地图更新规则；目标端仍保留现有密集旧坐标。GUFU 的图聚合/地图更新提供方法参考，作者代码的缺件需明确修复才能复现，不能伪称直接跑通。停止继续叠加本轮 AP 打分技巧。

## 项目位置及复现

服务器：`/home/panyushuo/projects/panyushuo/FeMLoc-Learned-AP-Map`，分支 `learned-ap-map`，从 `signal-calibrated-map@e8e9a16` 建立。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 ../DKT-UJI/env/bin/python -m scripts.check_learned_ap_map
../DKT-UJI/env/bin/python -m scripts.run_learned_ap_map --config configs/learned_ap_map.json --output outputs/new_two_channel_run
../DKT-UJI/env/bin/python -m scripts.summarize_learned_ap_map outputs/new_two_channel_run
../DKT-UJI/env/bin/python -m scripts.run_learned_ap_map --config configs/scm_t_learned.json --output outputs/new_scm_t_run
../DKT-UJI/env/bin/python -m scripts.summarize_learned_ap_map outputs/new_scm_t_run --report docs/SCM_T_LEARNED_RESULTS.md
```

运行约 34/30 秒；目标端更新只解小型线性方程，没有梯度步。原始 row-id manifest、逐 episode 结果、曲线、权重与 completed 标记分别在 `outputs/learned_ap_map_v1`、`outputs/scm_t_learned_v1`。完整表见 `LEARNED_AP_MAP_RESULTS.md`、`SCM_T_LEARNED_RESULTS.md`。

已核验：24 权重文件各 600 步且有限；学习参数无目标楼层/留出建筑拟合；两个实验固定 manifest 一致；每 Support 位置 3 条不同指纹；Query 与 Support 的位置互斥；所有指标有限。官方 validation 未进入拟合/停止选择。三个既有最终楼层均已多次被查看，只作探索性附录。

原有工作树和其未提交文件保持原状。`AGENTS.md` 已加入原文/作者仓库优先复用规则，并记录 GUFU 固定 commit、实际移植函数与未能直接复用的具体原因。
