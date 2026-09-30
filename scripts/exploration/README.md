# 探索快照（2026-09-30）

本目录是推导"迁移先验地图"主线时的一次性诊断。主线代码在下一个提交里精简，本目录随之删除；需要复现时检出本提交。
结论汇总在 `docs/SIGNAL_CALIBRATED_MAP.md`。`logs/` 是运行当时的原始输出。

运行方式：`PYTHONPATH=. ../DKT-UJI/env/bin/python scripts/exploration/<脚本>.py`（服务器分支根目录）。

| 脚本 | 问题 | 结论 |
|---|---|---|
| diag_space.py | 跨楼层残差应在哪个空间建模 | 两层都可靠检出的单元上，每 AP 一个常数的留一 R²：潜在（检出值）空间 0.78，插补空间 0.55；逐 AP 偏移按 AP 所在楼层双峰；漏检率随信号平滑下降 |
| check_obs.py | hurdle 观测模型本身是否正确 | 用目标层完整地图时 hurdle 匹配 7.58 m，优于 Tobit 7.93 m |
| attrib.py | SCM 虚拟地图的误差来自哪类单元 | 旧层可靠检出单元的误差贡献最大 |
| ppc.py | 零标注先验的后验预测检验 | 残差随旧层检出水平系统变化，引出 KOH 斜率 ρ_z；检出概率在高端偏高约 0.1 |
| gp_struct2.py | 跨楼层残差的协方差结构 | ρ_z 在 7/7 留出楼层上提高似然（−0.14 nats/单元）；再加长程尺度无留出收益 |
| diag_rho.py | ρ_z 对定位的作用 | 留出复合似然 5/7 折改善；零标注与锚点更新后的定位都改善 |
| diag_ldpl.py | 三维路径损耗物理模型能否迁移 | 负结果：零标注 18.50 m，差于旧图 17.34 m |
| diag_lrm_T.py、diag_2x2.py | 潜在空间 EP + hurdle 匹配为何落后 | AP 独立乘积似然过度自信（需温度 T≈3–4）；场与匹配两处都落后于 SCM-T |
| diag_prior_base.py | 先验地图作 SCM 基图 | 主线来源：val 11.06 → 10.15 m，K=3 14.60 → 11.95 m |
| diag_update.py | 锚点更新放潜在空间（EP）还是观测空间（GP） | EP 在同日异位置 Query 上 0/7 层改善，平均差 1.48 m |

`scripts/evaluate_latent_map.py`、`diag_lrm_T.py`、`diag_2x2.py`、`ppc.py` 运行时 `fit_transfer` 还没有 ρ_z（等价于 ρ≡1）；本提交的 `fit_transfer` 默认 `slope=True`，直接重跑这几个脚本的数字会与 `logs/` 略有不同。其余脚本与本提交代码一致。
