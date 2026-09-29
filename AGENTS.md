# MapMeta 跨楼层少样本分支

本分支独立于 `metaloc-few-shot` 和 `ridge-meta-fewshot`；不能修改这些分支或主分支。先读 `docs/MAPMETA_METHOD.md`、`configs/map_meta.json`。

- 使用旧楼层完整 radio map 与新楼层 10 位置×3 扫描标注；目标 Query 和官方 validation 标签只用于最终评价。
- 先在四对历史楼层转移上选地图温度并训练；三对整层留出转移上选残差模型的正则和训练轮数。未达到文档中的源端门槛，不用目标结果调参或启动正式目标评价。
- 方法只有一个：旧地图预测作坐标先验，Support 残差经可微岭回归校正。训练特征使少量残差能迁移到未采集位置。不要加入特殊门控、异常处理或目标专属启发式。
- 单独保存 MapMeta 与零目标标签旧地图、普通固定特征残差参照。有效的少样本收益必须超过零标注旧地图。
- 服务器执行先读 gpu-cluster-ops，经 `ssh lab-server 'bash -s'`；无 Slurm。GitHub 推送只经本机 SSH 临时隧道，凭证不进源码。
