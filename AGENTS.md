# 跨设备源任务分支

- 本分支 `cross-device-episodes` 从 `support-conditioned` 分出；旧代码、权重和结果仅用于只读比较。
- 当前方法及信息权限以 `docs/CROSS_DEVICE_METHOD.md` 和 `configs/cross_device.json` 为准。
- 每次实质修改对应一个可用自然语言解释的训练或表示思想；先检查源任务、坐标与目标信息权限，再实验。
- 训练只用七个源楼层；完整开发楼层只用于训练轮数选择；确认楼层仅开放固定 Support，Query 标签只用于评分。
- 保留所有随机种子的实际划分、训练曲线和负结果。不要按确认目标或官方 validation 反调方法。
- 不引入新的算法框架、复杂容错、自动修复和额外数据清洗。只做影响科学结论的必要检查。
- 服务器操作先读 `gpu-cluster-ops`，通过 `ssh lab-server 'bash -s'`。旧分支文件与输出不修改。
