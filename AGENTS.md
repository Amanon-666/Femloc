# 迁移先验校准地图分支（TPM）

- 当前分支 `tpm-learned-map` 从 `tpm-gufu-unlabeled` 的 `42f5484` 分出；仅在此工作树推进新实验，不修改旧分支。
- `docs/SIGNAL_CALIBRATED_MAP.md` 是方法、数据权限、结果与负结果的依据。主入口 `scripts/evaluate_transfer_map.py`；模型在 `models/radio_map.py`（观测）与 `models/cross_floor.py`（跨楼层先验）。
- 主基准是 7 对历史相邻楼层转移：选参只用它们的 trainingData；第 d 层作目标时先验不含该层（留一楼层）；这 7 层的官方 validation 只在冻结参数后评价。FeMLoc 三目标层只作探索性附录。
- UJI 的 100 是删失，不是 −110 dBm 的测量值。新组件必须能从"跨楼层先验 → 锚点校正 → 删失匹配"这条主线推导出来；推导不出的不进主线。
- 服务器使用 `lab-server` 的 `bash -s`；GitHub 推送只走本机，不在服务器直连。凭证不进入文件、日志或提交。

- 本轮推导与预注册见 `docs/research/GUFU_UNLABELED_DERIVATION.md`；先写推导再实现。无标签更新接口不得接收无标签坐标或 Query 数据。
- 官方 validation 不进入图构建、更新、选参；历史楼层曾多次查看，只称开发基准，不能再称盲测。

- 本轮先读 `docs/research/LEARNED_MAP_DERIVATION.md`。按建筑外层隔离拟合全部统计量与选择参数；第一轮没有 U，只训练锚点条件地图修正规则。
- 不能把历史方法设计已经看过的数据称为全新盲测；只能声称本轮拟合与选参排除了外层建筑。测试建筑的旧层地图是显式授权的部署输入。
