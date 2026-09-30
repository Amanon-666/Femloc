# 迁移先验校准地图分支（TPM）

- 当前分支 `signal-calibrated-map` 从 `adjacent-map-few-shot` 分出；不得修改其他分支。
- `docs/SIGNAL_CALIBRATED_MAP.md` 是方法、数据权限、结果与负结果的依据。主入口 `scripts/evaluate_transfer_map.py`；模型在 `models/radio_map.py`（观测）与 `models/cross_floor.py`（跨楼层先验）。
- 主基准是 7 对历史相邻楼层转移：选参只用它们的 trainingData；第 d 层作目标时先验不含该层（留一楼层）；这 7 层的官方 validation 只在冻结参数后评价。FeMLoc 三目标层只作探索性附录。
- UJI 的 100 是删失，不是 −110 dBm 的测量值。新组件必须能从"跨楼层先验 → 锚点校正 → 删失匹配"这条主线推导出来；推导不出的不进主线。
- 服务器使用 `lab-server` 的 `bash -s`；GitHub 推送只走本机，不在服务器直连。凭证不进入文件、日志或提交。
