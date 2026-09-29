# 信号空间校准地图分支

- 当前分支 `signal-calibrated-map` 从 `adjacent-map-few-shot` 分出；不得修改其他分支。
- `docs/SIGNAL_CALIBRATED_MAP.md` 是方法、数据权限和结果依据。
- 选参只用 7 对历史相邻楼层转移（trainingData）；目标只读 10 位置×3 扫描 Support；`validationData.csv` 只用于冻结参数后的评价。
- 服务器使用 `lab-server` 的 `bash -s`；GitHub 推送只走本机 SSH 隧道，不在服务器直连。凭证不进入文件、日志或提交。
