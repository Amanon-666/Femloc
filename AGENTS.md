# AP 偏移校准实验分支

- 本分支 `ap-offset-map-adaptation` 从 `adjacent-map-few-shot` 分出。不得修改其他分支、旧 WIFI-loc 或其结果。
- `docs/AP_OFFSET_METHOD.md` 先于实现写定了假设、数据权限、模型、选择和失败判据。不得依据 B0F3/B1F3/B2F4 的 Query 结果调参。
- 目标只开放固定 manifest 的 10 个位置×3 次扫描作为 Support；训练集异位置 Query 与官方 validation 只能评价，不参与校准。
- 旧楼层完整指纹图可供全部方法推理。报告零标注旧图、校准旧图、目标 Support WKNN 在完全相同 Query 上的误差。
- 优先复用现有 UJI 数据读取和旧图预测，代码只覆盖本方法；只检查标签权限、manifest 和有限指标。
- 服务器经 `ssh lab-server 'bash -s'` 操作；GitHub 推送只能走本机 SSH 隧道，凭证不得进入源码或日志。
