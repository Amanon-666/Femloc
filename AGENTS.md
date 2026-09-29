# Support 位置交叉验证校准分支

- 本分支 `support-ranked-map-adaptation` 从 AP 偏移校准的负结果分出。不得修改任何旧分支或 WIFI-loc。
- `docs/SUPPORT_RANKED_METHOD.md` 是本分支事前写定的方法和门槛；不得看目标 Query 或目标官方 validation 来改候选、规则、参数。
- 只使用历史七对相邻楼层转移作源端门槛。若门槛失败，记录并停止目标评价，不叠加补丁。
- 目标 Support 固定 10 个位置×3 次扫描；所有 Query 标签只用于最终评价。旧楼层完整指纹图是公开源信息。
- 复用上一分支的数据、旧图和信号校正函数；只加选择机制所需代码。注释说明功能，不堆异常处理和无关测试。
- 服务器经 `ssh lab-server 'bash -s'` 操作；GitHub 推送仅走本机 SSH 隧道，凭证不得进入源码或日志。
