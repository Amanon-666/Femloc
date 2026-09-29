# 少量锚点更新旧楼层信号地图的独立分支

- 本分支 `spatial-radiomap-adaptation` 从 `adjacent-map-few-shot` 分出，不能修改其他分支或旧 WIFI-loc。
- `docs/SPATIAL_RADIOMAP_METHOD.md` 先于代码写定方法、数据权限与源端门槛；不得依据目标 Query 修改超参数或额外添加机制。
- 旧楼层完整带坐标指纹图可以输入；目标只有固定 10 位置×3 次扫描作为 Support；Query 标签只能评价。
- 历史相邻楼层转移先完成整对留出与较晚验证门槛；失败就停，不在既定三个目标楼层选法。
- 代码只实现信号地图的空间更新和源端评价；复用现有读取、地图、episode，不建通用框架。功能性注释即可。
- 服务器经 `ssh lab-server 'bash -s'` 操作；GitHub 推送只用本机 SSH 临时隧道，凭证不进入源码、日志或提交。
