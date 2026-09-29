# 相邻楼层空间参照实验分支

- 当前分支 `adjacent-map-few-shot` 从 `metaloc-few-shot` 分出；不得修改 `main`、`metaloc-few-shot` 或旧 WIFI-loc。
- `docs/ADJACENT_MAP_METHOD.md` 是本分支的方法和数据权限依据。先写清方法，再运行；不按目标 Query 改参数。
- 目标只给 10 个位置×3 条扫描；Query 位置不进入拟合和选择；`validationData.csv` 不用。
- 优先复用已有 UJI 读取与 manifest；只加当前方法所需代码，不做通用算法框架或未来功能。
- 必要检查仅覆盖目标标签权限、source-only 参数选择和相同 manifest。
- 服务器使用 `lab-server` 的 `bash -s`；GitHub 推送只走本机 SSH 隧道，不在服务器直连。凭证不进入文件、日志或提交。
