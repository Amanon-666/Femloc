# Learned Map Kernel

- 唯一修改工作树为 FeMLoc-Learned-Map-Kernel，分支 learned-map-kernel。其他工作树不修改。
- 方法依据 docs/MAP_KERNEL_METHOD.md，参数来自 configs/map_kernel.json。先定义假设再训练，当前只训练共享 2x4 核矩阵 B。
- 完整相邻旧地图为部署输入；目标仅用 10 坐标、每处 3 条不同 RSSI/坐标观测。目标 Query 坐标只用于评价；source Query 坐标只用于 source 外层损失。
- source 训练按建筑留出，不能拟合留出建筑或三个最终目标。validation 不用于拟合、挑步骤或改参数。
- 所有比较读取同一已保存 manifest，不重新抽样。候选、匹配器、编码与固定 SCM-T 共用。
- 优先复用作者代码和已有模块，记录实际函数、固定 commit 与任务改造；不将改造称原文复现。
- AP 参数共享，无 AP/建筑/楼层/用户/设备/时间身份 embedding。注释说明模块功能。
- 只检查实验语义和真实计算路径，不添加通用兼容、fallback、重试框架、额外算法或未来功能。
- 本轮固定预算后如实报告无收益或退化，不依据最终 Query 打补丁或追加训练。
- GitHub 推送只经过本机临时 SSH 反向 SOCKS 隧道；凭证不进入代码、日志和提交。
