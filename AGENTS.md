# Support 闭式适应主线

当前分支 `r2d2-uji-reference`，来源 `ridge-meta-fewshot@eb0ae3f`；不得修改其他工作树。
先读 `docs/RIDGE_REFERENCE_METHOD.md` 和 `configs/ridge_reference.json`。

- 研究目标：历史楼层学表征，新楼层只标定 10 个位置、每位置 3 扫描，一次求解坐标映射，定位未标定位置。
- Meta-Ridge 与 Sup-Ridge 共用网络、源 episode、目标 manifest 和岭回归；源监督只用固定楼层原点和私有源线性头。分别用历史留出楼层选择参数，不借目标评价选择。
- Query 坐标不进入适应，官方 validation 只在训练选择完成后评价。目标楼层已有历史结果，本轮仍是探索性实验。
- 保留原始 RSSI、随机特征、Support-WKNN 和完整相邻楼层地图作为对照；地图的额外空间信息须说明。
- 每次实质修改先写可解释的方法假设；不叠加 fallback、抽象工厂、未来模块或大批测试。只写必要科学不变量检查和模块功能注释。
- 服务器通过 `ssh lab-server 'bash -s'` 操作，无 Slurm。GitHub 只走本机临时隧道；不输出或提交凭证。
