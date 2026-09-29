# MetaLoc 式少样本实验分支

## 目标与依据
- 当前分支 `metaloc-few-shot` 研究 UJI 跨楼层少样本快速适应；`main` 的 FeMLoc EXP1 和旧 WIFI-loc 的 6eb7a63 均为历史对照，不修改。
- 先阅读 docs/METALOC_FEWSHOT_DESIGN.md 和 configs/metaloc_fewshot.json；历史 FeMLoc 审计文档仅供追溯。
- 2026-09-29用户授权自主决定未定项、推送到Amanon-666/Femloc并训练测试。
- 当前执行依据为 docs/METALOC_FEWSHOT_DESIGN.md 与 configs/metaloc_fewshot.json。
- 自行决策须有论文/数学/成熟实现依据，标清工程选择；不可按目标Query调参。
- 论文明确事实、数学推导、原始数据事实、自行设计必须分开标注。
- 原文相互冲突时保留冲突，不把选择的一种解释称为作者确切实现。

## 实现纪律
- 每次实质修改先写清方法思想、依据、成立条件、可能失败原因；实验验证判断。
- 目标预算固定为 10 个位置、每点 3 条 Support 扫描；不从目标 Query 提取 AP 统计、训练或选参。
- 不从旧仓库 import，不复制 TaskManifest、配置或训练入口；原始 UJI 文件可只读共享。
- 不加入 Anchor、Kernel、GGA、GUFU、Set/GNN 或通用算法框架。
- 简单功能性注释；按实际需要拆分 data/models/training/evaluation/scripts/configs/docs。
- 只做必要的数据权限、梯度更新、MI/TL/RI 配对和张量契约检查。
- 不依赖测试指标必须改善来判定代码正确；MI 无收益也如实记录。
- 官方 validationData 不参与任何拟合、模型选择或早停。

## 运行
- 服务器 lab-server；经 bash -s 执行，先读取 gpu-cluster-ops。
- Git 仓库独立，已授权推送新仓库；凭证仅限.git，不输出或提交；GitHub 访问使用本机 SSH 临时隧道。
- 启动已批准实验时同步建立自动值守，结束准确汇报；当前已授权独立实现和训练测试。
