# FeMLoc Reproduction

## 目标与依据
- 独立复现 arXiv:2405.11079v1 的 UJI 实验。旧 WIFI-loc 的 6eb7a63 仅为审计对象。
- 先阅读 docs/FeMLoc_Reproduction_Audit.md、docs/FeMLoc_Reproduction_Specification.md 和 docs/Unresolved_Decisions.md。
- 2026-09-29用户授权自主决定未定项、推送到Amanon-666/Femloc并训练测试。
- 当前执行依据为 docs/IMPLEMENTATION_DECISIONS.md 与 configs/exp1.json；审计文件保留历史。
- 自行决策须有论文/数学/成熟实现依据，标清工程选择；不可按目标Query调参。
- 论文明确事实、数学推导、原始数据事实、自行设计必须分开标注。
- 原文相互冲突时保留冲突，不把选择的一种解释称为作者确切实现。

## 实现纪律
- 每次实质修改先写清方法思想、依据、成立条件、可能失败原因；实验验证判断。
- 不继承 T1/T2、K=10、r=3、20% query、旧清洗或坐标缩放规则。
- 不从旧仓库 import，不复制 TaskManifest、配置或训练入口；原始 UJI 文件可只读共享。
- 不加入 Anchor、Kernel、GGA、GUFU、Set/GNN 或通用算法框架。
- 简单功能性注释；按实际需要拆分 data/models/training/evaluation/scripts/configs/docs。
- 只做必要的数据权限、梯度更新、MI/RI 配对和张量契约检查。
- 不依赖测试指标必须改善来判定代码正确；MI 无收益也如实记录。
- 官方 validationData 不参与任何拟合、模型选择或早停。

## 运行
- 服务器 lab-server；经 bash -s 执行，先读取 gpu-cluster-ops。
- Git 仓库独立，已授权推送新仓库；凭证仅限.git，不输出或提交；GitHub 访问使用本机 SSH 临时隧道。
- 启动已批准实验时同步建立自动值守，结束准确汇报；当前已授权独立实现和训练测试。
