# DKT-UJI 分支规则

用户2026-09-29终止本分支FeMLoc路线，授权新分支用官方DKT仓库接入UJI并复现。
当前依据：docs/DKT_UJI_PROTOCOL.md、configs/dkt_uji.json。
历史FeMLoc文件仅保留版本历史，不再约束DKT实现，不从其模块import。

- 尽量原样使用BayesWatch/deep-kernel-transfer；源码固定commit，提取类须能与原文逐AST核对。
- 所有UJI适配决策写清依据；不把自己的数据设置、二维扩展称作论文原始实现。
- 每个目标episode仅10位置×3扫描；目标阶段冻结网络、核和likelihood参数，以Support条件化GP。
- Query标签只供评价；不使用目标数据拟合归一化、训练AE或选超参。
- 用简单功能性注释，避免框架、自动修复、兼容层；必要检查仅针对实验契约和数学实现。
- 服务器使用gpu-cluster-ops，经ssh lab-server bash -s；不修改其他工作树。
- GitHub只走本机SSH临时隧道，凭证不得输出/提交。
- 每次训练同步更新自动值守范围；失败不盲重试，完成如实报告。
