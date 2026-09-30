# 第二阶段：由第一阶段负结果引出的开发实验

本登记发生于首次官方 validation 评分之前。第一轮只使用 trainingData：7 个历史转移、K=3/10、各3个 episode，TPM 平均9.004 m；Student-t ν=30 为8.948 m；三种图残差配置约8.999–9.010 m；四种EM配置约9.308–9.591 m。第一轮数据用于方法开发，不是最终成绩。

数学诊断：若先用相同锚点拟合 TPM 地图，再在这些锚点的拟合残差上传播，残差信号可能已经接近零；线性系统右端接近零，自然不会得到有意义的新校正。EM 提高自己定义的混合得分目标，不保证地图位置语义正确，训练似然上升不能推出定位误差下降。

下一阶段不把这两个模块强行保留。增加成熟的非零均值 GP 条件化作为比较：f(x)=m0(x)+g(x)，g~GP(0,k)，后验均值 m0(x)+k(x,S)(K+λI)^(-1)(Y_S-m0(S))。m0 使用未读目标锚点的 TPM 零锚点先验或 source-only GConvLoc，避免对已经被锚点拟合过的模型再使用近零残差。k 分别基于原始指纹或 GConvLoc 的冻结源表示；不把本实现称作原文 DKT。

依据：Rasmussen/Williams GPML (2006), Chapter 2；Patacchiola 等 NeurIPS 2020 Deep Kernel Transfer：https://proceedings.neurips.cc/paper/2020/hash/b9cfe8b6042cf759dc4c0cccb27a6737-Abstract.html 。原文是贝叶斯元学习，本版只是采用冻结源表示的条件 GP，对训练思想和适用范围作明确区分。

对照包含 TPM 原版、几何中位数决策、GConvLoc 0-shot、同标签预算的 GConvLoc+GP、TPM 先验+raw/deep-kernel GP，以及源端选择的简单凸组合（stacking 对照，不作创新声明）。核长度基准由相邻旧层256条源扫描的成对距离中位数决定；倍率0.3/1/3、ridge0.03/0.3/3。候选只在历史 trainingData 的新episode上选择，参数选择和最终外部评分分离。

如果简单方法已经最好，主线就保留简单方法；如果全部不优于TPM，明确交付改进未成立，而不是强行冠以新方法名称。既有10.31 m不是新协议基线。无标签池数据权限也不能被描述成和纯few-shot完全等价。
