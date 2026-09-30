# TPM 改进研究：实验前登记

基线：TPM `ab1a60b17c7b85542755937c68f89d63ce26255c`。新分支 `research/tpm-evidence-guided-20260930`。
GConvLoc 复用来源：`Amanon-666/uji-explore@c698d5e5ff2014397ecfee9e98bb369d5a52001a`，只复用独立重实现，不称作者代码。保持旧分支不变。

## 问题与选择准则

不预设必须采用 GUFU、GAT 或无标签更新。固定 TPM 的跨楼层先验和锚点 GP 校正，比较：
1. 稳健 Student-t 观测得分，检查平方惩罚是否放大少数错误 AP。
2. 带原地图收缩先验的 EM 更新，用独立目标无标签池估计地图；检查目标函数上升是否真的改善定位。
3. 图拉普拉斯残差回归，仅传播锚点有监督残差，而不无条件平滑预测坐标。
4. 决策目标检查：MDE 最优的 Bayes action 是加权几何中位数；TPM 使用的后验均值对应平方距离损失。
GConvLoc 作为同源数据权限的对照，重新训练留出目标楼层的模型，不复用已看过目标训练数据的标准定位权重。

只根据 trainingData 内的开发实验选择保留的机制。没有益处的模块作为负结果保留，不为了名字完整而进入最终推理路径。不得根据官方 validation 的后验成绩选择方案。

## 数据权限

十个历史楼层：B0F0–F2，B1F0–F2，B2F0–F3。七对历史相邻转移沿用 TPM。目标楼层 d 的迁移参数拟合排除 d 以及包含 d 的转移；GConvLoc 源拟合也排除 d。三个顶部目标 B0F3/B1F3/B2F4 单独作为探索性外部任务，其目标数据不进入本次方法选择。

历史七层已经用于历次研究：本轮新模块的选择只读 trainingData，但这些楼层的 validation 并非从未看过的全新盲测。原 TPM 的 (40 m,0.3,12 dB,-85 dBm) 也来自历史开发。这些限制必须与结果一起披露。

新的统一 episode：固定按坐标分组的 20% 内部 Query，剩余位置为采集池；K=3/5/10/20 位置嵌套，每位置 3 扫描。无标签池从非 Query、非所有 20 个备选锚点位置中抽取最多 300 扫描。位置分组仅由 benchmark 划分器使用，模型接口只接收无标签 RSSI，不接收其坐标。所有方法共享相同支持、无标签池和 Query 行号；正式官方 validation 不能进入无标签池。

新协议改变了支持抽样和内部 Query 定义，因此必须重跑 TPM 对照；禁止直接拿旧的 10.31 m 当新协议基线。

## 文献与数学依据

- GUFU，IMWUT 2025：https://arxiv.org/abs/2507.11038 。图含扫描节点与 AP 节点，涉及虚拟边和增量更新；本研究的简化 EM 或图残差不冒充 GUFU 复现。
- GConvLoc，IEICE 2023：https://doi.org/10.1587/transinf.2022EDL8081 。参考指纹图与注意力聚合。
- Correct and Smooth，ICLR 2021：https://arxiv.org/abs/2010.13993 。区分误差传播与预测平滑；本文原任务是分类，坐标残差版本属于本研究适配。
- Belkin 等，JMLR 2006：https://jmlr.org/papers/v7/belkin06a.html 。Manifold regularization。
- Pan 等，AAAI 2006：https://cdn.aaai.org/AAAI/2006/AAAI06-155.pdf 。半监督流形正则化降低无线定位校准成本。
- Dempster/Laird/Rubin，1977：https://doi.org/10.1111/j.2517-6161.1977.tb01600.x 。EM 的不完全数据优化；不能由训练似然上升推出定位误差下降。
- Jylanki 等，JMLR 2011：https://jmlr.org/papers/v12/jylanki11a.html 。Student-t 稳健观测模型。
- Vardi/Zhang，PNAS 2000：https://doi.org/10.1073/pnas.97.4.1423 。几何中位数与修正 Weiszfeld。

同时调研过 OT 分布对齐（https://arxiv.org/abs/2403.13847）与 WiFi-Diffusion（https://arxiv.org/abs/2503.12004）。前者需要论证采样分布对齐是否保持位置语义；后者依赖 UJI 未提供的障碍布局等信息。暂不作为本轮直接实现路线，不代表这些方法无效。

## 不作的承诺

模型归一化得分没有自动成为已校准的置信度。UJI 的 100 是未报告 AP；低于阈值只是建模假设，还可能含随机漏检。已有 TPM 的判别式 Tobit 参数不等于物理检测阈值。完整目标地图只是高数据参考，不是数学下界。既不承诺某篇论文一定提高指标，也不把不同标签预算下的数字排列成排行榜。
