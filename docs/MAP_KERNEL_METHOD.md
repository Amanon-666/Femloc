# 学习旧地图的残差传播核

## 本轮假设（训练前确定）

固定 SCM-T 已优于前一轮学习 AP 可靠度的结果。该学习没有改变地图修正本身。本轮只验证：依据旧 AP 分布学习修正相关性，能否比固定几何相关性更有效地利用同一批目标锚点。

输入为完整相邻旧楼层地图和目标 10 个坐标、每处 3 条不同 RSSI/坐标观测。目标 Query 坐标只在评价时读取。历史 source Query 坐标用于 source 外层定位损失；这一权限不延伸到目标。

任务仍是同建筑相邻楼层、具有水平坐标对应的地图更新。旧地图属于部署输入；不称未知整栋建筑的无地图适应。

## 数据与数学定义

复用原 SCM-T 的数据读取、旧图聚合、三近邻反距离查询、候选与匹配器。100 编成 -110；归一化 e(r)=(r+110)/110，100 对应 0。旧图 v0 是编码平均值，不能解释成仅检出时的平均物理 RSSI。旧检出率 pi0 继续由 cells 的不同 RSSI/坐标观测计算，不额外清洗旧图或重新定义样本权重。

在 Support 位置 s，d_a(s)=三条编码扫描的平均-v0_a(s)。旧图在 s 的 v0 和 pi0 均以既有 3 个几何近邻、1/(距离+1) 插值取得。o 为完整旧图坐标均值，L=100 m。

g_a(p)=[(p_x-o_x)/L,(p_y-o_y)/L,v0_a(p),pi0_a(p)]。

B 是所有 AP 共享的 2x4 无偏置矩阵，唯一的 8 个训练参数。核为 k_a(p,p')=exp(-||B(g_a(p)-g_a(p'))||^2/2)。AP identity 由 WAP 列对应保留，没有 AP/楼层/建筑身份 embedding。

B 初值为 [[L/60,0,0,0],[0,L/60,0,0]]。核的 outputscale=lengthscale=1，均不训练；lambda=.3。初值严格对应已有 60 m 几何核。B 的正交旋转不改变核，所以参数变化同时记录 B^T B 的变化，不能仅凭 B 变化宣称地图改变。

每个 AP 解 (K_SS+.3I)c=d_S，更新旧点 v_hat=v0+K_JS c。所有张量使用 float64，坐标先在 NumPy float64 中减 o。编码结果恢复为 -110+110*v_hat，按旧规则限于 [-110,0]。候选为修正后的全部旧图原型再拼上真实 Support 平均原型；即使坐标重合仍保留两项，与原 SCM-T 相同。Support 原型直接使用观测，不使用收缩后的 GP 均值。没有读取 Query 坐标来补候选。

AP 范围为旧图或 Support 见过的并集。其余 AP 在原 SCM-T 中为对所有候选相同的常数项，因此去掉不改变预测；实际检查全 520 列 NumPy 匹配器与本实现初始化预测一致。

固定匹配 score=-sum_detected((RSSI-V)^2)/(2*16^2)+sum_missing(logPhi((-80-V)/16))；softmax 后输出候选坐标均值。它是沿用的判别匹配分数，不把 -80 解释为真实统一设备检出阈值。

## 源训练与目标适应

只更新 B。每次抽建筑、建筑内转移、转移内 episode，均等概率。目标位置隔离 10x3 不变。

loss=mean_Query ||predicted_xy-true_xy||_2，单位米；梯度经过固定匹配器、地图更新和线性求解回到 B。没有另加重建、对比或 AP 加权损失。Adam lr=.003，默认 betas=(.9,.999)、eps=1e-8，weight_decay=0。8 参数模型不继承旧网络的 weight decay，避免在无数据证据时把非零几何初始化压向零。

固定 600 source 步，与上一轮预算相同。0/100/300/600 只记录拟合建筑的新 source episode，不按目标或 validation 选 checkpoint。目标端冻结 B，解每 AP 的 10x10 系统；目标梯度步为 0。部署耗时单独计地图构建和核求解，不将它描述为完成一个 MAML 梯度步。

## 代码复用

作者仓库 https://github.com/BayesWatch/deep-kernel-transfer，commit 61d95d6ab783be679c09803a33e3c2302287cbc4。

原样提取 methods/DKT_regression.py::ExactGPLayer 到 models/dkt_reference.py，运行直接使用该类的 ScaleKernel(RBFKernel) covariance module；只冻结尺度到 1。Support 条件回归的计算形式参考 sines/train_DKT.py。用 torch.linalg.solve 表达小型可微系统，避免把未观测 query 当成 GP 训练数据或复用已 detach 的预测缓存。

更换输入到上述地图上下文，作者网络替换为 B；均值以旧地图+零均值残差表达；原版联合边际似然替换为 source Query 定位风险。因此本轮是 DKT 代码参考下的任务改造，不是 DKT、FeMLoc 或 GUFU 原文复现。作者类 AST 在检查脚本中与固定原始快照核对。

## 固定比较与范围

前一轮 outputs/scm_t_learned_v1/manifest.json 原样保存为 protocol/map_kernel_source_manifest.json，不重新抽样。既有 MetaLoc 三种子目标划分的 support / unseen_position_query 行号保存为 protocol/map_kernel_targets.json。作者原始回归代码同时保存为 third_party/dkt/DKT_regression.py，避免复现依赖其他工作树的输出。7 对历史转移，每对 32 fit、8 source development、20 evaluation episodes。3 折建筑留出 x 3 采样种子，共 9 模型；全历史 source x 3 种子另训练 3 模型，评价既有 B0F3/B1F3/B2F4 三组 Support manifest，仅作探索性附录。初始化相同，三种子只改变 source 抽样顺序。

主比较是同一完整旧图、同一 Support/Query、同一定位器下的固定几何核与学习核。所有 12 模型训练完并保存权重后，才加载官方 validation。源码不将 validation 交给任何 fit 函数。

建筑等权汇总：先每层平均其 20 个 episode，再建筑内层等权，再 3 建筑等权。报告 3 采样种子的均值和样本标准差，以及学习-固定的配对差；这些种子不等于 3 个独立新场地。扫描等权与位置等权分别计算。

只有三个建筑、七对相关转移；几何核及匹配参数已有全部历史建筑的调优历史。官方 validation 和最终三层曾反复查看，结果属于探索性。设备、用户、时间影响混合；修正拟合的是观测变化，不宣称将其分离。学习可以无收益；本轮不根据目标结果调核、换损失、挑最好步骤或叠加其他模块。
