# FeMLoc Reproduction Specification — 已确定部分

状态：**审计版，非可执行训练规范**。阻塞项见 `Unresolved_Decisions.md`；未决项没有默认数值。

## 1. 仓库与范围

独立Git仓库 `/home/panyushuo/projects/panyushuo/FeMLoc-Reproduction`。旧WIFI-loc不是运行依赖。
唯一目标为FeMLoc v1的UJI EXP1重实现。算法、实验表和缺失信息之间的差别必须记录。
不执行旧T1/T2、GGA、图或kernel方法。本轮没有获准将旧凭证移入新项目。

## 2. 原始接口

- 输入：原始CSV，只读路径 `/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/`。
- `row_id=(file_name, one_based_record_number)`；记录号不包括标题行。
- RSS：`WAP001..WAP520`，原始缺失符号100，AP列顺序固定。
- `environment=(BUILDINGID,FLOOR)`。网络标签为二维坐标`(LONGITUDE,LATITUDE)`；不把经纬度字段名误当角度转弧度。
- USERID/PHONEID/TIMESTAMP/SPACEID/RELATIVEPOSITION保留为元数据；不作为定位网络特征。
- 不预先去重、删除全缺失行或不足扫描的位置。清洗选择仍待定。
- 审计输出保留每floor的原始行数、精确xy数、SpaceID/RelativePosition数、AP列表和文件摘要。
- 官方validation本轮仅做独立原始文件盘点；不得将其统计用于拟合或决定模型参数。

## 3. 算法对象

每环境k有 `D_k=(D_k^s,D_k^q)`，行级数据权限必须分开。

\[
E_{\alpha_k}:\mathbb R^{m_k}\to\mathbb R^d,\quad
G_\theta:\mathbb R^d\to\mathbb R^{32},\quad
M_{\beta_k}:\mathbb R^{32}\to\mathbb R^2,
\qquad f_k=M_{\beta_k}\circ G_\theta\circ E_{\alpha_k}.
\]

AE的decoder为 `D_{α'_k}:R^d→R^{m_k}`；它不在定位forward中。
模块宽度依据Table II：E隐藏1024；decoder隐藏1024；G隐藏256/128/64；M隐藏64/32。
激活、初始化、AE训练阶段、d的冲突处理尚未批准，不写可运行网络默认。

## 4. Meta-training状态和更新

初始化 global θ；每client创建私有α_k、β_k。每轮r：

1. 把同一个 θ^r 复制为各client的 θ_k^{0,r}。
2. α_k、β_k从上轮接续，首次从初值开始。
3. 在D_k^s执行5次local更新，同时更新α_k、θ_k、β_k。局部批大小32有V-B依据，采批规则待定。
4. 在D_k^q计算适应后θ的梯度：`g_k=∇_{θ_k^N} L_k(f_k,D_k^q)`。
5. `ρ_k=|D_k^q|/Σ_j|D_j^q|`，global `θ^{r+1}=θ^r−0.001 Σ_k ρ_k g_k`。
6. 不平均私有参数，不把query梯度继续用于私有参数更新，不穿过inner优化轨迹补加Hessian。

通信轮数1000来自V-B。query梯度可以分块累计，但等价归约必须在loss决策后明确；不能把单批数量误当完整Dq数量。
client数为10、测试环境为3且每建筑1层，有论文依据；精确floor集合与文件内拆分尚待确认。
Adam与SGD的选择、状态生命周期、loss归约仍未冻结。

## 5. Meta-testing / 配对比较

新环境私有模块不能复制源环境encoder/mapper。MI和RI使用相同目标训练数据、相同私有初值、相同批顺序和适应预算。

\[
\Omega^{MI}_0=(\alpha_0,\theta^R,\beta_0),\qquad
\Omega^{RI}_0=(\alpha_0,\theta^{random},\beta_0).
\]

目标Support更新全部E/G/M。Query仅用于冻结快照的报告，不回传梯度、不早停、不选步数或配置。
AE阶段若采纳，也须MI/RI共享相同目标AE初始结果，单独计其计算预算。
主要比较是MI/RI；不自动添加旧WKNN/MLP实验。

## 6. 评价与实际数据使用

Eq.(20)：`MDE = mean_i ||prediction_i−truth_i||_2`，坐标逆变换后以米计算。
逐测试环境报告，不能将旧“采集组→位置→建筑等权”结果伪称论文MDE。
输出每步或已批准步点的MDE、批大小、累计样本访问数、耗时；阈值和最大步数必须先定。
阈值到达是描述性评价，未到达写未到达，不填上限当作已到达。

训练数据使用记录只需简单计数：每原始row在AE、inner、outer中的访问次数，分别汇总unique覆盖和总访问量。不要为此建设监控框架。
source query本就是meta-training监督信号；target query绝非此权限。
AE要报告与恒零和用AE训练数据估计的per-AP mean预测相比的重建误差，不能只凭loss下降判断训练有效。

## 7. 下一阶段实施条件

先集中确认待决策包，再补齐一个可执行配置。实现顺序为原始读取/明确划分→AE机制→一轮meta更新→配对适应→完整EXP1。
最小机制验证应从source/internal development抽取模拟target，不能用最终三测试楼层反复调实现。
检查权限、梯度和配对状态；不要求MI一定胜RI作为通关条件。
实际长实验启动时建立对应自动值守。目前没有启动训练。
