# FeMLoc Reproduction Audit

日期：2026-09-29。对象：arXiv **2405.11079v1** 与 WIFI-loc **6eb7a6304e5d165b9af86e28d355165b443606c5**。
本文件审计论文与旧实现的对应关系，不声称已复现论文结果。新仓库没有继承旧实现。

## 1. 证据及分类

- P：[FeMLoc v1](https://arxiv.org/html/2405.11079v1)。核对 III-A/B/C、IV-A/B/C 与 Algorithms 1/2、V-A/B、Tables II/III；其余实验用于区分 EXP1 与 EXP2/3，不能互借结论。
- U：[UJI 官方说明](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)。字段和原始数量依据。
- O：旧 commit 的 `configs/v1.yaml`、`data/load.py`、`data/preprocess.py`、`models/femloc.py`、`training/common.py`、`training/source.py`、`training/adapt.py`、`evaluation/metrics.py`。
- D：本仓库 `scripts/audit_uji.py` 的原始文件盘点，结果 `UJI_RAW_AUDIT.json`。仅统计，不清洗、不选超参。
- R：[andryr/indoor_localization](https://github.com/andryr/indoor_localization)，仅用于理解另一种 UJI 使用方式。其 README 按 USERID 划分内部训练/验证，官方 validation 作为 test。这不是 FeMLoc 协议。

分类：**A** 原文明示；**B** 公式直接推出；**C** 数据事实；**D** 旧 V1 自定；**E** 原文不足以唯一确定。一个项目可同时有 A/E：比如明确需要 AE，但没交代预训练停止规则。

## 2. 最重要的修正

1. **不能把固定 d=50 直接判成旧 V1 发明。** P Definition 2/Eq.(1) 定义训练任务 AP 数中位数，Table II 同时写 d=50。按 trainingData、Table III 展示的十个源楼层、只删除全缺失 AP 计算，中位数为 **155**。这说明两种定义在这个可核验设置下不一致；不能据此倒推作者用了何种阈值或数据子集。
2. **不能把所有原数值一律废弃。** 1024/256/128/64/32 层宽、模块学习率 .0095/.0005 来自 Table II；32 批大小、5 局部步、.001 外层步长、1000 轮来自 V-B。它们可留作“论文事实”，未解决输入/损失尺度前不能作为已可执行配置。
3. **旧等权聚合不是一般情况下的 Eq.(11)。** 原文权重为客户端 query 数占比。旧每域 query 规模相同，等权恰是那个特殊情况。恢复不等规模 task 后不能继续无条件均分。
4. **原文有实现歧义，不能通过换仓库消除。** SGD 与实验 Adam、Eq.(6) 样本求和与 V-B 的 MSE、d 中位数与固定50、AE 独立预训练细节均需明确解释。
5. **MI 优于 RI 不是代码正确性的判据。** 正确更新和无泄漏可以检查，收益需要观测，不能强制得到“合理正收益”。理论部分依赖光滑性、近似及充分条件，不保证任意数据拆分都改善。

## 3. 逐项事实审计

### 3.1 任务与数据

|项目|类别与原文|旧实现|结论/处理|
|---|---|---|---|
|localization task|A，III-A Def.1：环境的 RSS→坐标及其 loss|Building-Floor 域|环境级语义一致|
|UJI task|A，V-B：每个楼层|13 个 BF|保留 BF 定义|
|meta-training task|A，V-B：10 floors|T1 同楼其他层；T2 其他楼|不一致，旧范围不继承|
|meta-testing task|A，V-B：每建筑留一层，共3|逐一13目标|恢复10:3的实验语义|
|展示的目标身份|A，Table III EXP1：B0F3/B1F3/B2F4|多目标重复|可用作首个具名复现实验；随机10:3重复次数/种子未报|
|Support/Query 定义|A，III-A/IV-C：task 的训练/测试，source query 提供 outer 信号|位置隔离小 episode|保留两种权限，旧抽样方式无原文依据|
|每 task 样本量|E，未给出完整 floor 内 train/test 数量|固定小 episode|必须另定，不把 batch32解释成整个task32条|
|物理位置隔离|E，未明确|强制位置隔离|不能声称原论文如此；确会改变插值任务|
|K=10 位置|D|强制10 RP|删除；论文 K 指客户端数，不是每目标 RP 数|
|r=3 扫描|D|每位置3采集组|删除，不再据此排除位置|
|query_fraction=.2|D/E|位置级20%|删除默认值；原文没提供同样比例|
|source 数据覆盖|A/E：完整 D 分 support/query；批训练|fit pool 上采小批|应遍历允许池；小 batch 本身不等于丢弃整个 source 池，需统计真实访问次数|
|按样本还是 RP 划分|E|按 RP|须选；不能把某种方式伪称原文|
|training/validation 文件角色|C 文件用途；E FeMLoc具体取法|training only|原文介绍数据集，未给清楚逐文件协议；training-only 是待确认重实现选择|
|数据清洗|E|丢弃全缺失、去重、排除不足3组位置|不自动继承；报告原始异常再定|
|User/Phone/Time|C 元数据|不用作网络输入|不输入不等于影响已被排除；本轮无控制采集实验|
|坐标位置 RP|C 数据含xy和SpaceID；E论文RP标识规则|精确BFxy|审计同时统计xy位置与SpaceID/RelativePosition，二者不总一一对应|

### 3.2 表示与模块

|项目|类别与原文|旧实现|结论/处理|
|---|---|---|---|
|AP数 m_k|A，III-C 删除全缺失AP，可选可见率阈值|允许fit池可见并集|“all measurements”权限范围及阈值未报；目标query不得静默用于AP拟合|
|meta维度 d|A冲突：Eq.(1)中位数；Table II固定50|固定50|不能判定50无依据，也不能称已满足Eq.(1)|
|missing|C：100表示未检测|转0|数值0是旧预处理后的编码，不是原始语义|
|RSSI变换|A，Eq.(7) min-max 后幂次；β以e为例，数据集具体值E|[-110,0]线性|明确不一致；边界、指数、imputation需重定|
|AE架构|A，Table II：m→1024→d→1024→m|同层宽、d50|层宽一致；ReLU/Sigmoid非表中明示|
|AE重建目的|A，III-B：维度映射及重建|重建RSSI|角色一致；不能据此推出缺失重加权|
|AE loss|E，未给独立重建loss公式/归约|逐元素MSE均值|可建议平方重建，不能当作作者具体实现|
|AE独立预训练|A/E：重建训练的角色明确；Algorithms未列具体预训练阶段|独立200步|先预训练是可行解释，精确阶段未交代|
|AE数据|E|source AE使用保存的fit批；target全部30support|只允许自己的training/support；具体内部分割需定|
|AE步数|E|200；诊断2000|均不继承，选择source内部重建标准|
|AE优化器|A，Table II Adam/.0095|Adam/.0095|学习率有来源，动量/epsilon未报|
|Encoder继续定位训练|A/B，Eq.(9)(10)更新Ω含α|继续更新|一致|
|Decoder生命周期|A结构中不进入f；E独立训练时序|AE后弃用|定位链无decoder正确；不自动添加联合重建|
|shared架构|A，Table II：d→256→128→64→32|同层宽|一致；激活、初始化仍E|
|mapper架构|A，Table II：32→64→32→2|同层宽|一致；不共享mapper|
|mapper初始化|A随机设初值，E具体分布|Xavier/零bias|旧具体分布不能当作者选择|
|shared初始化|A，IV-C1随机NN，E具体分布|Xavier/零bias|同上|
|坐标变换|E|每fit池不同origin并除100|不继承100；平移/缩放影响输出参数和梯度，须明确|

### 3.3 训练与评价

|项目|类别与原文|旧实现|结论/处理|
|---|---|---|---|
|inner更新模块|A，Ω=(α,θ,β)|三者|一致|
|inner步数|A，V-B：N=5|5|有依据|
|inner优化器|A冲突：IV-A/Eq.(9)SGD；Table II Adam|Adam|实验取Adam有根据；不称其与SGD方程逐步相同|
|local batch|A，V-B全部local更新32|episode30，target30全批|不一致；需按32批训练，尾批策略E|
|轮数|A，V-B：R1000|1000|有依据|
|外层梯度位置|A，Eq.(11)对适应后的θ求query梯度|如此|一致；不是穿过inner优化器的完整MAML二阶梯度|
|外层权重|A，ρ_k=|Dq_k|/Σ|Dq| |恒1/K|只在query数相等时一致|
|outer更新|A，Eq.(11)θ减η加权梯度，V-B η=.001|手动SGD/.001|形式一致|
|私有参数跨轮|A，Eq.(10)与IV-A明确保留α/β|保留|正确|
|shared跨轮|A，每轮从global广播|load_state_dict|正确|
|Adam state跨轮|E，不等同权重生命周期|私有state保留，sharedstate清空|旧选择未获原文直接支持，需明示|
|定位loss|A冲突：Eq.(6)样本求和、V-B MSE；坐标维归约E|torch均值MSE|不能静默依赖torch默认|
|target标注量|E|30条|删除固定30|
|target更新模块|A，Alg.2/Eq.(12)：α/θ/β|三者|一致|
|target步数|E，Alg.2以Nκ参数表示；Table III列固定步评价|500|500不能据旧代码固定；50/100/150有EXP1表格依据|
|最终MDE|A，Eq.(20)逐样本欧氏距离平均|主指标先采集组再位置平均|改为原文扫描MDE；附加位置指标不能替代它|
|adaptation speed|A，Eq.(21)含batch×steps；Eq.(22)文字“inverse”与公式MDE/b不一致|稀疏快照/秒数|保存MDE步曲线、batch和样本访问量；不混用步数与实时时间|
|MI/RI|A，IV-D：同α0/β0，区别θm与θ0|配对私有AE/mapper|比较原则一致；具体随机种子配对属工程控制|
|EXP1 baseline|A，V-B及Table III：RI/MI|五方法|本轮先RI/MI，不搬旧五方法任务集|
|KNN/SVR/TL|A，V-D在第15个月数据比较|andryr WKNN等|不能当作EXP1必需比较|

## 4. 原始数据盘点与范围限制

`trainingData.csv` 19937行、`validationData.csv` 1111行，均529列/520WAP/13楼层，与U一致。D保留全部记录，尚未生成任何Support/Query。
按Table III展示目标排除 B0F3、B1F3、B2F4 后，训练文件十个源楼层共有 **16496** 行。
它们的AP计数排序为 `[88,119,121,133,149,161,168,168,178,190]`，中位数 `(149+161)/2=155`。
这是对一个明确盘点范围的事实，不是论文真正AP选择方式的反向证明。

trainingData有76条全AP缺失记录；逐floor完全重复记录合计637条（保留首次后的重复数）。这些统计不构成自动删除指令。
RP若按精确坐标定义与SpaceID/RelativePosition数量可不同，例如B1F3为50与42。validation的SpaceID/RelativePosition几乎不提供对应RP区分，不能沿用其为位置主键。
本仓库实际 source training 覆盖率目前为0；Support/Query数量未定义，不能把原始行数写成“已参与训练”。

## 5. 数学上必须保留的区别

设每样本误差平方和为 q_i=||ŷ_i-y_i||²。二维输出下，torch默认MSE为 `Σq_i/(2B)`；逐样本平方和再平均为`Σq_i/B`；总平方和为`Σq_i`。三者梯度相差2或2B倍。
在普通SGD中，同样参数更新需反向补偿学习率；Adam有epsilon和状态，不能据此声称严格等价。
如果各task用总和loss且再乘论文ρ，其贡献还多受task query规模影响；若均值loss配ρ，则是扫描加权均值梯度。两者都不能在不声明归约的情况下称“按原式”。

坐标 y'=(y-c)/s，在同一物理预测下平方loss缩为1/s²。对归一化输出ŷ'的导数是`2(ŷ-y)/s`；对物理输出ŷ的导数是`2(ŷ-y)/s²`。网络参数化变换后不能直接沿用原优化轨迹。只平移(s=1)保留物理距离尺度，但也改变输出层初始化对应的物理位置。

## 6. 现在取消的默认配置

T1/T2、10RP×3扫描、位置隔离20%、固定500目标步、固定200AE步、[-110,0]线性缩放、坐标除100、原hash抽样、私有/共享Adam状态规则、Xavier与各激活默认值，全部不作为新仓库运行默认。
50维、32批、1000轮、5步和表中学习率保留其来源，是否构成一个一致的可执行实验，见待决策文档。
本次搜索未核验到作者官方代码，不能把“未找到”写成“作者没有发布”。
