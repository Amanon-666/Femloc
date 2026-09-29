# 用户任务与范围覆盖

最新用户要求：另建仓库开展附件工作，不复用当前任务框架。用户随后选择：先只建服务器仓库。

因此附件第十/十一节在旧仓库新增路径的要求，由独立Git仓库取代。旧仓库只读审计；原始CSV可只读共享。以下保留收到的任务附件，作为来源记录。

---

你现在接手仓库：

https://github.com/Amanon-666/WIFI-loc

当前重要历史节点：
commit 6eb7a63

本轮目标不是继续在现有 V1 上调参、打补丁，也不是继续扩展 Anchor、Kernel、GGA、Set Transformer、GUFU 等新方法。

本轮唯一目标是：

【重新以 FeMLoc 原论文和 UJIIndoorLoc 数据本身为依据，对当前 FeMLoc 实现进行一次算法主导的针对性重构。】

核心原则：

1. 先服从 FeMLoc 算法，再决定数据怎样组织。
2. 不再要求 FeMLoc 服从我们以前自行定义的 T1/T2、K=10、r=3、20% Query 等统一框架。
3. 论文明确规定的内容严格还原。
4. 论文未说明的内容，不允许沿用旧 V1 的经验数字，只因为它们“已经存在”。
5. 论文没有规定、但实现必须决定的部分，可以采用我们自己的设计，但必须先给出明确的数理、统计或数据结构依据。
6. 禁止出现没有来源、没有推导、仅凭经验写下的 magic number。
7. 不预先建设“能兼容未来所有算法”的通用框架。代码结构服务于 FeMLoc 当前算法，而不是反过来让算法迁就框架。
8. 允许复用当前仓库中已经正确、独立且不改变实验语义的工程模块，但不得因为复用方便而保留旧实验假设。

--------------------------------------------------
一、开始编码前，先重新审计 FeMLoc 原文
--------------------------------------------------

完整阅读并以以下内容作为主要依据：

FeMLoc:
https://arxiv.org/html/2405.11079v1

重点核对：
- III-A problem formulation
- localization task 定义
- support/query 定义
- meta signal space 定义
- autoencoder
- shared meta-model
- environment-specific mapper
- Algorithm 1 meta-training
- Algorithm 2 meta-testing / adaptation
- UJIIndoorLoc 实验部分
- 网络结构表
- optimizer / learning rate / gradient steps / communication rounds
- RSSI preprocessing
- UJI 的 task 划分和实验比较方法

同时参考 UJIIndoorLoc 官方数据说明和我们此前调研过的成熟 UJI 项目，理解：
- trainingData.csv
- validationData.csv
- 520 WAP
- missing=100
- Building/Floor/Coordinates/User/Phone/Time
各字段的真实语义。

但注意：

【UJI 公开项目的数据使用方式只能帮助判断数据如何合理使用，不能反过来修改 FeMLoc 算法。】

--------------------------------------------------
二、首先建立“论文事实审计表”
--------------------------------------------------

不要直接修改代码。

先把当前 FeMLoc 实现中的所有关键设计逐项分类为：

A. FeMLoc 原文明确定义
B. 可由 FeMLoc 数学公式直接推出
C. UJI 数据自身客观决定
D. 我们旧 V1 自行定义
E. 当前仍无法从论文确定

至少审计：

- 什么是一个 localization task
- UJI 中一个 task 对应什么
- meta-training task 如何选择
- meta-testing task 如何选择
- support/query 如何形成
- 一个 task 中应使用多少样本
- 是否要求物理位置隔离
- 是否需要 K=10 个位置
- 是否需要每位置 3 条扫描
- 是否需要固定 query_fraction=0.2
- source 是否应该使用全部允许数据
- support/query 是否按样本还是按 RP 划分
- trainingData / validationData 各自应承担什么角色
- AP 数量 m_k 如何确定
- meta signal dimension d 如何确定
- RSSI preprocessing
- AE 架构
- AE loss
- AE 是否独立预训练
- 如果预训练，使用哪些数据
- AE 训练步数
- AE optimizer
- Encoder 是否继续参加 localization training
- Mapper 初始化
- Shared model 初始化
- inner loop
- outer update
- optimizer state 是否跨轮保留
- target adaptation 使用多少样本
- target adaptation 更新哪些模块
- adaptation step 数
- evaluation metric
- MI / RI 的准确比较方式

对于 D/E 类：

【先取消旧配置作为默认真值。】

例如旧 V1 中：
K=10
scans_per_position=3
query_fraction=0.2
AE steps=200
target steps=500
RSSI线性缩放
固定latent=50
各种初始化策略
optimizer state 生命周期

只有当论文明确支持，或能够重新给出独立推导时，才允许保留。

不要因为这些数值已经存在于 configs/v1.yaml 就默认它们合理。

--------------------------------------------------
三、对“论文没有说清”的内容采用下面的规则
--------------------------------------------------

绝对禁止：
“论文没写，所以随便取一个常见值”。

必须按照以下优先级决定：

第一优先级：
论文公式、Algorithm、Table、Figure 或实验正文可以推出。

第二优先级：
UJI 数据结构本身可以推出。

第三优先级：
能够通过明确数学或统计理由决定。

例如：
- 维度由实际 AP 数决定；
- meta-space dimension 若论文定义为训练任务 AP 数中位数，就实际计算；
- loss reduction 必须从论文数学定义分析，而不是随手使用 PyTorch 默认；
- 坐标变换如果为了数值稳定，必须写清变换后梯度尺度怎样改变。

第四优先级：
如果无法唯一推导，则把它定义成“待选择的实验参数”，不能静默固定。

这类参数只能：
- 在 source/internal development 数据上选择；
- 或直接报告多组敏感性结果；
- 绝不能根据最终 target query / validationData 结果反复调整。

每一个自行加入的数值参数都必须在文档里回答：

1. 为什么需要它？
2. 为什么是这个数值？
3. 如果改大/改小，数学上或优化上会发生什么？
4. 它来自论文、数据推导还是我们的工程设计？

答不出来就不要加入。

--------------------------------------------------
四、本轮 FeMLoc 数据使用必须由算法定义
--------------------------------------------------

不要再先写一个“统一 TaskManifest”，然后强迫 FeMLoc 使用它。

FeMLoc 原文把每个 localization environment 视为一个 task，
每个 task 有自己的 dataset D，并划分 support Ds 和 query Dq。

因此先按照论文实验重新确定：

UJI 的一个 localization task 到底是什么；
论文 UJI 实验中的 meta-training tasks 和 meta-testing tasks 是什么；
它们使用哪些样本；
每个 task 内 support/query 如何划分。

如果论文明确使用 13 个 floor tasks、10 个 meta-training、3 个 meta-testing，
就优先恢复这个实验语义，而不是沿用我们自己的：
“T1=同建筑其他楼层”
“T2=排除整栋目标建筑”
“K=10位置×3扫描”。

旧 T1/T2 可以保留为以后我们自己的扩展实验，
但不能再冒充 FeMLoc reproduction protocol。

原则：

【复现 FeMLoc 时，先做 FeMLoc 的问题；扩展 FeMLoc 时，再做我们的问题。】

--------------------------------------------------
五、充分利用允许使用的数据
--------------------------------------------------

不要为了形式整齐而人为丢弃 source 数据。

如果某 source task 的数据按照论文权限全部可以用于训练，
就应充分使用它们。

如果算法需要 support/query episode，
就在完整可用 source 数据之上按照论文机制构造 episode。

不要因为旧代码定义：
10 positions × 3 scans
就导致一个有大量样本的 source floor 长期只按固定小集合训练。

必须统计并报告：
- 每个 floor 原始样本数
- RP 数
- AP 数
- 实际参与 source training 的样本覆盖率
- support/query 样本数量
- 每条样本在训练过程中被使用的情况

目标是确认：
数据减少是“算法要求”，不是“旧框架要求”。

--------------------------------------------------
六、trainingData 与 validationData 的使用重新定义
--------------------------------------------------

不要机械规定 validationData 一定参与 FeMLoc 训练或一定不参与。

先区分两个问题：

A. FeMLoc 原文实验本身怎样构造 meta-training / meta-testing；
B. UJI 官方 validationData 能提供什么额外评价。

优先忠实还原 A。

同时可以把 validationData 保留为额外的 external evaluation，
用于检查：
- 后期采集
- 时间变化
- 用户/设备变化
- RSSI drift

但如果 FeMLoc 原论文没有这样评测，
必须明确标记为：

“本项目新增 external robustness evaluation”

不能包装成 FeMLoc 原实验的一部分。

validationData 在任何情况下不能用于：
- 拟合 scaler
- AP筛选
- 坐标变换统计
- early stopping
- hyperparameter tuning

除非某个实验明确把其中的一部分重新定义为允许使用的 development 数据，
且另有真正独立 test。

--------------------------------------------------
七、重新审查 AutoEncoder，而不是修旧 AE
--------------------------------------------------

旧 V1 已经观察到 AE decoder 全零 collapse。

不要继续围绕：
lr=.0095 / steps=200
做补丁。

重新从 FeMLoc 原文定义出发回答：

1. AutoEncoder 在 FeMLoc 中到底承担什么角色？
2. 它是单独预训练还是与定位联合训练？
3. Decoder 在什么时候使用？
4. AE 使用 task 的全部 training/support 数据还是部分数据？
5. latent dimension 应怎样确定？
6. reconstruction loss 是否应该让大量 missing=0 主导？
7. 原文 preprocessing 与我们旧 preprocessing 是否一致？
8. 原文 learning rate 所对应的输入/损失尺度是否与我们一致？

如果论文没有明确 AE 训练步数，
不要再随手规定 200、500、2000。

应根据：
- 收敛条件；
- 独立 source validation reconstruction；
- 简单 reconstruction baseline；
来决定停止方式。

至少必须和：
- 恒零预测
- per-AP mean prediction
比较。

AE loss 下降本身不能作为训练成功依据。

--------------------------------------------------
八、Meta-training 必须直接对应算法
--------------------------------------------------

保留 FeMLoc 真正核心机制：

每个 client/task：
- 私有 encoder
- 共享 meta-model
- 私有 mapper

每轮：
- 从相同 global shared 参数开始；
- 在 task support 上执行局部适应；
- 在 task query 上评价 adapted model；
- 根据论文定义更新 global meta-model；
- 私有参数生命周期严格按照原文。

不要为了复用旧代码而保留：
- 旧 optimizer state 生命周期
- 旧梯度聚合方式
- 旧 loss reduction
- 旧 coordinate scale

除非重新证明它们与原文一致。

同时记录最基本的机制诊断：
- support loss
- query loss
- global parameter update norm
- MI vs RI adaptation curve

只记录直接服务于算法验证的量，
不要建立大型 monitoring framework。

--------------------------------------------------
九、Baseline 也服从 FeMLoc 原实验
--------------------------------------------------

先确认 FeMLoc 原论文到底用什么 baseline、怎样初始化、怎样训练、比较什么。

优先复现论文中的：
- random initialization / NN baseline
- meta initialization
- adaptation speed / fixed-step localization accuracy

不要为了保留旧项目而默认：
WKNN
MLP-Scratch
MLP-FT
必须全部进入 FeMLoc reproduction。

andryr/indoor_localization 可以作为独立 UJI sanity reference，
但不能让它的数据协议决定 FeMLoc。

如果保留 WKNN：
明确它只是经典 fingerprint baseline，
并按照当前实验允许的数据构建 fingerprint database。

--------------------------------------------------
十、当前 GitHub 仓库怎样处理
--------------------------------------------------

不要推翻一切，也不要被原结构绑架。

commit 6eb7a63 及以前的：
- V1
- diagnostics
- Anchor / Raw / Scan exploration
全部作为历史结果冻结。

优先复用这些已经独立且可信的模块：
- UJI CSV 基础读取
- 基本字段解析
- prediction / metric 工具
- 简洁的目录划分
- 实验结果保存
- baseline vendor 原件

以下内容不得因为“已经实现”就自动复用：
- T1/T2 task semantics
- K=10
- r=3
- query_fraction=.2
- fixed TaskManifest 抽样规则
- AE=200 steps
- adapt=500 steps
- 原RSSI normalization
- coordinate_scale=100
- 原 outer lr
- 原 source episode 大小
- 原 target AP 表规则

新的 FeMLoc reproduction 使用独立配置和独立结果目录，
旧结果不可覆盖。

代码仍保持：
data/
models/
training/
evaluation/
scripts/
configs/
docs/

但不要新增 factory、plugin、registry、抽象基类体系，
也不要为了未来 Set/GNN/GGA 设计接口。

【算法需要什么模块，就实现什么模块。】

--------------------------------------------------
十一、工作顺序
--------------------------------------------------

第一步：
阅读 FeMLoc 原文 + 当前代码，输出《FeMLoc Reproduction Audit》。

内容包括：
- 原文明确定义
- 当前实现
- 是否一致
- 不一致原因
- 原文未说明项
- 推荐处理原则
- 哪些旧超参数必须废弃

第二步：
输出新的《FeMLoc Reproduction Specification》。

要求：
- 只包含现在真正决定下来的内容；
- 每个非论文设计注明来源和推导；
- 禁止 magic number；
- 不讨论未来算法。

第三步：
在当前仓库中新建独立 FeMLoc reproduction 路径，
尽量复用可靠基础设施，
逐项替换旧 V1 中不再成立的假设。

第四步：
先做最小机制验证：
- 一两个 source tasks
- 一个 target task
- 检查 AE 是否正常
- 检查 meta update 是否真实发生
- 检查 MI / RI 是否出现合理差异

这里只验证实现，不调目标测试集结果。

第五步：
确认机制正确后，
再完整运行 FeMLoc 原论文 UJI protocol。

最后才讨论：
- 我们自己的 cross-building
- official validationData robustness
- JPRL/DG
- GGA
- Set/Graph representation

这些全部不进入本轮。

--------------------------------------------------
十二、强制研究纪律
--------------------------------------------------

请长期遵守：

【1】没有论文依据或数学推导的数值，不写进代码。

【2】不能因为一个参数“常见”就采用它。

【3】不能根据 target test/query 表现不断试参数。

【4】每一次实质修改必须能够用一句完整的方法思想描述，
而不是“这个结果不好，所以加一个补丁”。

【5】如果某个设计无法说明：
“它解决什么问题、为什么数学上应该有效、可能什么时候失败”，
就暂时不要实现。

【6】实验失败时先检查：
任务定义 → 数据权限 → 表示 → loss → optimization，
而不是直接增加网络模块。

【7】不要预先实现未来功能。

【8】不要为了统一工程接口改变论文算法语义。

【9】不要自动把我们的旧设计继承为默认值。

【10】遇到论文真正没有说明、数学上也无法唯一决定且会显著影响实验语义的问题，
不要擅自填值；集中列入“Unresolved Decisions”，一次性说明：
- 为什么必须决定；
- 可选方案；
- 每个方案依据；
- 推荐方案；
然后等待确认。

--------------------------------------------------
最终目标
--------------------------------------------------

这一轮的目标不是追求最低 UJI 误差。

目标是得到一套我们能够明确回答下面问题的 FeMLoc 实现：

1. 哪些部分确实来自 FeMLoc？
2. 哪些部分是我们为了 UJI 重实现而补充？
3. 每个补充设计为什么这样做？
4. 训练数据到底怎样被使用？
5. meta-training 到底学了什么？
6. 新环境适应时到底使用了什么信息？
7. MI 相对于 RI 的收益来自哪里？
8. 如果结果与论文不同，我们能够定位差异来自任务、数据、表示还是优化，而不是只能继续调参。

在完成这套 FeMLoc reproduction 之前，
不要加入任何新研究方法。
