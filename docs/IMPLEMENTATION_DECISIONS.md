# EXP1 实现决策与数值来源（2026-09-29）

用户本轮授权自行决定未定项、推送Amanon-666/Femloc并训练。本文与configs/exp1.json取代旧审计版中的“等待确认”，没有复用V1任务框架。

## 方法思想

**在不同环境持续训练私有输入/输出映射，通过各环境适应后的Query梯度学习一个可供新环境初始化的共享中间模型；用配对MI/RI隔离共享初始化的作用。**
任务设置以FeMLoc Table II及EXP1为准。本文填补论文未报告的信息，不声称获得了作者精确配置。
预期风险：私有AE空间不对齐、目标数据并不很少、共享梯度可能被客户端规模主导、Adam重置解释与作者未知实现不同。这些风险先明确，不能看到目标结果后再改变定义。

## 1. 数据与任务

- Table III展示目标固定B0F3/B1F3/B2F4；其他10层参与source训练，共16496原始行。每个目标只做二维回归，不输入Building/Floor。
- 仅trainingData。validationData始终不进入训练程序，原始文件盘点不等于拟合或评价。
- 以完全相同的520维RSSI和二维坐标作为一个观测组；忽略元数据差异后仍相同的输入/标签不能跨Support和Query。这比只按完整CSV行去重更严格，保留组内记录及其原数据权重。
- 使用与GroupShuffleSplit一致的“随机观测组留出”思想。选取5-fold的一个fold为Query，数量`ceil(G/5)`，其余Support。**这里的1/5是成熟五折留出惯例的工程预算，不能数学证明是最优，也不是沿用旧位置级20%规则。** 选择理由是为训练保留多数样本、使几百到几千条记录的环境仍有可评价的Query；增大Query降低训练预算和评价方差，反之亦然。
- 不要求不同RP，目标使用约4/5观测组。明确称“论文式新环境适应”，不称10-shot或严格少标注实验。没有以原文未提供的few-shot预算替代其实验。
- 全缺失和重复扫描保留，不自动清洗。Source Support批流无放回遍历并保留尾批，避免只反复使用固定少量样本。Source Query每轮全部参与outer梯度。
- seeds=0/1/2是三个预先声明的随机拆分与初始化重复；数字只是随机流标识，不经性能选择。三次是本轮计算预算，能给出样本标准差而不足以建立强统计结论。所有重复使用同一具名目标集合。

参考：[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)、[KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)。本实现使用NumPy完成相同的组留出语义，无额外sklearn依赖；不声称随机流与sklearn逐位相同。

## 2. 预处理与网络

- AP表仅来自各自Support中出现过的列，列顺序升序；无额外可见率阈值，query-only AP被忽略且计数。
- 设Support可见RSSI最小值a、最大值b，缺失填`a−1dBm`。1来自UJI整数RSSI分辨率，且使缺失严格小于最弱实测。`x=clip((r−(a−1))/(b−(a−1)),0,1)^e`。e为论文Eq.(7)示例；不是声称UJI作者实际指数已确定。指数越大越强调强信号，稀疏性也增加。
- Query不重新fit；范围外固定截断并报告数量。最大/最小统计不能从最终Query推导。
- 坐标只减Support扫描坐标均值，scale=1米，不做坐标除100。去掉百万量级偏置，保留误差量纲；不声称其与未经平移的随机初始化等价。
- d=50按Table II，不使用Eq.(1)的中位数；155的原始盘点保留作为差异说明。
- 层宽全部来自Table II：E m→1024→50；D 50→1024→m；G 50→256→128→64→32；M 32→64→32→2。
- 隐层ReLU，模块末层线性。AE采用线性重建输出：平方损失下最优预测为条件均值，输出不需要额外Sigmoid约束；避免把稀疏近零目标再叠加Sigmoid饱和导数。这是有数学依据的补充，不是论文已明确激活。
- ReLU隐藏层He/Kaiming初始化（fan-in方差2/n），线性末层Xavier；bias=0。前者补偿ReLU保留部分输入能量，后者平衡线性输入输出方差，不引入专门可调gain。无BN/dropout/梯度裁剪。

参考：[He等，2015](https://arxiv.org/abs/1502.01852)、[Glorot与Bengio，2010](https://proceedings.mlr.press/v9/glorot10a.html)。新增激活与初始化仍属于重实现选择。

## 3. AE的独立选择过程

AE在定位前独立预训练，定位阶段不加入重建loss；decoder之后不用。重建loss是样本和AP全部元素的平方误差均值，不重新加权missing。

唯一自动选择的参数是AE学习率和epoch预算，规则在看目标结果前冻结：

1. 从seed0的每个source Support内再次按观测组留出1/5作为AE内部验证；仅其余4/5拟合AP表和RSSI统计。Source定位Query也不参加该AE选择。
2. 候选学习率为Table II出现的`.0095`与`.0005`，共享同一AE初值和批流；前者是论文AE值，后者是论文另一模块的较小值，若被选中明确报告为AE实验偏离。
3. epoch候选按`1,2,4,...`倍增，以覆盖不同数量级的收敛预算；二倍间隔控制搜索成本，细化会增大成本但不增加任务信息。
4. 最大候选epoch由论文定位local总更新数`1000×5=5000`决定：所有source候选中最大每epoch批数乘epoch不得超过5000。这是计算预算约束，不是假设AE一定在此收敛。
5. 每候选计算十楼层“留出重建MSE / 仅AE-fit数据估计的per-AP mean基线MSE”的等权平均；取最小值，平手选较短epoch、再较小lr。每层另报恒零baseline。不用目标位置误差或目标Query选超参。
6. 选定lr/epoch后，每个source在完整Support重训AE；每个target仅在自己Support执行同样epoch预算。它不是每环境固定同一梯度步数，实际访问量单独记录。

不把AE loss下降当作成功，若所选平均重建误差仍不及per-AP mean，停止进入完整定位训练并报告。此门槛1表示“相对基线没有改进”，不是任意容差。

## 4. 定位训练与优化状态

- batch32、每轮5局部步、1000轮、outer lr=.001、E lr=.0095、G/M lr=.0005均有FeMLoc实验依据。
- 定位loss=`Σ_i Σ_{axis=1}^2 error²/(2B)`；选择实验MSE的均值解释，区别于Eq.(6)的字面sum。所有client统一归约，再按完整Query规模加权。
- Adam参数β1=.9、β2=.999、ε=1e−8使用Adam原论文推荐值，非FeMLoc声明。较大β延长动量记忆，ε限制近零二阶矩除法；weight_decay=0对应论文没有额外正则项。见[Adam原文](https://arxiv.org/abs/1412.6980)。
- α/β和它们的Adam状态跨轮保留；θ每轮复制global，local Adam状态清空。理由是θ位置被广播覆盖，而私有参数连续。论文只明确参数生命周期，优化状态规则为本实现解释。
- Query按32分块累计，每块mean梯度乘`块大小/所有client总Query数`，恰好等于每client完整Query均值梯度乘ρ。最后global手动减`.001×梯度`，无外层Adam、Hessian或参数平均。
- 日志：每轮每client Support/Query MSE、权重及global更新范数；每row AE/inner/outer访问次数。终止于非有限loss/gradient，不自动裁剪或重试。

## 5. 目标适应与报告

- MI/RI共享目标AE和mapper初态、Support批流；RI用同seed初始global，MI用1000轮global；两者全新Adam，并更新E/G/M。
- 最大适应步数400：Table III EXP1最迟报告的RI阈值到达为354步，按其50步报告间隔向上取整至400。覆盖表内慢适应范围；比旧500更有具体来源，但不保证任何目标一定到达阈值。
- 每步生成Query预测，训练函数不接收Query标签；全部适应完成后再计算曲线，不早停、不按曲线选模型。
- 预先报告50/100/150（表内步点）和400（预算末步）；保存每步MDE及阈值首次到达步。三目标阈值来自Table III。未到达不替换成400。
- AE时间和定位适应访问量不能混作步数收益；报告MI−RI配对差值及跨seed均值/样本标准差，允许退化。

## 6. 运算与实现范围

独立Python venv，仅直接依赖NumPy2.2.6与PyTorch2.8.0，版本固定是复现环境选择，不改变任务。使用GPU0、单CPU线程以减少小矩阵调度开销；不影响数学批数。
CUDA确定性选项与CUBLAS_WORKSPACE_CONFIG是可复现执行配置，不是超参数。
先用两个source客户端和一个source内部模拟新环境检查数据/梯度/配对机制；不使用最终target Query调试。
通过后执行3次完整训练，共9个目标对比、18条MI/RI适应曲线。原任务自动值守改为监控此独立项目，失败不盲重试。
