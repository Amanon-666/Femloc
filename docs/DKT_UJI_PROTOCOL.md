# DKT-UJI V1：官方回归代码的定向适配

## 1. 依据和边界

论文：Patacchiola et al., NeurIPS 2020, *Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels*，§3、Algorithm 1、Eq.(6–7)、§5.1。
官方仓库：https://github.com/BayesWatch/deep-kernel-transfer ，固定commit `61d95d6ab783be679c09803a33e3c2302287cbc4`。

这是将DKT方法接入UJI，不是复现论文正弦或QMUL数据集的表格数字。原论文没有提供UJI方案。FeMLoc路线在本分支停止，旧文件不参与执行。

**方法思想：学习跨楼层可复用的RSSI相似性函数；新楼层用少量坐标锚点条件化GP，以后验推断代替参数微调。**

## 2. 哪些代码原样保留

- `sines/train_DKT.py::Feature`的完整类定义：两层40维ReLU MLP。构造后只把`layer1`替换为`Linear(520,40)`；不改第二层或激活。
- `methods/DKT_regression.py::ExactGPLayer`完整类定义：ConstantMean、ScaleKernel(RBFKernel)、GaussianLikelihood、ExactGP。
- 两个文件原样存入`third_party/dkt/`；类机械提取到`dkt/upstream_classes.py`，必要import为torch/nn/F/gpytorch。`dkt.check`逐AST对比原类，不通过导入QMUL图像加载器或绘图库制造兼容层。
- 训练保留`set_train_data → model(z) → -ExactMarginalLogLikelihood → backward → Adam.step`。
- 测试保留官方sines的`gp.train → set_train_data → gp.eval → likelihood(gp(query))`，切换train用于清除前任务缓存，没有optimizer或backward。
- 官方wrapper硬编码QMUL人员图像批，因此仅替换数据循环与二维封装，没有重写GP求解器。

## 3. 数据接口与任务

仅读取UJI `trainingData.csv`。一条记录为`(row_id, x∈R^520, p∈R², building, floor)`，`p=(LONGITUDE,LATITUDE)`沿用原始投影坐标数值，以米计算误差。

环境`d=(building,floor)`。目标固定B0F3/B1F3/B2F4；其他10楼层为source。用于延续可核查的楼层环境，但不是“整栋建筑未见”。Building/Floor只组织任务，不进入网络。User/Phone/Time不输入，也不声称其影响已消除。

位置由同楼层完全相同的二维坐标定义，不按SPACEID或房间类别定义。每位置按完整RSSI去重，仅在候选采样池中保留第一条对应row_id；不删除原数据。理由：三份完全一样的记录不能充当三份不同观测预算。Query保留原始扫描权重，另报位置等权MDE。

### Source一个训练任务

1. 均匀选一个source楼层（楼层等权，避免大楼层垄断更新）。
2. 从具有≥3种指纹的位置均匀无放回选10个Support位置，每点随机选3条不同指纹。
3. 从该楼层其余合格位置再选10个位置，每点3条，作为source Query。
4. 共60条带标注数据进入**联合边际似然**；不能换成query预测MSE或MAML inner/outer。

原论文正弦回归每任务5 Support+5 Query、测试仅5 Support；此处保持“训练S∪Q、测试只S”及1:1训练组成，数量改为UJI的10×3是明确适配。

### Target一个评价任务

- 固定`K=10`标注位置、每点`r=3`扫描，`|S|=30`。只能从这30条得到目标坐标中心。
- Query是该楼层所有其他坐标位置的记录；同Support位置的剩余记录也不进入Query，防止同坐标插值被当成未测位置定位。
- 推断函数签名`predict(support_x, support_y, query_x)`，不接收query_y。
- 30条Support仅用于GP后验，不训练新encoder，不训练AE，不更新核/噪声/均值，不使用其他目标扫描拟合模型。
- 每训练seed、每目标抽20个episode。任务随机流固定为10000+seed，保存实际row_id；同一episode两算法完全共享数据。Episode可以互相重叠，不视为独立训练重复。

数据盘点：各目标具有≥3种指纹的位置数为B0F3=68、B1F3=50、B2F4=66，足以实现。某些稀少重复位置不能作为Support，是本协议的候选限制，不能声称代表任意采集点。

## 4. 输入、坐标与二维扩展的推导

### 输入

UJI的520列已定义AP身份，保留固定列顺序，避免发明新表示。缺失100映射0；实测RSSI映射`(r+105)/105`，实际载入时确认落在[0,1]。−105作为低于UJI最弱实测−104的缺失参考，无需目标统计。采用一次线性变换，不沿用FeMLoc指数处理。它保持RSSI强弱次序，并让缺失弱于实测。

这一方式没有解决新AP跨建筑语义：源域未见AP对应权重没有可靠监督。当前明确研究跨楼层；不宣称固定MLP对任意AP集合通用。

### 坐标

绝对坐标百万量级、GP默认核/噪声量级约1，直接拟合会使尺度失衡。每episode取Support的二维均值`c_S`，定义`y=(p−c_S)/a`。`a`仅由10个source楼层各自中心化后的所有坐标分量计算RMS；全任务共用一个标量，保留欧氏距离的方向比例，目标不再fit尺度。

训练任务中S与Q都使用由S决定的c_S；测试同样只从S计算。预测回到`p_hat=a*y_hat+c_S`、方差乘`a²`。这种平移/缩放是UJI数值适配，不是原论文明确提供的预处理。

### 二维输出

官方GP是一维回归。最小扩展是两套原样ExactGPLayer，分别对应x/y，共享Feature，但各自有核、常数均值和噪声参数。假设两输出在给定深特征及参数后条件独立，不引入多输出相关核。

`L=−(log p(y_x|z)+log p(y_y|z))/(2n)`，其中GPyTorch MLL已经除n，封装只对两轴取均值。它与独立联合似然仅差常数，和每轴官方训练目标一致；数值梯度倍率已明确，不暗改为回归MSE。

## 5. 优化、数值和首轮预算

- RBF：官方回归支持的核；RSSI定位不具备已知周期性，因此不照搬正弦实验的spectral核。
- MLP层宽40来自官方正弦Feature。输入改520，输出为40，不自主扩大网络。
- Adam lr=.001沿用官方两个回归入口；PyTorch默认betas/eps，无附加正则、梯度裁剪、scheduler或目标fine-tune。
- torch2.8.0 / gpytorch1.14.2 / numpy2.2.6，固定依赖；未照搬Python3.6的旧运行环境。以直接Cholesky后验公式核验当前GPyTorch语义。
- float64用于GP及小MLP：同位置多个相似指纹可能使矩阵病态，提升计算精度而不更改核或添加训练正则。使用库默认GaussianLikelihood与约束。
- 首轮每seed **5000次**源任务更新，seeds0/1/2。这是预先限定的UJI运行预算（官方正弦代码50000次的1/10），不是论文UJI参数，也不承诺收敛。只保存终点，不按目标Query选择checkpoint。
- 所有episode数据与两个算法一致。对照`RBF-GP`只把Feature替换Identity，同样从source任务学习两轴核/均值/噪声；用于回答深特征有没有帮助，不叫原论文DKBaseline。

## 6. 评价和完成条件

主指标：每episode Query扫描欧氏误差均值MDE；同时记录位置等权MDE、二维RMSE、后验预测耗时。先平均各seed的20个episode，再计算三个seed的均值±样本标准差和配对DKT−RBF误差，不将60个重叠episode当独立重复。

目标梯度步固定0。计时包括Support特征、GP条件化和Query预测；不等同扫描采集耗时。不确定性记录GaussianLikelihood的观测方差，未经校准检验不宣称可信覆盖率。

必要检查：官方类未变、30条预算和位置隔离、Feature梯度存在、GP后验均值/方差与直接Cholesky一致、推断不修改参数。最终核验3×5000训练记录、360条episode方法结果、保存的row_id与预测有限。

本实验不使用官方validationData，不处理设备/用户/时间控制，不加入GUFU或其他表示改造。效果不好如实报告，先审视方法与任务假设，不按目标结果逐个打补丁。
