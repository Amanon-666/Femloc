# FeMLoc-Reproduction

独立重实现 [FeMLoc arXiv:2405.11079v1](https://arxiv.org/html/2405.11079v1) 的UJI EXP1；代码仓库 [Amanon-666/Femloc](https://github.com/Amanon-666/Femloc)。

## 方法与状态

- 每层一个环境，B0F3/B1F3/B2F4为目标，其他10层为source。
- 私有Encoder → 共享Meta-model → 私有Mapper。局部Adam，outer按Query规模加权的一阶梯度步。
- 以Table II为优先依据；d50、32批、5局部步、1000通信轮。
- 当前已实现并通过GPU最小机制检查；完整训练状态及结果见docs/TRAINING_STATUS.md及后续TRAINING_RESULTS.md。
- 这是明确补充缺失细节的独立重实现，不声称完整恢复作者隐藏配方。扫描级分组留出，不是旧位置级few-shot协议；相同位置可跨Support/Query。

## 运行

服务器：`/home/panyushuo/projects/panyushuo/FeMLoc-Reproduction`。独立venv，不依赖旧项目的运行代码或环境。

```bash
python3 -m venv env
env/bin/python -m pip install -r requirements.txt
CUBLAS_WORKSPACE_CONFIG=:4096:8 env/bin/python -m scripts.check_mechanism
bash scripts/train.sh outputs/exp1
```

输出路径必须是新目录；不自动覆盖、续训或重试。完整运行先从十个source内部选择AE学习率/epoch，再执行seeds0/1/2各1000轮，三目标分别配对MI/RI400步。程序结束自动生成docs/TRAINING_RESULTS.md。非有限数直接失败，保留日志。

## 文件职责

|目录|用途|
|---|---|
|data/|原始CSV、观测组划分、Support拟合预处理|
|models/|FeMLoc四个模块及初始化|
|training/|AE选择、联邦meta更新、目标配对适应|
|evaluation/|扫描MDE曲线和首次阈值到达|
|configs/|所有实验数值及固定任务范围|
|scripts/|机制检查、完整运行、结果汇总、原始盘点|
|docs/|审计、决策依据、训练状态和结果|

## 阅读顺序

1. [当前实现决策](docs/IMPLEMENTATION_DECISIONS.md)：所有数值的出处和缺失细节解释。
2. [历史事实审计](docs/FeMLoc_Reproduction_Audit.md)：A/B/C/D/E分类与旧WIFI-loc差异。
3. [最小机制检查](docs/MECHANISM_CHECK.json)。
4. [原始数据盘点](docs/UJI_RAW_AUDIT.json)。

历史未定项与审计版规范保留但已由本轮自主决策覆盖。旧WIFI-loc commit6eb7a63只作审计资料；无旧manifest、模块import、配置继承。原始UJI CSV只读共享，官方validationData不进入训练入口。

## 评价边界

目标使用约4/5观测组作Support，不是30条标注。三个source建筑都有历史楼层，不是整栋建筑未见。MI/RI配对使用同一私有初值、批流、Query；源码不使用目标Query选超参或早停。

AE的候选选择仅用source内部重建误差，恒零和per-AP mean均报告；本轮不加入其他定位方法。不承诺MI必定优于RI。
