# DKT on UJIIndoorLoc

分支 `dkt-uji`：基于[官方DKT代码](https://github.com/BayesWatch/deep-kernel-transfer)的UJI少样本二维定位适配。
服务器独立工作树：`/home/panyushuo/projects/panyushuo/DKT-UJI`。

- 共享Feature MLP + 两个独立RBF ExactGP；官方类保持原样，MLP只替换520维输入层。
- 源任务按DKT边际似然训练；目标10位置×3扫描，冻结参数直接求后验。
- 目标Support/Query按物理坐标隔离；不做目标预训练、微调或Query拟合。
- 对照为在完全相同源任务训练的原始RSSI RBF-GP，用来检验深特征带来的收益。

阅读[实验规范与适配推导](docs/DKT_UJI_PROTOCOL.md)、[上游来源](third_party/dkt/PROVENANCE.md)。
历史FeMLoc代码保留在分支历史中，DKT不调用其训练或数据模块。

```bash
env/bin/pip install -r requirements-dkt.txt
CUBLAS_WORKSPACE_CONFIG=:4096:8 env/bin/python -m dkt.check
CUBLAS_WORKSPACE_CONFIG=:4096:8 env/bin/python -m dkt.run --output outputs/dkt_v1
env/bin/python -m dkt.summarize outputs/dkt_v1
```
