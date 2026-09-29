# UJI 跨楼层 Support 条件定位

这是 `Amanon-666/Femloc` 的 `support-conditioned` 分支。旧 `metaloc-few-shot` 分支和 FeMLoc EXP1 均保留。当前方法在七个历史楼层训练 RSSI 特征，来到新楼层后，只用 **10 个位置×每点 3 条扫描**直接求一个带正则的坐标映射，无需目标梯度步骤。训练目标是让这个映射准确预测未采集位置。

[方法、楼层隔离与评价](docs/SUPPORT_RIDGE_METHOD.md) · [冻结旧模型诊断](docs/FROZEN_HEAD_DIAGNOSTIC.md) · [本轮结果](docs/SUPPORT_RIDGE_RESULTS.md) · [下一步推导](docs/NEXT_STAGE_REASONING.md)

## 运行

服务器工作树：`/home/panyushuo/projects/panyushuo/FeMLoc-Support-Conditioned`，沿用服务器已有的 Python 环境：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 ../UJI-FeMLoc-V1/env/bin/python \
  -m scripts.run_support_ridge --config configs/support_ridge.json \
  --output outputs/new_run
```

输出目录必须尚不存在。运行保留每个种子的源训练曲线、开发及确认划分、最优编码器、四方法配对结果和完成标记，并生成 `docs/SUPPORT_RIDGE_RESULTS.md`。

本方法借鉴[可微闭式求解元学习](https://arxiv.org/abs/1805.08136)的思想，使用 UJI 的 520 维 RSSI 和二维坐标重新定义了跨楼层任务；不是该论文原任务或 FeMLoc 的复现。旧分支的文档仍在仓库中供历史追溯。
