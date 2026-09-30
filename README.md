# 用少量标定一次适应新楼层：UJI Support 闭式适应

当前分支为 `r2d2-uji-reference`，从 `ridge-meta-fewshot@eb0ae3f` 建立；其他分支和原始结果保留。

- 历史楼层学习 RSSI 表征，新楼层只提供 **10 个位置 × 每点 3 条扫描 = 30 条标注**。
- 由 Support 一次求解岭回归坐标映射；表征冻结，目标梯度步数为 0。
- 在历史留出层分别选择 Meta-Ridge 与普通多楼层监督表征的 λ、源训练步数。
- 比较普通监督、随机特征、原始 RSSI、Support-WKNN 和完整相邻楼层地图。
- 所有方法共享五组位置隔离划分；分别评价同日未标定位置和官方 validation。

[方法与数据权限](docs/RIDGE_REFERENCE_METHOD.md) · [完整结果](docs/RIDGE_REFERENCE_RESULTS.md) · [核验与分析](docs/RIDGE_REFERENCE_ANALYSIS.md)

## 运行

服务器工作树：`/home/panyushuo/projects/panyushuo/FeMLoc-R2D2-UJI`。复用原 FeMLoc 项目 Python 环境，运行：

```bash
env/bin/python -m scripts.check_ridge_reference
env/bin/python -u -m scripts.run_ridge_reference --output outputs/new_run
```

输出目录必须尚不存在。一次运行顺序完成历史开发选择、十源楼层重训、最终评价与结果报告。自动值守使用实际进程、日志及阶段完成标志，不自动追加训练。

依据 [R2-D2 原文](https://arxiv.org/abs/1805.08136)及[作者代码](https://github.com/bertinetto/r2d2)的可微求解思想改造 UJI 坐标回归；不是其原图像分类实验的精确复现。设备、用户和时间没有被隔离，目标楼层已有历史查看记录。本轮仍是探索性实验。
