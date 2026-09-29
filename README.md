# MetaLoc 式 UJI 少样本跨楼层定位

这是 `Amanon-666/Femloc` 的 `metaloc-few-shot` 分支。`main` 保留独立 FeMLoc EXP1 重实现及原始结果；本分支只读共用 UJI 原始 CSV，使用新的 episode、共享 RSSI 回归网络和训练入口。

- 每个目标楼层只给 **10 个位置 × 每点 3 条扫描 = 30 条标注扫描**。
- 十个历史楼层组成源域 episode；三个目标楼层为 B0F3、B1F3、B2F4。
- 比较 MetaLoc 式 MAML 初始化（MI）、普通源监督初始化（TL）和随机初始化（RI）；三者共享目标划分、网络、适应学习率和 0–10 步预算。
- 主指标为目标楼层**未采集位置**的二维定位 MDE；已采集位置上的新扫描另列为辅助指标。

[方法与数据权限](docs/METALOC_FEWSHOT_DESIGN.md) · [完整结果](docs/METALOC_FEWSHOT_RESULTS.md)

## 运行

服务器工作树：`/home/panyushuo/projects/panyushuo/FeMLoc-MetaLoc-few-shot`。安装依赖后：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 env/bin/python -m scripts.run_metaloc \
  --config configs/metaloc_fewshot.json --output outputs/new_run
```

输出目录必须尚不存在。完整运行保存每个随机种子的目标划分、源训练日志、源模型、逐步误差曲线和完成标记，并生成 `docs/METALOC_FEWSHOT_RESULTS.md`。

MetaLoc [论文](https://arxiv.org/html/2211.04258v5)及[作者公开代码](https://github.com/StatFusion/MetaLoc/tree/2a3f7ae6dffcf7ebfc72a82b8091cc23db1aead9)提供 episode、内外层训练和逐步测试的参照。作者的真实场地实验以 CSI 图像做位置分类；这里采用 UJI RSSI 坐标回归，并将源域 Query 放到未采集位置，以对齐本项目的主目标。因此这些数值不能与原文的 CSI 精度直接比较。
