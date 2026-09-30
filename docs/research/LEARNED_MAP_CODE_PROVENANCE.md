# 学习型地图适应：思想与代码对应

## 版本

- 继承 TPM/GUFU 分支：`42f5484`。
- 推导与预注册提交：`e8467b8`，早于模型实现及训练。
- 首轮训练实现提交：`f84647b`。完整代码与数据哈希另存实际运行的 `outputs/learned_map_v1/provenance.json`。
- 本机及 ray-web 使用 `tpm-learned-map`，独立工作树 `FeMLoc-TPM-Learned-Map`。未改旧分支。

## 直接复用和改写

| 本轮代码 | 实际来源 | 改动 |
|---|---|---|
| 去重后旧地图统计、观测常数 | 原仓库 `models/radio_map.py:cells,fit_observation` | 原文件不改；每折仅传入源建筑 |
| 跨楼层参数和初始地图 | 原仓库 `models/cross_floor.py:fit_transfer,prior_map` | 原文件不改；新驱动明确限制源楼层与源转移对 |
| 锚点聚合与几何插值 | 原仓库 `scripts/evaluate_signal_calibrated_map.py:position_map,lookup` | 直接调用；特征也采用同一插值 |
| 空间GP条件化 | 同文件 `gp_mean` 的数学实现 | 改为 PyTorch 按 AP 批量求解，加入可训练的特征核因子；恒定因子严格回退 |
| Tobit匹配 | `scripts/evaluate_scm_tobit.py:tobit_weights,tobit_match` | 移植为 PyTorch 可微版本；中心化坐标、float64；对照原实现测试 |
| 正定系统求解、自动微分 | PyTorch `torch.linalg.solve`、autograd | 调用库，不另造数值求解器 |

## 外部论文与仓库

1. **DKT**：学习跨任务共享特征核、目标任务解析条件化的依据。查阅作者仓库 https://github.com/BayesWatch/deep-kernel-transfer 的 `61d95d6ab783be679c09803a33e3c2302287cbc4`，尤其 `methods/DKT_regression.py`。其 `test_loop` 无需目标优化器，`ExactGPLayer` 在学习特征上建核。原训练为GP边缘似然，本轮训练为定位Query损失，二者不能混称。同仓库README说明旧GPyTorch版本兼容性限制；没有整包复制该旧依赖栈。
2. **MetaFun**：函数/场更新的思想，非逐字移植。https://proceedings.mlr.press/v119/xu20i.html 。作者TensorFlow/Sonnet实现并未加入本仓库。
3. **TE-TNP**：空间条件函数的平移等变性。https://arxiv.org/abs/2406.12409 。本轮不用Transformer，只用相对位置距离和无绝对位置输入的AP共享特征来满足对应对称性。
4. **GUFU**：无线地图重构及关系建模的前一轮来源。https://arxiv.org/abs/2507.11038 。本轮不使用无标签池，不是GUFU复现；原仓库静态检查继续见 `CODE_PROVENANCE.md`。

本轮的具体组合——7维旧图/先验描述符、196参数共享编码器、空间与特征RBF乘积、Support信号残差条件化、定位Query外层损失——是本项目实现的待检验方案，不冒称来自某篇论文的原公式，也未证明方法新颖性。

## 数学与实现边界

- 特征RBF只抑制原空间核关联，尚不能学习跨AP残差混合或任意远距离增益。
- 平移/排列测试使用没有距离并列的测试坐标。继承的3近邻插值在距离并列且截断邻居集合时，可能随节点排列发生选择差异；这不是全体输入上严格排列不变的证明。
- 残差拟合处理编码观测均值。Tobit匹配沿用既有能量，并不使整个组合自动成为严格物理概率模型。
- 监督预训练临时线性头只在源端训练，目标适应使用同一编码器、同一求解器；这一特定控制不涵盖所有可行预训练。
