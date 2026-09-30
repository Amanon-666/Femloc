# 代码取舍与来源

## 被接续的原仓库

Amanon-666/Femloc 的 signal-calibrated-map，提交 ab1a60b。
本轮在 tpm-gufu-unlabeled 上工作；本机和 ray-web 使用同一套提交，原工作树未修改。

直接复用：
- models/cross_floor.py：跨楼层先验拟合与地图期望；未改。
- scripts/evaluate_transfer_map.py:calibrate：锚点校正；未改。
- scripts/evaluate_scm_tobit.py：只将权重计算提取成 tobit_weights；原输出逐元素等价测试通过。
- scipy.linalg.eigh / solve(assume_a='pos')：现成 LAPACK 特征分解和正定求解，未自写数值线性代数。

新增 models/unlabeled_map.py 是本轮推导的闭式目标实现，不能称为 GUFU 官方实现。

## GUFU 官方仓库实查

- URL：https://github.com/khchiuac/GUFU
- commit：e1f14c73165daceead6992b5047507704fb0d7c5
- 本机只读参考副本：/Users/Admin/Desktop/project/迁移学习/参考代码/GUFU
- 参考位置：util.py:k_virtual、sage_ve.py、pred.py；论文章节 §5.2。

阻止直接复用完整训练入口的静态证据：
1. sage_ve.py 导入 `utils` 和 `config`，仓库只提供 util.py，没有对应的配置模块。
2. `class LossFunc(torch.nn.Module, sigma)` 中 sigma 没有顶层定义。
3. pred.py 对 SAGE_VE(...)[1] 进行索引，所示类没有 __getitem__；随后 run 调用也缺少必需的 model。
4. util.py:k_virtual 的自环过滤比较的是邻居排名 k 与节点 id cur_id，而非查询节点 i1；重复相似度用 list.index 也会重复选相同邻居。
5. pred.py 的部分文件路径保留未填的 {}，输出列表直接 f.write，不是完整可靠的端到端入口。

这些是所查提交的静态事实，未声称已经复现/测量 GUFU 的论文成绩，也不能据此否定论文方法。为避免将修补作者仓库变成新的独立项目，本轮借鉴它的图对应与指纹重构机制，复用当前项目可靠的地图与匹配代码。没有把缺失部分随意补成一个网络再叫 GUFU。
