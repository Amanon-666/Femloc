# uji-explore：新旧分支都保留，包括负结果

| 分支 / 场景 | 方法 | 标签与聚合 | 同日 / m | 官方 / m |
|---|---|---|---|---|
| main：少样本 pilot | CMANP | 10 位置×1；3 层等权 | 25.50 | 23.17 |
| main：少样本 pilot | Support-WKNN | 10 位置×1；3 层等权 | 19.34 | 18.93 |
| main：少样本 pilot | MLP + 50 步微调 | 10 位置×1；3 层等权 | 16.63 | 未运行 |
| research：常规监督 | GConvLoc | 完整监督；1111 条加权 | — | 7.61 |
| research：留出 FLOOR=3 | GConvLoc | 0 目标标签；扫描加权 | 12.80 | 9.78 |
| research：留出 FLOOR=3 | GConvLoc | 0 目标标签；3 层等权 | 13.44 | 10.38 |
| research：4 次楼层留出 | JPRL | 0 目标标签；4 任务等权 | 31.91 | 99.17 |
| research：4 次楼层留出 | 同骨干 ERM | 同 JPRL 协议 | 31.91 | 99.10 |
| research：尚未实现 | UE-GLoc | 无运行结果 | 未运行 | 未运行 |

research 指 research/gconvloc-jprl-uegloc-20260930。main 仅 1 模型种子、2 支持种子；新分支也为单种子。
main 的官方测试还排除与支持集同位置的样本；跨场景数字不应直接作算法排名。
