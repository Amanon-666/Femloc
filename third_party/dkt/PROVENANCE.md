# 上游来源

Repository: https://github.com/BayesWatch/deep-kernel-transfer
Commit: 61d95d6ab783be679c09803a33e3c2302287cbc4
Authors: Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, Michael O'Boyle, Amos Storkey.
Paper: Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels, NeurIPS 2020.

`DKT_regression.py`对应上游`methods/DKT_regression.py`；`train_DKT.py`对应上游`sines/train_DKT.py`，均为完整原样快照。
运行只用其中两个类：ExactGPLayer和Feature，机械提取到dkt/upstream_classes.py。测试逐AST比对，原始归属保留。
UJI适配实现见dkt/model.py、data.py、run.py；不能把这些数据和任务定义归给原作者。
