# FeMLoc-Reproduction

依据 [FeMLoc arXiv:2405.11079v1](https://arxiv.org/html/2405.11079v1) 和 [UJIIndoorLoc官方说明](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) 进行独立重实现。

## 当前状态

论文/旧实现审计完成；仅确定部分已形成规范；影响实验语义的缺失项等待集中确认。**没有启动训练，没有新的MI/RI结果。**

- [事实审计](docs/FeMLoc_Reproduction_Audit.md)：原文、公式、数据、旧设计与未决项逐项对应。
- [确定部分的规范](docs/FeMLoc_Reproduction_Specification.md)：数据接口、模块、更新及评价权限。
- [待决策项](docs/Unresolved_Decisions.md)：六组问题、依据、选项和建议。
- [原始数据盘点](docs/UJI_RAW_AUDIT.json)：未清洗的逐楼层统计。
- [任务来源](docs/REQUEST.md)：用户附件及独立仓库覆盖说明。

## 独立性

服务器Git根目录：`/home/panyushuo/projects/panyushuo/FeMLoc-Reproduction`。
WIFI-loc历史参考commit：`6eb7a6304e5d165b9af86e28d355165b443606c5`。
本项目不fork、不import旧代码，不使用旧manifest/配置/训练环境。只读引用共享原始CSV，尚未安装模型环境。
用户选择只建立服务器仓库，当前没有GitHub remote，也没有迁移旧凭证。

## 重跑原始文件盘点

在服务器项目目录用系统Python3（仅标准库）：

```bash
python3 scripts/audit_uji.py \
  /home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc \
  --output docs/UJI_RAW_AUDIT.json
```

原始官方文件应为19937/1111行，每份529列、520WAP、13楼层。脚本不做清洗、划分或训练。盘点中AP中位数155仅针对说明的原始trainingData十楼层范围，不构成已批准latent维度。

目录按实际实现阶段增加：data/、models/、training/、evaluation/、configs/。本阶段只包含所需scripts/与docs/，不创建空算法模块或通用框架。
