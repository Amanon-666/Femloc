# SCM 机制审计分支

- 本工作树只属于 `scm-field-ranking-audit`，不修改主分支或其它工作树。
- `docs/FIELD_RANKING_AUDIT_DESIGN.md` 记录审计问题与数据权限，`docs/FIELD_RANKING_AUDIT_RESULTS.md` 记录实测结果。
- 固定 SCM v1 和 WKNN；审计不得根据 B0F3/B1F3/B2F4 的 Query 或 validation 结果调整模型、参数、阈值或数据划分。
- 七对历史相邻楼层来自 trainingData，目标 Support 只取既有 manifest。Query 坐标仅用于离线评价场误差和定位误差。
- 服务器命令经 `ssh lab-server 'bash -s'`；GitHub 推送只经本机 SSH 临时隧道，凭证不得进入源码、日志或提交。
