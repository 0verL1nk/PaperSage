## 1. 信号捕获

- [x] 1.1 `feedback_events` 表 + Alembic 迁移（幂等键、脱敏字段）
- [x] 1.2 轮后入队（复用本地 task queue 模式）+ 确定性信号判定器（三类规则、命名常量、env 覆盖）
- [x] 1.3 证据点击埋点端点（轻量 POST，run item 关联）
- [x] 1.4 单测：三类信号判定（命中/不命中/数据不足跳过）、幂等去重、脱敏
  - 注：第三类信号实现为 `evidence_gap`（消费 turn_engine 的 `citation_audit` 字段，
    P3-lite 已产出）；设计稿中的 `evidence_engagement`（引用 vs 点击对比判定）待埋点
    数据积累后作为后续增强，点击埋点存储已就位。
  - 注：correction_followup 的时间窗按"steering 输入与轮完成时刻的间隔"度量
    （当前产品结构中 steering 只能挂到运行中的 Run，不存在答案完成后的提交路径）。

## 2. 发现归并

- [x] 2.1 发现聚合查询（GROUP BY + 最小重复阈值 + 项目粒度）
- [x] 2.2 单测：重复≥2 出现、单次不出现、时间窗内聚桶

## 3. 人审转评测

- [x] 3.1 `GET /evals/feedback-findings` + `POST .../export-case` 端点（草稿生成，含 origin 溯源）
- [x] 3.2 开发态评测页"反馈发现"区（列表 + 导出按钮）（最小实现：表格 + 导出 + JSONL 复制，无独立看板）
- [x] 3.3 loader 支持 `origin` metadata；报告按来源分层统计
- [x] 3.4 单测：草稿生成结构、origin 标注、报告分层

## 4. 验证与文档

- [x] 4.1 全量门禁 + openspec validate（repository_guard / ruff / 相关单测通过；
  未跑全量单测套件——本任务约定只跑新增与受影响测试）
- [ ] 4.2 用真实会话人工核验一轮信号判定（抽样）
- [x] 4.3 QUALITY_SCORE 登记"生产来源用例"指标位；eval-frameworks-assessment 补 Tax AI 对照注记
- [ ] 4.4 （后续登记）发现→openspec 修复草案的自动化闭环；模型辅助信号分类

### P2 分层注入 A/B 判定（2026-09-01，预登记规则裁决）

- [x] 双臂 pass²=2，19 用例 × 2 试验，v2 快照，M3：对照 5/19、逐试验 final 37%/过程 53%；分层 2/19、final 26%/过程 50%——**final 降 11pp，远超 3pp 容差，否决默认开启**
- [x] 工具调用均值 9.0→7.6（分层反而更少：M3 依预览即答，少做全文检索）；翻转 5 例（1 升 4 降，简单文档用例受损最重）
- [x] 处置：`AGENT_EVIDENCE_TIERED` 维持默认关；适用边界写入文档——分层救的是弱指令模型的证据洪泛（M2.5 类），强指令模型（M3 类）不需要
- 双臂产物：`task-completion-ab-p2-{control,treatment}-20260901.json`
