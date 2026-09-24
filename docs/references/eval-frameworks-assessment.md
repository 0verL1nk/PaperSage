# 测评框架适配性评估（2026-08-23）

裁决：**保持手搓 harness + 定向借鉴（立即执行）；Inspect AI 作 runner 的 PoC 列为观察项**。
来源：inspect.aisi.org.uk 官方文档、各候选仓库/公告（详见调研记录）。

## 需求矩阵得分（满分 20；前六行为硬门槛）

| 需求 | Inspect AI | DeepEval | promptfoo | Phoenix | Langfuse | Weave | Braintrust |
|---|---|---|---|---|---|---|---|
| Python-first / local-first / 进程内 agent | 2/2/2 | 2/2/2 | 2/2/1 | 2/1/2 | 2/1/1 | 2/0/1 | 2/0/1 |
| LLM-judge / 确定性检查并存 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| pass^k 原生 | 2 | 1 | 1 | 1 | 0 | 1 | 1 |
| 逐 case JSON / Windows / 许可 / 活跃 | 2/1/2/2 | 1/2/2/2 | 2/2/2/2 | 1/1/1/2 | 1/1/1/2 | 1/2/0/2 | 1/2/0/2 |
| **总分** | **19** | **18** | **17** | **15** | **13** | **13** | **13** |

排除项：OpenAI Evals（2026-11-30 关停，官方推荐迁 promptfoo）；lm-evaluation-harness（静态基准，
不支持 agent 任务）；观测平台类（评测数据住进平台，与 local-first + 自包含 JSONL/HTML 链路冲突）；
promptfoo（2026-03 被 OpenAI 收购，中立性顾虑；Python provider 走子进程）。

## Inspect AI：唯一值得考虑的 runner（MIT，UK AISI）

原生覆盖我们最贵的三件基础设施：`--epochs-reducer pass_k_{k}`（pass^k）、任意 async Python
solver + `agent_bridge()`（进程内接现有 LangChain agent）、逐样本本地 JSON 日志 + `inspect view`。
硬伤：**Windows 无官方支持**（纯 Python wheel 可装，已知原生 issue #1754/#4767/#3574，官方建议
WSL2）。全量替换成本 10–15 人日（前端/报告需重写），不做。

### PoC 触发条件（满足任一，投入 1–2 人日验证 Windows 原生 + DashScope judge + 全链路）

1. 评测规模扩到 50+ cases 或 k>3 常态化；
2. 需要多模型对比矩阵与 CI 集成；
3. Inspect 的 Windows 支持转官方；
4. 需要"事后重评分"工作流。

## Tax AI 对照注记（2026-09-01，随 research-feedback-loop 落地）

OpenAI Tax AI（2026-07《building-self-improving-tax-agents-with-codex》）的核心路径是
"生产失败信号 → 审查行归并 → 人审 → 评测目标 → 修复 → 回归"，与我们保持手搓 harness 的
裁决一致：它证明补齐的不是框架，而是**语料来源**。PaperSage 侧对应关系：

- 生产信号捕获：`agent/feedback/`（确定性规则：correction_followup / mode_switch_reask /
  evidence_gap，复用 durable 事件 + worker 模式，事件只存 digest 与短预览）；
- 发现归并：`feedback_events` 按项目/文档/信号类型 GROUP BY（≥2 次才成为发现）；
- 人审转评测：开发态评测页"反馈发现"区一键导出 JSONL 草稿（`origin: production-finding`
  溯源），操作者审核后手工并入 fixture——与 Tax AI 一样保留人工关卡，不做全自动入库；
- 分层统计：评测报告 `origin_breakdown` 区分自写与生产用例通过率
  （见 [QUALITY_SCORE](../QUALITY_SCORE.md) 的"生产来源用例占比"指标位）。

对照结论：Tax AI 的循环后三段（评测基础设施、追踪、回归基线）本仓库已有，本变更补第一段；
框架替换（Inspect AI PoC）触发条件不变。

## 立即借鉴（2–4 人日，已登记 eval 变更 §8）

1. **两段式评测（收益最大）**：先落原始轨迹，再离线跑 judge——judge prompt 迭代不再重跑昂贵
   的 agent 执行（对应 inspect 的 scoring/execution 分离）。
2. **Reducer 家族**：mean / at_least_{n} / pass_at_{k} / pass_k_{k} 显式语义，独立成聚合层。
3. **错误预算**：fail_on_error 的 bool/比例/数量三级声明 + 重试历史落样本记录；注意其告诫
   "重试引入分布偏移，应对比重试/未重试样本"。
### 已裁决的 harness A/B（方法论存档）

| 假设 | 判定 | 数据 |
|---|---|---|
| 工具结果内渐进提示（nudge） | **否决默认开** | 双臂 pass²：6/19(关) vs 5/19(开)，拖垮回归层 |
| 证据分层注入（P2） | **否决默认开** | 双臂 pass²：5/19(关) vs 2/19(开)，final 37%→26%；分层救弱指令模型（M2.5 类）、伤强指令模型（M3 类） |

"4. Backlog：人工评分修正留痕（author/reason）、日志 header 索引 + 惰性读取、断点续跑保留已完成样本。
