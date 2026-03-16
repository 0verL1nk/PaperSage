## ADDED Requirements

### Requirement: 使用 Node-style Hooks 追踪执行阶段
系统 SHALL 实现 TraceMiddleware 类，继承自 AgentMiddleware，使用 before_model 和 after_model hooks 自动记录 agent 执行阶段。

#### Scenario: before_model hook 提取当前阶段
- **WHEN** before_model hook 被调用
- **THEN** middleware 从 state["messages"] 中提取最新的 performative 字段，调用 phase_label_from_performative 转换为中文标签，并追加到 state 的 trace_labels 列表中

#### Scenario: after_model hook 记录模型响应
- **WHEN** after_model hook 被调用
- **THEN** middleware 从新增的 AIMessage 中提取 performative，记录到 trace_labels 列表中

### Requirement: 保持现有 Trace API
系统 SHALL 保持 `agent/domain/trace.py` 中定义的 `PHASE_BY_PERFORMATIVE`、`phase_label_from_performative` 和 `phase_summary` 函数的公共 API 不变。

#### Scenario: 现有代码继续工作
- **WHEN** 其他模块调用 `phase_label_from_performative("plan")`
- **THEN** 返回 "规划"，与之前行为一致

### Requirement: 扩展 AgentState Schema
系统 SHALL 定义自定义的 AgentState schema，添加 trace_labels 字段（类型为 NotRequired[list[str]]）用于存储执行阶段标签。

#### Scenario: 初始化 trace_labels
- **WHEN** agent 开始执行且 state 中不存在 trace_labels
- **THEN** middleware 初始化 trace_labels 为空列表

#### Scenario: 追加阶段标签
- **WHEN** middleware 提取到新的 performative
- **THEN** 将对应的中文标签追加到 state["trace_labels"] 列表中

### Requirement: 使用 after_agent Hook 生成阶段摘要
系统 SHALL 实现 after_agent hook，在 agent 执行完成后调用 phase_summary 函数生成去重的阶段路径摘要。

#### Scenario: 生成去重的阶段路径
- **WHEN** after_agent hook 被调用且 state["trace_labels"] 包含多个标签
- **THEN** 调用 phase_summary(state["trace_labels"]) 生成摘要（如 "接收请求 -> 规划 -> 输出最终答案"），并存储到 state["trace_summary"] 中

#### Scenario: 空标签列表处理
- **WHEN** state["trace_labels"] 为空
- **THEN** phase_summary 返回 "无"
