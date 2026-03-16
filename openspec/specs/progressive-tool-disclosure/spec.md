## ADDED Requirements

### Requirement: 符合 LangChain Middleware 标准
系统 SHALL 重构 ProgressiveToolDisclosureMiddleware，使其继承自 LangChain 的 AgentMiddleware 基类，并使用标准的 wrap_model_call hook。

#### Scenario: 使用 wrap_model_call 过滤工具
- **WHEN** agent 准备调用模型
- **THEN** middleware 通过 wrap_model_call hook 拦截请求，根据激活历史过滤工具列表

### Requirement: 支持 Fixed 和 Lazy 工具可见性
系统 SHALL 支持两种工具可见性模式：fixed（始终可见）和 lazy（按需激活）。

#### Scenario: Fixed 工具始终可见
- **WHEN** 工具标记为 fixed 可见性
- **THEN** 该工具在所有模型调用中都可见

#### Scenario: Lazy 工具按需激活
- **WHEN** 工具标记为 lazy 可见性且未被激活
- **THEN** 该工具在模型调用中不可见

#### Scenario: 通过 search_tools 激活 Lazy 工具
- **WHEN** agent 调用 search_tools 工具并指定工具名称
- **THEN** 该 lazy 工具在后续模型调用中变为可见

### Requirement: 从消息历史提取激活记录
系统 SHALL 从 agent 的消息历史中提取已激活的工具名称，用于判断哪些 lazy 工具应该可见。

#### Scenario: 解析 tool_calls 中的激活记录
- **WHEN** 消息历史中包含 search_tools 的 tool_call
- **THEN** middleware 正确提取被激活的工具名称列表

### Requirement: 环境变量控制开关
系统 SHALL 通过环境变量 AGENT_PROGRESSIVE_TOOL_DISCLOSURE 控制是否启用渐进式工具披露功能。

#### Scenario: 禁用渐进式工具披露
- **WHEN** AGENT_PROGRESSIVE_TOOL_DISCLOSURE 设置为 false
- **THEN** 所有工具始终可见，不进行过滤
