## ADDED Requirements

### Requirement: 只传入新消息

系统SHALL在调用agent时只传入新消息，历史消息由checkpointer自动加载。

#### Scenario: 用户发送新消息
- **WHEN** 用户输入新的问题并提交时
- **THEN** 系统只将新消息传递给agent，不传入完整的消息历史

#### Scenario: Checkpointer加载历史
- **WHEN** Agent开始处理请求时
- **THEN** LangGraph自动从checkpointer加载该thread的历史消息并追加新消息

#### Scenario: Middleware自动压缩
- **WHEN** 历史消息超过阈值时
- **THEN** SummarizationMiddleware自动压缩旧消息，压缩后的状态保存到checkpointer

### Requirement: 移除手动压缩逻辑

系统SHALL移除旧的手动压缩逻辑，完全依赖middleware自动压缩。

#### Scenario: 移除apply_auto_compact调用
- **WHEN** 用户发送消息时
- **THEN** 系统不再调用apply_auto_compact函数进行手动压缩

#### Scenario: 移除inject_compact_summary
- **WHEN** 构建提示词时
- **THEN** 系统不再手动注入压缩摘要到提示词中

### Requirement: 消息状态同步

系统SHALL在agent执行后从checkpointer同步最新的消息状态到UI层。

#### Scenario: Agent执行后同步状态
- **WHEN** Agent完成一轮对话后
- **THEN** 系统从checkpointer读取最新的消息状态（包括压缩后的消息）

#### Scenario: UI显示压缩后的消息
- **WHEN** 用户查看对话历史时
- **THEN** 系统显示checkpointer中的消息状态，而不是外部维护的完整列表

### Requirement: 配置传递

系统SHALL在每次agent调用时传递正确的配置，包括thread_id。

#### Scenario: 传递thread_id
- **WHEN** 调用agent.stream或agent.invoke时
- **THEN** 系统在config参数中传递{"configurable": {"thread_id": thread_id}}

#### Scenario: 配置一致性
- **WHEN** 同一会话的多次调用时
- **THEN** 系统使用相同的thread_id，确保状态连续性
