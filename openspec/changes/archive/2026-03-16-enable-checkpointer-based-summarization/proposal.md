## Why

当前项目使用外部管理消息列表的方式（`st.session_state.agent_messages`），每次调用agent时传入完整历史。这与LangChain的`SummarizationMiddleware`设计不兼容，导致middleware压缩失效。需要改为有状态模式，使用checkpointer管理消息历史，让自动压缩机制正常工作。

## What Changes

- 添加持久化checkpointer实现（SqliteSaver）用于保存agent状态
- 修改agent调用方式：只传入新消息，历史由checkpointer管理
- 移除旧的手动压缩逻辑（`apply_auto_compact`）
- 从checkpointer同步压缩后的消息状态到UI层

## Capabilities

### New Capabilities

- `persistent-checkpointer`: 实现基于SQLite的持久化checkpointer，用于保存和恢复agent的对话状态
- `stateful-message-flow`: 实现有状态的消息流转机制，只传入新消息，历史由checkpointer加载

### Modified Capabilities

<!-- 无现有capability需要修改 -->

## Impact

**代码影响**:
- `agent/runtime_agent.py` - 需要传入checkpointer实例
- `agent/paper_agent.py` - 创建并传入checkpointer
- `agent/stream.py` - 修改消息传递方式
- `ui/agent_center_page.py` - 移除手动压缩调用
- `agent/application/agent_center/conversation_state.py` - 移除`apply_auto_compact`函数

**数据影响**:
- 需要SQLite数据库文件存储checkpointer状态
- 现有的`st.session_state.agent_messages`不再作为唯一的消息来源

**行为变更**:
- **BREAKING**: 消息历史管理方式改变，从外部管理改为checkpointer管理
- 压缩机制从手动触发改为middleware自动触发
