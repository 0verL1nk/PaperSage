## Context

当前项目使用"外部消息管理"模式:
- UI层维护完整的消息列表(`st.session_state.agent_messages`)
- 每次调用agent时传入完整的消息历史
- 手动触发压缩逻辑(`apply_auto_compact`)

这种模式与LangChain的`SummarizationMiddleware`设计不兼容:
- Middleware期望只传入新消息,历史由checkpointer自动加载
- Middleware的压缩结果保存在checkpointer中,外部无法感知
- 导致压缩失效,消息重复传递

需要迁移到"checkpointer-based"有状态模式,让middleware正常工作。

## Goals / Non-Goals

**Goals:**
- 启用`SummarizationMiddleware`自动压缩机制
- 支持程序重启后恢复agent会话状态
- 移除手动压缩逻辑,简化代码
- 保持UI层消息显示的一致性

**Non-Goals:**
- 不改变用户可见的对话体验
- 不迁移历史会话数据(新会话使用新机制)
- 不支持跨设备会话同步

## Decisions

### Decision 1: 使用SqliteSaver作为持久化checkpointer

**选择:** SqliteSaver
**理由:**
- LangGraph官方支持,API稳定
- 本地文件存储,无需额外服务
- 支持事务和并发访问
- 轻量级,适合单机部署

**备选方案:**
- InMemorySaver: 程序重启后丢失状态,不满足恢复需求
- PostgresSaver: 需要额外数据库服务,过重
- RedisSaver: 需要Redis服务,增加部署复杂度

### Decision 2: Thread ID持久化策略

**选择:** 在现有数据库中添加thread_id字段
**理由:**
- 复用现有的session表结构
- 通过(project_uid, session_uid)唯一确定thread_id
- 首次创建session时生成thread_id并保存
- 后续访问时查询并复用

**实现:**
```python
# 伪代码
def get_or_create_thread_id(project_uid, session_uid):
    thread_id = db.query_thread_id(project_uid, session_uid)
    if not thread_id:
        thread_id = f"thread-{uuid4().hex}"
        db.save_thread_id(project_uid, session_uid, thread_id)
    return thread_id
```

### Decision 3: 消息传递模式改造

**选择:** 只传入新消息,历史由checkpointer加载
**理由:**
- 符合LangGraph有状态agent的设计
- Middleware能正确压缩和持久化
- 减少每次调用的数据传输量

**改造点:**
- `agent.stream({"messages": [new_message]}, config={"configurable": {"thread_id": thread_id}})`
- 移除外部的消息列表拼接逻辑
- Agent执行后从checkpointer同步最新状态到UI

### Decision 4: UI消息状态同步策略

**选择:** Agent执行后从checkpointer读取最新消息状态
**理由:**
- Checkpointer是唯一的真实来源(Single Source of Truth)
- 确保UI显示的消息与agent内部状态一致
- 支持压缩后的消息正确显示

**实现:**
```python
# 伪代码
result = agent.invoke({"messages": [new_message]}, config)
# 从checkpointer读取最新状态
latest_state = checkpointer.get(config)
st.session_state.agent_messages = latest_state["messages"]
```

## Risks / Trade-offs

### Risk 1: 破坏性变更 → 渐进式迁移

**风险:** 现有会话无法直接迁移到新机制
**缓解:**
- 新会话使用新机制,旧会话保持原有逻辑
- 通过feature flag控制切换
- 提供数据迁移脚本(可选)

### Risk 2: Checkpointer数据库文件管理 → 配置化路径

**风险:** SQLite文件位置不当可能导致权限或备份问题
**缓解:**
- 默认路径:`./data/checkpoints.db`
- 支持环境变量配置:`CHECKPOINTER_DB_PATH`
- 文档说明备份策略

### Risk 3: 并发访问冲突 → SQLite WAL模式

**风险:** 多进程同时访问可能导致锁冲突
**缓解:**
- 启用SQLite WAL模式提升并发性能
- 单用户场景下风险较低
- 未来可升级到PostgreSQL

### Risk 4: UI状态不一致 → 强制同步

**风险:** Checkpointer状态与UI显示不同步
**缓解:**
- 每次agent执行后强制从checkpointer同步
- 移除外部消息列表的独立修改
- 添加状态一致性检查

## Migration Plan

### Phase 1: 基础设施准备
1. 添加SqliteSaver依赖
2. 实现thread_id持久化逻辑
3. 创建checkpointer实例并传递给agent

### Phase 2: 消息流改造
1. 修改agent调用方式,只传入新消息
2. 移除`apply_auto_compact`调用
3. 实现checkpointer状态同步到UI

### Phase 3: 测试验证
1. 验证新会话的压缩机制
2. 验证程序重启后的会话恢复
3. 验证UI消息显示正确性

### Phase 4: 清理旧代码
1. 移除`apply_auto_compact`函数
2. 移除`inject_compact_summary`逻辑
3. 更新相关文档

### Rollback策略
- 保留旧代码分支
- Feature flag快速切换回旧逻辑
- Checkpointer数据库独立,不影响现有数据

## Open Questions

无待解决问题。设计已明确。

