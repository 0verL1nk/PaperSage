## 1. 基础设施准备

- [x] 1.1 添加 langgraph-checkpoint-sqlite 依赖到项目
- [x] 1.2 在数据库 schema 中添加 thread_id 字段到 session 表
- [x] 1.3 实现 get_or_create_thread_id 函数(查询或创建 thread_id)
- [x] 1.4 创建 SqliteSaver 实例并配置数据库路径
- [x] 1.5 修改 create_runtime_agent 调用,传入 checkpointer 参数

## 2. Thread ID 持久化

- [x] 2.1 在 agent/adapters.py 中添加 thread_id 查询函数
- [x] 2.2 在 agent/adapters.py 中添加 thread_id 保存函数
- [x] 2.3 修改 agent/paper_agent.py 中的 thread_id 生成逻辑
- [x] 2.4 确保首次创建 session 时保存 thread_id
- [x] 2.5 确保后续访问时复用已保存的 thread_id

## 3. 消息流改造

- [x] 3.1 修改 agent 调用方式,只传入新消息而非完整历史
- [x] 3.2 移除 ui/agent_center_page.py 中的 apply_auto_compact_fn 调用
- [x] 3.3 移除 build_turn_execution_context 中的 compact_summary 参数传递
- [x] 3.4 实现从 checkpointer 同步消息状态到 UI 的逻辑
- [x] 3.5 修改 agent.stream 调用,传入 config 参数包含 thread_id

## 4. 测试验证

- [x] 4.1 测试新会话创建和 thread_id 生成
- [x] 4.2 测试消息发送和 checkpointer 状态保存
- [x] 4.3 测试程序重启后会话恢复功能
- [x] 4.4 测试消息自动压缩触发(发送足够多消息)
- [x] 4.5 验证 UI 显示的消息与 checkpointer 状态一致

## 5. 清理旧代码

- [x] 5.1 移除 agent/application/agent_center/conversation_state.py 中的 apply_auto_compact 函数
- [x] 5.2 移除 agent/context_governance.py 中的 inject_compact_summary 调用
- [x] 5.3 移除 ui/agent_center/state.py 中的 apply_auto_compact_runtime 函数
- [x] 5.4 清理相关的 import 语句
- [x] 5.5 更新相关文档说明新的消息管理机制

