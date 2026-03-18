# 实现任务

## Phase 1: 清理旧代码

- [x] 删除 `agent/a2a/` 目录
- [x] 删除 `agent/orchestration/` 目录
- [x] 清理 `agent/__init__.py` 中的 A2A 导出
- [ ] 简化 `agent/application/turn_engine.py`（移除 orchestrator 调用）
- [x] 删除相关测试文件

## Phase 2: 实现 SubAgent Middleware

- [ ] 创建 `agent/subagent/loader.py`（配置加载器）
- [ ] 创建 `agent/middlewares/subagent.py`（中间件）
- [ ] 创建示例配置：researcher/reviewer/writer
- [ ] 在 `builder.py` 中注册中间件
- [ ] 编写单元测试

## Phase 3: 实现 Team 模式

- [ ] 创建 `agent/team/runtime.py`（核心运行时）
  - [ ] 实现 `TeamAgent` 数据类
  - [ ] 实现 `TeamRuntime` 类（ThreadPoolExecutor + Lock）
  - [ ] 实现文件存储机制（`.claude/team/{team_id}/{agent_id}.result.txt`）
  - [ ] 实现 Trace 事件记录
  - [ ] 实现 Logging
- [ ] 创建 `agent/tools/team.py`（5个核心工具）
  - [ ] `spawn_agent(name, role, model)` - 创建 agent
  - [ ] `send_message(agent_id, message)` - 异步发送任务
  - [ ] `list_agents()` - 列出状态概览
  - [ ] `get_agent_result(agent_id)` - 从文件读取结果
  - [ ] `close_agent(agent_id)` - 关闭 agent
- [ ] 创建 `agent/middlewares/team.py`（按需注入）
- [ ] 修改 `OrchestrationMiddleware` 激活逻辑
- [ ] 编写单元测试
  - [ ] 测试 agent 生命周期
  - [ ] 测试异步执行
  - [ ] 测试文件存储
  - [ ] 测试多轮对话

## Phase 4: 增强 TodoList

- [x] 创建 `agent/domain/todo_graph.py`（拓扑排序）
- [x] 创建 `agent/middlewares/enhanced_todolist.py`
- [x] 在 `builder.py` 中替换旧中间件
- [x] 编写单元测试

## Phase 5: 集成测试

- [ ] 编写端到端测试
- [ ] 验证 SubAgent 调用
- [ ] 验证 Team 协作流程
- [ ] 验证 Todo 依赖管理
