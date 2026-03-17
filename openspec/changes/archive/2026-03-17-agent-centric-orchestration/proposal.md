## Why

当前orchestration架构使用policy_engine在执行前自动决策是否启用plan/team模式，导致agent被动执行系统决策，缺乏自主性和灵活性。我们需要将控制权交还给agent，让它根据任务需求自主决策使用何种模式。

## What Changes

- **移除policy_engine自动决策**：删除orchestrator中的`intercept_policy()`调用，不再在执行前强制决策
- **创建OrchestrationMiddleware**：新增middleware在适当时机插入引导提示，建议agent使用特定工具
- **Plan由Leader自主生成**：提供CRUD工具让Leader管理自己的策略文档（create_plan, read_plan, update_plan, delete_plan），Plan内容由Leader根据上下文自己撰写，不使用旧的planning_service
- **Todolist通过Middleware提供**：创建TodoListMiddleware，注入todolist管理工具（write_todos, read_todos, update_todo, complete_todo）
- **Team作为Tool**：将`run_team_tasks`封装为activate_team_mode工具
- **简化Orchestrator**：移除mode检测循环，改为单次agent调用，处理tool调用结果

## Capabilities

### New Capabilities
- `orchestration-middleware`: Middleware that analyzes context and injects guidance prompts to suggest using plan/team tools
- `plan-management`: Tools for managing high-level execution plans (create_plan, read_plan, update_plan, delete_plan)
- `todolist-middleware`: Middleware that injects todolist management tools (write_todos, read_todos, update_todo, complete_todo)
- `team-activation`: Tool for activating multi-agent collaboration mode (activate_team_mode)

### Modified Capabilities
- `orchestrator-core`: Simplify orchestrator to remove policy_engine dependency and mode detection loop

## Impact

**Modified Files:**
- `agent/orchestration/orchestrator.py` - 移除policy_engine调用，简化执行流程
- `agent/application/turn_engine.py` - 更新orchestrator调用方式
- `agent/tools/local_ops.py` - 移除write_todo和edit_todo工具定义

**Removed Files:**
- `agent/orchestration/policy_engine.py` - 完全移除自动决策引擎

**New Files:**
- `agent/middlewares/orchestration.py` - OrchestrationMiddleware实现
- `agent/middlewares/todolist.py` - 集成LangChain TodoListMiddleware
- `agent/tools/plan_tools.py` - Plan管理工具（create_plan, read_plan, update_plan, delete_plan）
- `agent/tools/team_tools.py` - activate_team_mode工具

**Dependencies:**
- 依赖现有的progressive_tool_disclosure middleware
- 依赖planning_service和team_runtime的现有实现
- 依赖LangChain的TodoListMiddleware（需要安装langchain包）
