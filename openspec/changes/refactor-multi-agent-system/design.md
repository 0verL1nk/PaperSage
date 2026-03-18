# 架构设计

## 整体架构

```
Agent Runtime
├── Middleware Pipeline
│   ├── TraceMiddleware
│   ├── OrchestrationMiddleware（复杂度分析）
│   ├── SubAgentMiddleware（新增，常驻）
│   ├── EnhancedTodoListMiddleware（改造）
│   ├── PlanMiddleware
│   └── ToolSelectorMiddleware
└── Application Layer
    └── turn_engine.py（简化）
```

## 核心组件

### 1. SubAgent Middleware（常驻中间件）

**职责**：提供预定义的专业 agent 调用能力

**配置格式**（agent.md）：
```markdown
---
name: researcher
model: gpt-4
description: 专注于文献检索和证据收集
---

# System Prompt
你是一个研究型 agent...
```

**工具**：
- `invoke_subagent(name: str, prompt: str) -> str`

**特点**：
- 始终可用，不依赖 Team 模式
- 动态加载 `agent/subagent/*/agent.md`
- 上下文隔离（每次调用创建独立 agent 实例）

### 2. Team 模式（按需激活）

**职责**：动态创建和管理临时 agent 团队

**激活条件**：OrchestrationMiddleware 检测到复杂任务

**工具集**：
- `spawn_agent(name: str, role: str, model: str) -> str` - 创建新 agent，返回 agent_id
- `send_message(agent_id: str, message: str) -> str` - 发送任务（异步，非阻塞）
- `list_agents() -> str` - 列出所有 agents 的状态概览（id/name/status，不含完整结果）
- `get_agent_result(agent_id: str) -> str` - 获取特定 agent 的完整执行结果
- `close_agent(agent_id: str) -> str` - 关闭 agent

**工作模式**：

1. **基本流程**：
   ```python
   # 1. 创建 agent
   agent_id = spawn_agent("researcher", "负责文献检索", "gpt-4")

   # 2. 发送任务（异步，立即返回）
   send_message(agent_id, "检索关于 LLM 的论文")

   # 3. 查看所有 agents 状态
   agents_info = list_agents()
   # 输出示例：
   # - agent-abc123: researcher (idle)
   # - agent-def456: reviewer (busy)

   # 4. 获取特定 agent 的完整结果
   result = get_agent_result(agent_id)

   # 5. 关闭 agent
   close_agent(agent_id)
   ```

2. **并行协作**：
   ```python
   # 创建多个 agents
   r1 = spawn_agent("researcher1", "文献检索", "gpt-4")
   r2 = spawn_agent("researcher2", "数据分析", "gpt-4")

   # 并行发送任务（异步，非阻塞）
   send_message(r1, "检索关于 LLM 的论文")
   send_message(r2, "分析用户数据")

   # 稍后检查状态
   agents = list_agents()  # 查看哪些完成了

   # 获取完成的 agent 结果
   papers = get_agent_result(r1)
   analysis = get_agent_result(r2)
   ```

**结果传递机制**：
- `send_message` 异步执行，立即返回（不阻塞）
- `list_agents` 只显示状态概览（id/name/status），不包含结果内容
- `get_agent_result(agent_id)` 获取特定 agent 的完整执行结果
- 结果存储到文件：`.claude/team/{team_id}/{agent_id}.result.txt`
- `TeamAgent.result_file` 存储文件路径，`get_agent_result` 从文件读取内容

**Agent 实例管理**：
- 每个 agent 实例持久化，可多次交互（保留对话历史）
- 每个 agent 有独立的 thread_id（用于 checkpointer）
- 只有 leader 调用 `close_agent` 时才销毁实例
- 支持多轮对话：spawn → send → get_result → send → get_result → close

**状态机**：
```
idle → busy → idle
  ↓
closed
```

**数据结构**：
```python
@dataclass
class TeamAgent:
    agent_id: str
    name: str
    role: str
    status: AgentStatus  # idle/busy/closed
    model: str
    system_prompt: str
    created_at: str
    last_active: str
    result_file: str | None  # 结果文件路径
    agent_instance: Any  # LangChain agent 实例
    thread_id: str  # 用于 checkpointer

class TeamRuntime:
    team_id: str
    agents: dict[str, TeamAgent]
    executor: ThreadPoolExecutor  # 异步执行线程池
    _lock: Lock  # 线程安全锁
    trace_context: TraceContext  # Trace 上下文
    on_event: Callable[[TraceEvent], None] | None  # 事件回调
```

**Trace 和 Logging**：

Trace Events（记录所有关键操作）：
```python
# spawn_agent
build_trace_event(
    sender="leader",
    receiver=agent_id,
    performative="spawn_agent",
    content=f"Created agent: {name} ({role})",
    metadata={"agent_id": agent_id, "model": model}
)

# send_message
build_trace_event(
    sender="leader",
    receiver=agent_id,
    performative="dispatch_task",
    content=f"Task sent: {message[:100]}...",
    metadata={"agent_id": agent_id, "async": True}
)

# agent 执行完成
build_trace_event(
    sender=agent_id,
    receiver="leader",
    performative="task_complete",
    content=f"Result saved to {result_file}",
    metadata={"agent_id": agent_id, "result_size": len(result)}
)

# close_agent
build_trace_event(
    sender="leader",
    receiver=agent_id,
    performative="close_agent",
    content=f"Agent closed: {agent_id}",
    metadata={"agent_id": agent_id}
)
```

Logging（使用 Python logging）：
```python
import logging

logger = logging.getLogger("agent.team")

# INFO: 正常操作
logger.info(f"Agent {agent_id} spawned: {name}")
logger.info(f"Task sent to {agent_id}")
logger.info(f"Agent {agent_id} completed task")

# WARNING: 异常情况
logger.warning(f"Agent {agent_id} still busy, task queued")

# ERROR: 错误
logger.error(f"Agent {agent_id} execution failed: {error}")
```

### 3. Enhanced TodoList（增强版）

**职责**：任务依赖管理和拓扑排序

**数据结构**：
```python
@dataclass
class Todo:
    id: str                    # todo-1, todo-2
    content: str
    status: TodoStatus         # pending/in_progress/completed
    depends_on: list[str]      # 依赖的 todo id
    assigned_to: str | None    # 分配给的 agent_id
    created_at: str
    completed_at: str | None

class TodoGraph:
    todos: dict[str, Todo]
    adjacency: dict[str, list[str]]

    def get_executable_todos(self) -> list[Todo]:
        """拓扑排序，返回可执行的 todo"""

    def detect_cycle(self) -> list[str] | None:
        """检测依赖环"""
```

**工具**：
- `write_todos(todos: list[dict]) -> str` - 创建 todo 列表（检测依赖环）
- `update_todo(todo_id: str, status: str, assigned_to: str) -> str` - 更新状态
- `list_todos() -> str` - 列出所有 todos 及可执行状态

**关键特性**：
- 依赖环检测（DFS 算法）
- 拓扑排序展示可执行 todo
- 自动更新后继节点状态

