# T0.1: 环境检查 - LangGraph 0.3+ 依赖兼容性

## 执行日期
2026-03-14

## 验证结果

### 1. LangGraph 版本
- **Installed**: `1.0.10`
- **pyproject.toml 约束**: `langgraph>=0.3.0`
- **状态**: ✅ 满足约束

### 2. LangChain 版本
- **langchain**: `1.2.10`
- **langchain-core**: `1.2.17`
- **pyproject.toml 约束**: 
  - `langchain>=1.0.0,<2.0.0`
  - `langchain-core>=1.0.0,<2.0.0`
- **状态**: ✅ 满足约束

### 3. 单元测试结果
- **测试文件**: `tests/unit/test_runtime_agent.py`
- **测试用例**: 1 passed
- **状态**: ✅ PASSED

## 结论
- 所有依赖版本满足 plan 要求 (LangGraph 0.3+)
- 无需修改 pyproject.toml
- T0.2 可以正常推进

---

# T0.2: Feature Flag 机制（双轨切换开关）

## 执行日期
2026-03-14

## 实现内容

### 1. 创建文件
- `agent/orchestration/feature_flags.py` - Feature Flag 核心实现
- `tests/unit/test_feature_flags.py` - 单元测试（13 个测试用例）

### 2. 修改文件
- `agent/orchestration/__init__.py` - 添加 feature flags 导出

### 3. 功能特性
- **5 个 LangGraph 迁移相关 flags**:
  - `use_langgraph_runtime` (默认: False)
  - `use_langgraph_team` (默认: False)
  - `use_langgraph_memory` (默认: False)
  - `langgraph_streaming` (默认: True)
  - `langgraph_persistence` (默认: False)
- **环境变量加载**: 通过 `LANGRGRAPH_<FLAG_NAME>` 前缀读取
- **优先级**: runtime override > env var > default
- **测试支持**: `reset_flags()`, `set_env_loader()` 工具函数

### 4. API
- `is_enabled(flag_name) -> bool`
- `enable(flag_name) -> None`
- `disable(flag_name) -> None`
- `get_all_flags() -> dict[str, bool]`
- `list_flags() -> list[str]`

## 验证结果

### 单元测试
- `test_feature_flags.py`: **13 passed**
- `test_runtime_agent.py`: **1 passed** (regression check)

### 代码质量
- `basedpyright`: No diagnostics
- 遵循 AGENTS.md 约束（无 utils/utils.py 修改，无 giant file）

## 结论
- T0.2 完成，feature flag 基础设施已就绪
- 可支持 P1 阶段的 LangGraph 迁移任务双轨切换

---

# 战略变更: 移除 Feature Flag 策略

## 执行日期
2026-03-14

## 变更内容
- 用户决定：直接全部替换删除，不使用 feature flag 机制
- 原因：避免一堆老代码，保持代码库简洁

## 执行操作
1. **删除文件**:
   - `agent/orchestration/feature_flags.py`
   - `tests/unit/test_feature_flags.py`

2. **修改文件**:
   - `agent/orchestration/__init__.py` - 移除 feature flags 导出

3. **更新 Plan**:
   - `docs/plans/2026-03-14-langgraph-refactor-todo.md`:
     - Architecture: 改为"直接替换策略 (Direct Cutover)"
     - 移除 feature flag 相关任务和引用
     - T0.2 改为"搭建 LangGraph Checkpointer 基础设施"

## 验证结果
- ✅ `tests/unit/test_runtime_agent.py`: 1 passed
- ✅ Import check: `from agent.orchestration import *` 成功
- ✅ 无残留引用: `grep -r feature_flags` 无结果

## 影响范围
- 后续迁移任务不再使用 feature flag 切换
- 采用单一 canonical 路由/执行路径
- Plan 文件中所有 feature flag 相关步骤需要相应调整

---

# T0.2: 搭建 LangGraph Checkpointer 基础设施（直接迁移）

## 执行日期
2026-03-14

## 实现内容

### 1. 创建文件
- `agent/orchestration/checkpointer.py` - Checkpointer 工厂实现
- `tests/unit/test_checkpointer.py` - 单元测试（8 个测试用例）

### 2. 修改文件
- `agent/orchestration/__init__.py` - 添加 checkpointer 导出
- `pyproject.toml` - 添加 `langgraph-checkpoint-sqlite>=2.0.0` 依赖

### 3. 功能特性
- **工厂函数**: `create_checkpointer(checkpointer_type, *, conn_string)`
- **支持类型**: `memory` (InMemorySaver), `sqlite` (SqliteSaver)
- **类型安全**: `CheckpointerType` literal 类型
- **显式验证**: `UnsupportedCheckpointerTypeError` 自定义异常

### 4. API
```python
create_checkpointer("memory")  # -> InMemorySaver
create_checkpointer("sqlite")  # -> SqliteSaver (in-memory)
create_checkpointer("sqlite", conn_string="./checkpoints.db")  # -> SqliteSaver (file-based)
```

## 技术发现

### LangGraph Checkpoint 依赖结构
- `langgraph-checkpoint` (已安装): 提供 InMemorySaver
- `langgraph-checkpoint-sqlite` (新增): 提供 SqliteSaver
- SqliteSaver 3.0.x 使用 `SqliteSaver(conn: sqlite3.Connection)` 直接实例化

### API 差异
- InMemorySaver: 直接实例化 `InMemorySaver()`
- SqliteSaver: 需传入 sqlite3.Connection 对象

## 验证结果

### 单元测试
- `test_checkpointer.py`: **8 passed**
- `test_runtime_agent.py`: **1 passed** (regression check)

### 依赖安装
- `langgraph-checkpoint-sqlite==3.0.3`
- `aiosqlite==0.22.1` (自动安装)
- `sqlite-vec==0.1.6` (自动安装)

## 结论
- ✅ T0.2 完成，LangGraph Checkpointer 基础设施已就绪
- 为 P1 阶段状态/持久化迁移提供工厂支持

---

# T0.3: 基线测试 - 确保现有单元测试通过

## 执行日期
2026-03-14

## 验证结果

### 1. 单元测试
- **命令**: `uv run --extra dev python -m pytest tests/unit -q --tb=short`
- **结果**: 
  - ✅ 360 passed
  - 1 skipped
  - 242 warnings (non-blocking)
- **状态**: **PASSED**

> Note: 初始预期 53 个测试，实际运行 360 个（包含 T0.2 新增的测试）

### 2. 质量门禁 (Core)
- **命令**: `bash scripts/quality_gate.sh core`
- **结果**:
  - ✅ ruff check: All checks passed!
  - ✅ mypy: Success: no issues found in 11 source files
  - 仅有的 notes: 2 个 annotation-unchecked 提示（不影响通过）
- **状态**: **PASSED**

### 3. 回归检查
- 确认 T0.2 的改动未引入任何新失败
- 现有测试套件全部通过

## 结论
- ✅ T0.3 完成，基线测试全部通过
- 可安全进入 T0.4 阶段（P1 LangGraph 运行时改造）
- 无需代码修改，无阻塞问题

---

# T0.4: 搭建迁移沙箱环境（独立 Worktree）

## 执行日期
2026-03-14

## 实现内容

### 1. Worktree 创建
- **路径**: `../paper-sage-langgraph-migration`
- **分支**: `feat/langgraph-migration`
- **基础**: 复用现有 `.worktrees/` 模式，已验证 gitignored

### 2. 依赖同步
- 使用 `uv sync --extra dev` 安装
- **LangGraph 版本**: `1.1.2` (满足 LangGraph 1.1.x 要求)
- **其他关键依赖**:
  - `langgraph-checkpoint==4.0.1`
  - `langgraph-checkpoint-sqlite==3.0.3`
  - `langchain==1.2.12`
  - `langchain-core==1.2.19`

### 3. 基线测试验证
- **命令**: `uv run --extra dev python -m pytest tests/unit/test_runtime_agent.py -q`
- **结果**: **1 passed in 6.47s**
- **状态**: ✅ PASSED

## 验证结果

### 1. Worktree 验证
```bash
ls -la ../paper-sage-langgraph-migration  # ✅ 存在
git -C ../paper-sage-langgraph-migration status  # On branch feat/langgraph-migration, clean
```

### 2. 依赖验证
- 所有依赖正确安装
- LangGraph 1.1.2 满足最新 1.1.x 环境要求
- 无遗留 0.3.x 版本残留

### 3. 测试验证
- 单元测试通过
- 回归检查通过

## 結論
- ✅ T0.4 完成，迁移沙箱环境已就绪
- Worktree 路径: `../paper-sage-langgraph-migration`
- 分支: `feat/langgraph-migration`
- 可进入 P1 阶段 LangGraph 运行时改造
- P1 任务应在此 worktree 中执行

---

# T1.1: 状态持久化迁移 - InMemorySaver → SqliteSaver + checkpointer

## 执行日期
2026-03-14

## 实现内容

### 1. 修改文件
- `agent/runtime_agent.py` - 将默认 checkpointer 从 InMemorySaver 改为 create_checkpointer("sqlite")

### 2. 新增测试
- `tests/unit/test_runtime_agent.py` - 新增 2 个测试用例:
  - `test_default_uses_sqlite_checkpointer` - 验证默认使用 sqlite
  - `test_explicit_checkpointer_overrides_default` - 验证显式 checkpointer 优先

### 3. 技术决策
- **循环导入解决方案**: 使用函数内 late import 避免 `agent.orchestration.checkpointer` 触发 `__init__.py` 加载导致循环
- **默认行为**: sqlite (in-memory) 持久化
- **向后兼容**: 调用方仍可显式传入 checkpointer 参数覆盖默认

## 验证结果

### 单元测试
- `test_runtime_agent.py`: **3 passed** (1 existing + 2 new)
- `test_checkpointer.py`: **8 passed** (regression check)

### 代码质量
- 无新增 lsp_diagnostics
- 遵循 AGENTS.md 约束

## 结论
- ✅ T1.1 完成，runtime_agent 默认使用 SqliteSaver
- 为 T1.2 快照管理器集成扫清道路

---

# T1.2: 状态快照与回滚能力实现

## 执行日期
2026-03-14

## 实现内容

### 1. 新增文件
- `agent/orchestration/snapshot_manager.py`
  - 新增 `Snapshot` 元数据模型（`thread_id/checkpoint_id/created_at/metadata`）
  - 新增 `SnapshotManager` API：`save_snapshot` / `list_snapshots` / `get_checkpoint`
  - `list_snapshots` 通过 `checkpointer.list(...)` 读取线程历史并映射快照
  - `get_checkpoint` 优先走 `get_tuple(config)`，并保留 `get(config, checkpoint_id)` 兼容分支（仅在 TypeError 回退）

### 2. 新增测试
- `tests/unit/test_snapshot_manager.py`
  - TDD 红灯验证：先因缺失 `snapshot_manager` 模块失败
  - 覆盖保存快照元数据、按线程列举快照、按 checkpoint 获取状态、缺失 checkpoint 返回 `None`
  - 使用 `InMemorySaver.put/list/get` 真实 API 路径验证 LangGraph 1.1.x 兼容性

## 关键发现
- LangGraph 1.1.x 下 `InMemorySaver.get` 签名为 `get(config)`，`checkpoint_id` 通过 `configurable.checkpoint_id` 传入。
- `InMemorySaver.get_tuple(config)` 可直接返回 `CheckpointTuple`，用于回滚前检查点检索更直接。
- `CheckpointTuple.config` 中可读取 `thread_id/checkpoint_id`，可稳定映射为快照列表项。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_snapshot_manager.py -q` → `4 passed`
- `uv run --extra dev python -m pytest tests/unit/test_checkpointer.py -q` → `8 passed`
- `uv run --extra dev python -m pytest tests/unit/test_runtime_agent.py -q` → `3 passed`

---

# T1.3: Plan-Act 工作流迁移 - planning_service.py → LangGraph ToolNode + 条件边

## 执行日期
2026-03-14

## 实现内容

### 1. 新增文件
- `agent/orchestration/langgraph_plan_act.py`
  - 新增 `PlanActState` 显式状态模型（包含 `prompt/attempt/max_attempts/plan/status/errors/messages`）
  - 新增 `create_plan_act_graph(planner=...)`，使用 `ToolNode` 执行 `build_execution_plan_tool`
  - 新增条件边 `_route_after_consume`：`continue -> prepare_plan_call`，`end -> END`
  - 新增 `run_plan_act_graph(...)` 统一执行入口

### 2. 修改文件
- `agent/orchestration/planning_service.py`
  - 抽取 `_build_execution_plan_with_llm(...)` 作为规划核心
  - `build_execution_plan(...)` 直接切换到 `run_plan_act_graph(...)` 路径（无 feature flag）
  - 保留异常回退到 `_build_execution_plan_with_llm(...)`，保证兼容与稳定

### 3. 新增测试
- `tests/unit/test_langgraph_plan_act.py`
  - 覆盖 ToolNode + 条件边重试路径（首次失败，二次成功）
  - 覆盖最大尝试次数耗尽后的失败行为
  - 覆盖 `planning_service.build_execution_plan` 直连 LangGraph 入口（direct-cutover）

## 关键发现
- 在 LangGraph 1.1.x 中，`ToolNode` 推荐在 `StateGraph` 中使用；直接单独 `ToolNode.invoke(...)` 可能触发缺失 config key 报错。
- `ToolNode` 工具结果通过 `ToolMessage.content` 回传，适合用 JSON 串做稳定解析并映射回 `ExecutionPlan`。
- 条件边 continue/end 逻辑可用状态中的 `plan` 是否存在 + `attempt/max_attempts` 组合判定，行为确定性强。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_langgraph_plan_act.py -q` → `3 passed`

---

# T1.4: 循环执行迁移 - 自研 step_verify → LangGraph while 节点

## 执行日期
2026-03-14

## 实现内容

### 1. 新增文件
- `agent/orchestration/retry_loop.py`
  - 新增 `build_retry_route(...)`：统一重试循环退出语义（success 或 attempt 达到上限）
  - 新增 `add_retry_loop_edge(...)`：封装 `StateGraph.add_conditional_edges(...)` 的 continue/end while 路由

### 2. 修改文件
- `agent/orchestration/langgraph_plan_act.py`
  - 移除本地 `_route_after_consume`
  - 改为使用 `retry_loop.build_retry_route + add_retry_loop_edge` 驱动计划生成重试 while 节点
  - 保持 direct-cutover 单路径，无 feature flag

### 3. 新增测试
- `tests/unit/test_retry_loop.py`
  - 覆盖成功退出路径：在第 2 次 attempt 成功时提前结束
  - 覆盖上限退出路径：持续失败时在 `max_attempts` 到达后结束

## 关键发现
- 将重试条件（成功判定 + attempt 上限）抽象为 `build_retry_route` 后，Plan-Act 图的 while 语义可复用且保持确定性。
- `add_retry_loop_edge` 仅封装 LangGraph 条件边映射（`continue -> node`, `end -> END`），能在不引入新分支策略的前提下直接替换本地循环判定。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_retry_loop.py -q` → `2 passed`
- `uv run --extra dev python -m pytest tests/unit/test_langgraph_plan_act.py -q` → `3 passed`
- `uv run --extra dev python -m pytest tests/unit/test_runtime_agent.py -q` → `3 passed`
- `lsp_diagnostics` changed files:
  - `agent/orchestration/retry_loop.py` → no diagnostics
  - `agent/orchestration/langgraph_plan_act.py` → no diagnostics
  - `tests/unit/test_retry_loop.py` → no diagnostics

---

# T1.5: 策略路由迁移 - policy_engine 决策逻辑保留，执行承载迁移至 LangGraph 条件路由节点

## 执行日期
2026-03-14

## 实现内容

### 1. 新增文件
- `agent/orchestration/langgraph_route_node.py`
  - 新增 `route_from_policy_decision(...)`：将 `PolicyDecision(plan_enabled/team_enabled)` 映射到 `react|plan_act|plan_act_replan`
  - 新增 `run_policy_route_node(...)`：通过 LangGraph `StateGraph + add_conditional_edges` 执行条件路由节点并返回模式

### 2. 修改文件
- `agent/orchestration/orchestrator.py`
  - 接入 `run_policy_route_node(advisory_decision)` 作为执行承载入口
  - 保持 `policy_engine.intercept` 作为业务策略决策源，不改 plan/team 规则
  - 新增 policy 路由触发的 `pending_plan_from_policy/pending_team_from_policy` 执行路径（direct-cutover）
  - 保留既有 leader mode tool 触发兼容路径，避免破坏已有协议行为测试

### 3. 新增测试
- `tests/unit/test_langgraph_route_node.py`
  - 覆盖 route-node 三档映射：`react / plan_act / plan_act_replan`
  - 覆盖 parity：route-node 映射与 `agent/a2a/router.py::_policy_to_workflow_mode` 对齐
  - 覆盖 orchestrator 使用 route-node 触发 plan 执行（trace 中出现 `plan` 与 `step_dispatch`）

## 关键发现
- 在 direct-cutover 要求下，最小迁移方式是“保留 policy_engine 决策 + 将执行承载切到 route-node”，无需改动策略 prompt/rules。
- 使用 `StateGraph.add_conditional_edges(...)` 构建单节点路由器时，编译图可通过 `@lru_cache` 复用，避免每次执行重新构图开销。
- 为减少存量协议测试回归，route-node 承载可与 leader mode activation 并存：前者负责默认执行路径，后者保留兼容触发语义。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_langgraph_route_node.py -q` → `5 passed`
- `uv run --extra dev python -m pytest tests/unit/test_workflow_router.py -q` → `10 passed`
- `uv run --extra dev python -m pytest tests/unit/test_orchestration_protocol.py -q` → `21 passed`
- `lsp_diagnostics` changed files:
  - `agent/orchestration/langgraph_route_node.py` → no diagnostics (error severity)
  - `agent/orchestration/orchestrator.py` → no diagnostics (error severity)
  - `tests/unit/test_langgraph_route_node.py` → no diagnostics (error severity)

---

# T1.5 语义修正: `replan` 作为 `plan_act` 内部能力（非独立路由模式）

## 执行日期
2026-03-14

## 修正内容
- 路由模式集合从三档收敛为两档：`react|plan_act`。
- `team_enabled` 不再映射为 `plan_act_replan` 标签，而是映射到 `plan_act`。
- `replan` 语义保留在 `plan_act` 执行生命周期内部（step_verify 失败触发 revise/replan），不作为外部路由模式。

## 代码变更
- `agent/orchestration/langgraph_route_node.py`
  - `WorkflowMode` 调整为 `Literal["react", "plan_act"]`
  - `route_from_policy_decision(...)` 调整为 `plan_enabled or team_enabled => plan_act`
  - LangGraph 条件边与结果校验移除 `plan_act_replan`
- `agent/a2a/router.py`
  - `_policy_to_workflow_mode(...)` 调整为两档映射（team 归并到 plan_act）
- `agent/orchestration/orchestrator.py`
  - `pending_plan_from_policy` 仅依赖 `workflow_mode == "plan_act"`
  - `pending_team_from_policy` 由 `advisory_decision.team_enabled` 直接派生（不依赖第三模式字符串）
- `tests/unit/test_langgraph_route_node.py`
  - team policy 映射断言改为 `plan_act`
- `tests/unit/test_workflow_router.py`
  - team 相关 mode 断言改为 `plan_act`

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_langgraph_route_node.py -q` → `5 passed`
- `uv run --extra dev python -m pytest tests/unit/test_workflow_router.py -q` → `10 passed`
- `uv run --extra dev python -m pytest tests/unit/test_orchestration_protocol.py -q` → `21 passed`
- `lsp_diagnostics` (error severity) changed files:
  - `agent/orchestration/langgraph_route_node.py` → no diagnostics
  - `agent/orchestration/orchestrator.py` → no diagnostics
  - `agent/a2a/router.py` → no diagnostics
  - `tests/unit/test_langgraph_route_node.py` → no diagnostics
  - `tests/unit/test_workflow_router.py` → no diagnostics

---

# T2.1: 依赖拓扑迁移（DAG -> Send API）

## 执行日期
2026-03-14

## 实现内容

### 1. 新增文件
- `agent/orchestration/langgraph_team_dag.py`
  - 新增 `build_ready_task_dispatches(...)`：基于 LangGraph `Send` 生成 ready-task 派发列表
  - 实现真实依赖过滤：仅当依赖任务存在且状态为 `done` 时，任务才会 ready
  - 保留 team runtime 既有执行顺序语义：`round -> role_order -> todo_id`

### 2. 修改文件
- `agent/orchestration/team_runtime.py`
  - `run_team_tasks(...)` 调度阶段改为调用 `build_ready_task_dispatches(...)`
  - direct-cutover 单路径接入（无 feature flag）
  - 当 dispatch 指向不存在任务时，设置 `fallback_reason="todo_dispatch_target_missing"` 并停止

### 3. 新增测试
- `tests/unit/test_langgraph_team_dag.py`
  - 覆盖仅派发 ready `todo` 任务
  - 覆盖依赖完成后解锁下游任务
  - 覆盖排序与 runtime 语义对齐

## 关键发现
- 将 ready-task 选择抽到 LangGraph Send helper 后，team runtime 可以在不改业务主流程的前提下完成拓扑迁移的第一步。
- 依赖过滤必须把“未知依赖”视为未满足，否则会出现误派发风险。
- 通过 `role_order` 注入可保持与现有 runtime 排序一致，减少行为漂移。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_langgraph_team_dag.py -q` → `3 passed`
- `uv run --extra dev python -m pytest tests/unit/test_team_runtime.py -q` → `2 passed`
- `lsp_diagnostics` (error severity) changed files:
  - `agent/orchestration/langgraph_team_dag.py` → no diagnostics
  - `agent/orchestration/team_runtime.py` → no diagnostics
  - `tests/unit/test_langgraph_team_dag.py` → no diagnostics

---

# T2.2: Team 多角色编排 - 动态角色生成复用与 LangGraph 整合

## 执行日期
2026-03-14

## 实现内容

### 1. 新增文件
- `agent/orchestration/role_dispatcher.py`
  - 新增 `RoleDispatchPlan`（`roles/role_map/role_order`）作为可复用角色调度数据结构
  - 新增 `build_role_dispatch_plan(...)`：通过注入 `role_generator + role_normalizer` 构建确定性角色派发计划

### 2. 修改文件
- `agent/orchestration/langgraph_team_dag.py`
  - 新增 `build_ready_role_dispatches(...)`：在 T2.1 的 ready-task 选择基础上，合并角色派发元数据（`assignee/role_goal`）
  - 保持 direct-cutover 单路径，无 feature flag

### 3. 新增测试
- `tests/unit/test_role_dispatcher.py`
  - 覆盖 role dispatch plan 的确定性构建与顺序索引
  - 覆盖 `max_members` 边界钳制（<=0 时强制为 1）
  - 覆盖与 LangGraph DAG 派发顺序整合及缺失角色时的兼容派发语义

## 关键发现
- 将角色派发抽象为 `RoleDispatchPlan` 后，可复用 `role_order` 给 DAG 进行稳定排序，同时不引入并行模型。
- 为避免 `team_runtime -> langgraph_team_dag -> role_dispatcher -> team_runtime` 循环依赖，dispatcher 采用依赖注入（generator/normalizer）而非模块内硬导入。
- `build_ready_role_dispatches(...)` 选择“保留派发 + 空 `role_goal`”处理缺失角色，兼容既有 runtime 的下游 fallback 语义。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_role_dispatcher.py -q` → `4 passed`
- `uv run --extra dev python -m pytest tests/unit/test_langgraph_team_dag.py -q` → `3 passed`
- `uv run --extra dev python -m pytest tests/unit/test_team_runtime.py -q` → `2 passed`
- `lsp_diagnostics` (error severity) changed files:
  - `agent/orchestration/role_dispatcher.py` → no diagnostics
  - `agent/orchestration/langgraph_team_dag.py` → no diagnostics
  - `tests/unit/test_role_dispatcher.py` → no diagnostics

---

# T2.3: A2A 协议桥接层 - 保留自研核心，仅做接口适配

## 执行日期
2026-03-14

## 实现内容

### 1. 新增文件
- `agent/orchestration/a2a_bridge.py`
  - 新增 `A2AInvocation`：统一桥接层输入（`question/workflow_mode/max_replan_rounds`）
  - 新增 `A2ABridge`：
    - `orchestration_to_a2a_input(...)`：编排输入映射到 A2A 调用参数
    - `run_with_session(...)`：复用既有 `session.coordinator.run(...)` 执行 A2A 流程
    - `a2a_to_orchestration_output(...)`：A2A 输出回填编排层结果（`final_answer/trace/session_id`）
    - `create_session(...)`：仅透传到既有 `create_multi_agent_a2a_session(...)`，不改 A2A 核心

### 2. 修改文件
- `agent/orchestration/langgraph_team_dag.py`
  - 新增 `execute_team_task_via_a2a_bridge(...)` 作为最小集成 hook
  - DAG 可在集成点通过桥接层调用 A2A，会话执行逻辑仍由既有 coordinator 承担

### 3. 新增测试
- `tests/unit/test_a2a_bridge.py`
  - 覆盖默认输入映射（`prompt -> question`、默认 `workflow_mode/max_replan_rounds`）
  - 覆盖 `run_with_session(...)` 对 `coordinator.run(...)` 的参数传递与输出映射
  - 覆盖 DAG hook 对 bridge 的委托调用行为

## 关键发现
- 保持 `A2AMultiAgentSession + A2AMultiAgentCoordinator.run(...)` 原接口不变时，桥接层只需做输入键归一化与输出结构映射即可落地。
- 在 Team DAG 引入“函数级 hook”足以完成桥接点预留，避免直接耦合到 runtime 主循环，满足最小侵入目标。
- `agent/a2a/*` 无需改动即可完成 T2.3 要求。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit/test_a2a_bridge.py -q` → `3 passed`
- `uv run --extra dev python -m pytest tests/unit/test_a2a*.py -q` → `27 passed`
- `git diff --name-only agent/a2a/` → *(empty output)*
- `lsp_diagnostics` (error severity) changed files:
  - `agent/orchestration/a2a_bridge.py` → no diagnostics
  - `agent/orchestration/langgraph_team_dag.py` → no diagnostics
  - `tests/unit/test_a2a_bridge.py` → no diagnostics

---

# T3.1: 清理遗留双重路径，移除废弃代码分支

## 执行日期
2026-03-14

## 变更内容
- `agent/orchestration/team_runtime.py`
  - 删除未被任何调用路径引用的私有函数 `_todo_dependencies_done(...)`
  - 删除仅保留为历史兼容入口且无调用方的 `_build_team_todo_records(...)`
  - 同步修正文档注释：leader 规划失败时的 fallback 明确指向 `_build_team_todo_records_mechanical(...)`

## 证据与判断
- `grep -r "_native\|_legacy\|_old" agent/orchestration/` 无匹配，未发现按命名标记的遗留分支。
- 全局引用检索显示：
  - `_todo_dependencies_done(` 仅定义、无调用（可安全删除）。
  - `_build_team_todo_records(` 在代码中仅定义和注释提及、无运行时调用（可安全删除）。
- 保留单一路径：
  - 规划路径仍为 `planning_service -> run_plan_act_graph`（LangGraph-first）+ 异常兜底；
  - team 路径仍由 `run_team_tasks(...)` 走当前 DAG/bridge 逻辑，无 A2A 协议核心变更。

## 验证结果
- `uv run --extra dev python -m pytest tests/unit -q --tb=short` → `386 passed, 1 skipped, 242 warnings`
- `lsp_diagnostics` (error severity):
  - `agent/orchestration/team_runtime.py` → no diagnostics

---

# T3.2: 清理自研冗余代码（保留明确保留项）

## 执行日期
2026-03-14

## 实现内容

### 1. 修改文件
- `agent/orchestration/planning_service.py`
  - 删除仅做透传渲染的冗余包装函数 `build_execution_plan_text(...)`
  - 保留 `build_execution_plan(...)` 与 `revise_execution_plan(...)` 作为 Plan-Act 核心路径
- `agent/orchestration/orchestrator.py`
  - 删除无调用方的冗余 helper `_build_team_runtime_context(...)`
  - 保留 `intercept -> run_policy_route_node` 路由链，不引入并行旧入口
- `agent/orchestration/__init__.py`
  - 导出清单保持不变，仅统一为 tuple 形式，避免额外 alias/wrapper 扩散

### 2. 路由语义校正
- T3.2 文档中 canonical 路由链修正为 `react|plan_act`，不再暴露 `plan_act_replan` 外部模式。
- `replan` 保持为 `plan_act` 内部循环能力。

## 验证结果
- `ls -la agent/orchestration/`：完成目录审查并确认保留核心模块。
- `uv run --extra dev python -c "from agent.orchestration.policy_engine import intercept; from agent.orchestration.team_runtime import RoleRouterOutput; from agent.a2a import A2AMultiAgentCoordinator; print('All retained modules imported successfully')"` → `All retained modules imported successfully`
- `uv run --extra dev python -m pytest tests/unit -q --tb=short` → `386 passed, 1 skipped, 242 warnings`
- `lsp_diagnostics` (error severity):
  - `agent/orchestration/planning_service.py` → no diagnostics
  - `agent/orchestration/orchestrator.py` → no diagnostics
  - `agent/orchestration/__init__.py` → no diagnostics

---

# T3.3: 性能基准测试与优化

## 执行日期
2026-03-14

## 实现内容
- 新增 `tests/benchmark/test_langgraph_performance.py`，包含 offline-first 基准测试与 live guarded 路径。
- 离线路径不依赖真实密钥，不触发外部 API 调用；live 路径仅在 `BENCHMARK_LIVE=1` 且存在 `OPENAI_API_KEY` 时执行。
- 未修改 `pyproject.toml`（现有依赖已满足基准测试需求，无新增依赖必要性）。

## 基准结果（offline）
- 命令：`uv run --extra dev python -m pytest tests/benchmark/test_langgraph_performance.py -v`
- 结果：`3 passed, 1 skipped`
- 关键指标：
  - `checkpointer_creation_total=0.001047s`
  - `plan_act_graph_build_total=0.245319s`
  - `plan_act_graph_execution_total=0.243716s`

## 回归结果
- 命令：`uv run --extra dev python -m pytest tests/unit -q --tb=short`
- 结果：`386 passed, 1 skipped`

## 对比与优化结论
- 历史基线对比：N/A（当前仓库未提供 T3.3 前可比性能数据，当前结果作为首个基准记录）。
- 优化步骤：未触发（未观察到异常性能退化，维持现状）。
