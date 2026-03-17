## Context

当前orchestration架构使用policy_engine在执行前通过LLM自动决策是否启用plan/team模式。这种设计存在以下问题：

1. **系统控制流程**：policy_engine在orchestrator的前置阶段就决定了执行路径，agent只是被动执行
2. **缺乏灵活性**：无法根据执行过程中的发现动态调整策略
3. **额外成本**：每次请求都需要额外的LLM调用来做路由决策
4. **agent被动**：leader agent无法根据自己的判断选择最佳策略

现有架构中，orchestrator的执行流程是：
```
intercept_policy() → leader_agent.invoke() → 检测mode_activation_events → 执行plan/team
```

我们需要将控制权交还给agent，让它成为决策的主体。

## Goals / Non-Goals

**Goals:**
- 移除policy_engine的自动决策，让agent自主决策
- 将plan和team封装为tools，供agent主动调用
- 通过middleware提供上下文引导，而非强制决策
- 简化orchestrator逻辑，移除mode检测循环
- 保持现有planning_service和team_runtime的实现不变

**Non-Goals:**
- 不改变plan和team的内部实现逻辑
- 不修改progressive_tool_disclosure的核心机制
- 不保留policy_engine作为fallback（完全移除）
- 不改变agent的基础能力和工具集

## Decisions

### Decision 1: 完全移除policy_engine
**选择**: 删除policy_engine.py及其所有调用
**理由**:
- policy_engine的存在违背了agent-centric的设计理念
- 额外的LLM调用增加成本和延迟
- agent本身已经具备判断任务复杂度的能力

**替代方案考虑**:
- 保留作为fallback：增加复杂度，且与目标相悖
- 改为可选配置：仍然是系统控制，未解决根本问题

### Decision 2: 使用Middleware引导而非强制
**选择**: 创建OrchestrationMiddleware分析上下文并插入引导提示
**理由**:
- Middleware可以访问完整的对话历史
- 引导提示不强制agent决策，保持agent自主性
- 可以根据上下文动态调整引导策略

**实现方式**:
```python
class OrchestrationMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # 分析上下文复杂度
        if self._is_complex_task(request.messages):
            # 插入引导提示
            enhanced = self._inject_guidance(request.messages)
            request = request.override(messages=enhanced)
        return handler(request)
```

### Decision 3: Plan由Leader自主生成和管理
**选择**: Plan是Leader根据上下文自己撰写的策略文档，不是系统生成的
**理由**:
- Leader拥有完整的对话历史和上下文理解
- Leader根据自己的判断决定是否需要Plan以及Plan的内容
- Plan是Leader的工作文档，用于规划整体策略
- Plan可以在执行过程中由Leader动态更新
- 任务完成后Leader删除Plan，保持工作记忆清洁
- **不使用旧的planning_service.build_execution_plan** - 那是系统控制的方式

**工具实现**:
```python
# create_plan只是存储功能，不生成内容
def create_plan(goal: str, description: str) -> str:
    # 存储Leader提供的goal和description到agent state
    # 不调用LLM生成plan
    return "Plan created"

# 其他工具也只是CRUD操作
read_plan()           # 读取Leader创建的Plan
update_plan(description)  # 更新Plan内容
delete_plan()         # 删除Plan
```

**Plan生成方式**:
- **方案1（主要）**: Leader自己撰写Plan内容
  - Middleware检测复杂任务，插入引导提示
  - Leader根据上下文自己决定Plan内容
  - Leader调用create_plan(goal, description)存储
- **方案2（可选）**: Leader spawn子agent生成Plan
  - Leader决定需要Plan
  - Leader spawn子agent，传递上下文
  - 子agent分析后返回Plan建议
  - Leader使用或修改后存储

**存储**: Plan存储在agent state中，通过LangGraph的SqliteSaver checkpointer自动持久化到`./data/checkpoints.db`，支持跨会话访问

**工具可见性**: Plan工具标记为lazy，通过progressive disclosure激活

### Decision 4: Todolist通过Middleware提供
**选择**: 使用LangChain官方的TodoListMiddleware，删除现有的自定义todo工具
**理由**:
- 项目中已有`write_todo`和`edit_todo`工具（在local_ops.py中）
- LangChain官方的TodoListMiddleware提供了标准化的todolist管理能力
- 使用官方实现可以减少维护成本，获得社区支持
- Middleware自动注入todolist工具，agent可自主使用
- 参考：https://docs.langchain.com/oss/python/langchain/middleware/built-in#to-do-list

**迁移步骤**:
1. 从local_ops.py中移除`write_todo`和`edit_todo`工具定义
2. 集成LangChain的TodoListMiddleware到agent配置
3. 更新agent工具注册，移除todo相关工具

**存储**: Todolist由LangChain middleware管理，存储在agent state的PlanningState中，通过SqliteSaver checkpointer自动持久化，支持跨会话访问

**工具可见性**: Todolist工具由middleware自动注入，始终可见

### Decision 5: 简化Orchestrator为单次调用
**选择**: 移除while循环和mode检测逻辑
**理由**:
- tool调用的结果已经包含在agent response中
- 不需要额外的循环来处理mode activation
- 简化代码，减少复杂度

**新流程**:
```
orchestrator → leader_agent.invoke() → 提取结果（包含tool调用）→ 返回
```

## Risks / Trade-offs

### Risk 1: Agent可能不使用orchestration tools
**风险**: Agent可能不主动调用activate_plan_mode或activate_team_mode
**缓解**:
- OrchestrationMiddleware提供明确的引导
- 工具描述清晰说明使用场景
- 监控agent决策质量，必要时调整引导策略

### Risk 2: 引导策略可能不够准确
**风险**: Middleware的复杂度判断可能不准确，导致过度或不足引导
**缓解**:
- 使用简单的启发式规则（如prompt长度、关键词）
- 可以通过环境变量调整引导阈值
- 记录引导决策和agent响应，用于优化

### Risk 3: 向后兼容性
**风险**: 移除policy_engine可能影响依赖它的代码
**缓解**:
- 全面搜索policy_engine的所有引用
- 确保只在orchestrator中使用
- 添加测试验证新流程

### Trade-off: 决策质量 vs 成本
**权衡**: Agent自主决策可能不如专门的policy LLM准确
**选择**: 优先agent自主性和成本优化
**理由**:
- Agent本身已经是强大的LLM
- 减少一次LLM调用显著降低成本和延迟
- 可以通过引导和工具描述优化决策质量

## Migration Plan

### Phase 1: 创建新组件（不影响现有流程）
1. 创建OrchestrationMiddleware
2. 创建orchestration_tools.py（activate_plan_mode, activate_team_mode）
3. 添加单元测试

### Phase 2: 修改Orchestrator
1. 在orchestrator.py中注释掉policy_engine调用
2. 移除mode检测while循环
3. 简化为单次agent调用
4. 保留policy_engine.py文件（暂不删除）

### Phase 3: 集成测试
1. 测试简单任务（不应触发orchestration tools）
2. 测试复杂任务（agent应主动使用tools）
3. 验证middleware引导效果

### Phase 4: 清理
1. 确认所有测试通过
2. 删除policy_engine.py
3. 更新文档

### Rollback Strategy
如果新架构出现问题：
1. 恢复orchestrator.py中的policy_engine调用
2. 禁用OrchestrationMiddleware（通过环境变量）
3. 移除orchestration tools的注册

## Open Questions

1. **引导策略的具体实现**: 使用什么启发式规则来判断任务复杂度？
   - 候选方案：prompt长度、关键词匹配、历史对话轮数

2. **工具描述的优化**: 如何让agent更准确地理解何时使用orchestration tools？
   - 需要在实际使用中迭代优化

3. **监控和可观测性**: 如何监控agent的决策质量？
   - 需要添加metrics记录agent是否使用了orchestration tools
   - 记录任务成功率和用户满意度
