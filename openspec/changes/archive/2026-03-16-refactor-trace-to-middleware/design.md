## Context

当前项目中存在两个独立的系统：
1. **ProgressiveToolDisclosureMiddleware**（位于 `agent/capabilities.py`）：实现了渐进式工具披露功能，但实现方式不完全符合 LangChain middleware 标准
2. **Trace 系统**（位于 `agent/domain/trace.py`）：提供执行阶段追踪功能，但是通过手动调用函数实现，未集成到 middleware 架构中

LangChain 提供了标准的 middleware 机制，支持 node-style hooks（before_agent, before_model, after_model, after_agent）和 wrap-style hooks（wrap_model_call, wrap_tool_call），可以在 agent 执行的不同阶段自动拦截和处理。

**当前约束：**
- 必须保持 `agent/domain/trace.py` 的公共 API 不变，确保向后兼容
- 依赖 LangChain 的 middleware 接口（已存在）
- 需要支持通过环境变量控制功能开关

## Goals / Non-Goals

**Goals:**
- 建立统一的 middleware 架构，所有 middleware 集中在 `agent/middlewares/` 目录
- 重构 ProgressiveToolDisclosureMiddleware 使其完全符合 LangChain 标准
- 将 trace 功能迁移为 middleware 实现，自动追踪执行阶段
- 保持现有 trace API 的向后兼容性

**Non-Goals:**
- 不改变 trace 的功能行为（只改变实现方式）
- 不引入新的外部依赖
- 不修改 agent 的核心执行逻辑

## Decisions

### 决策 1: 新建 agent/middlewares/ 目录
**选择：** 创建独立的 `agent/middlewares/` 目录存放所有 middleware 实现

**理由：**
- 清晰的模块边界，便于维护和扩展
- 符合关注点分离原则
- 便于其他开发者快速定位 middleware 相关代码

**备选方案：** 保持在 `agent/capabilities.py` 中
- 缺点：文件过大（已超过 400 行），职责不清晰

### 决策 2: TraceMiddleware 使用 Node-style Hooks
**选择：** 使用 `before_model`、`after_model` 和 `after_agent` hooks 实现 trace 功能

**理由：**
- Node-style hooks 适合顺序执行的日志记录场景
- `before_model`: 在模型调用前提取当前 performative
- `after_model`: 在模型响应后记录新的 performative
- `after_agent`: 在执行完成后生成阶段摘要

**备选方案：** 使用 wrap-style hooks
- 缺点：wrap-style 适合需要控制执行流的场景（如重试、缓存），对于单纯的日志记录过于复杂

### 决策 3: ProgressiveToolDisclosureMiddleware 使用 Wrap-style Hook
**选择：** 使用 `wrap_model_call` hook 实现工具过滤

**理由：**
- 需要在模型调用前修改 request.tools 列表
- wrap-style hook 可以通过 `request.override(tools=filtered_tools)` 修改请求
- 符合 LangChain 文档中 "Dynamically selecting tools" 的最佳实践

**备选方案：** 使用 before_model hook
- 缺点：before_model 只能返回 state 更新，无法直接修改 ModelRequest

### 决策 4: 扩展 AgentState Schema
**选择：** 定义自定义 AgentState，添加 `trace_labels: NotRequired[list[str]]` 和 `trace_summary: NotRequired[str]` 字段

**理由：**
- 符合 LangChain 的状态管理模式
- 使用 NotRequired 保持向后兼容
- 便于在不同 hooks 之间传递 trace 信息

**备选方案：** 使用全局变量或 middleware 实例变量
- 缺点：不符合 LangChain 的设计模式，难以在分布式环境中使用

## Risks / Trade-offs

### 风险 1: 向后兼容性破坏
**风险：** 如果其他模块直接依赖 `agent/capabilities.py` 中的 ProgressiveToolDisclosureMiddleware，移动后会导致导入失败

**缓解措施：**
- 在 `agent/capabilities.py` 中保留一个 deprecated 的导入别名
- 添加 DeprecationWarning 提示开发者更新导入路径
- 更新所有已知的导入位置

### 风险 2: Trace 行为变化
**风险：** 从手动调用改为 middleware 自动调用，可能导致 trace 记录的时机或内容发生变化

**缓解措施：**
- 保持 `PHASE_BY_PERFORMATIVE`、`phase_label_from_performative`、`phase_summary` 函数不变
- 编写单元测试验证 trace 输出与之前一致
- 在集成测试中对比重构前后的 trace 结果

### 权衡 1: 增加 State 字段
**权衡：** 在 AgentState 中添加 trace_labels 和 trace_summary 字段会增加状态大小

**影响：** 对于长时间运行的 agent，state 会略微增大，但影响可忽略（每个标签约 10-20 字节）

## Migration Plan

### 阶段 1: 创建新的 Middleware 实现
1. 创建 `agent/middlewares/` 目录结构
2. 实现 `agent/middlewares/trace.py` 中的 TraceMiddleware
3. 重构 `agent/middlewares/progressive_tool_disclosure.py` 中的 ProgressiveToolDisclosureMiddleware
4. 编写单元测试验证新实现

### 阶段 2: 更新导入和注册
1. 在 `agent/capabilities.py` 中添加 deprecated 导入别名
2. 更新 `agent/runtime_agent.py` 等使用方，改为从新位置导入
3. 更新 middleware 注册逻辑，使用新的实现

### 阶段 3: 验证和清理
1. 运行完整的测试套件，确保功能正常
2. 对比重构前后的 trace 输出，确保一致性
3. 移除旧的实现代码（保留 deprecated 导入一段时间）

### 回滚策略
如果发现问题，可以快速回滚：
1. 恢复 `agent/capabilities.py` 中的原始实现
2. 恢复原始的导入路径
3. 禁用新的 middleware（通过环境变量）
