## Why

当前项目的 middleware 实现（ProgressiveToolDisclosureMiddleware）和 trace 系统分散在不同模块中，缺乏统一的架构规范。参考 LangChain 官方 middleware 最佳实践，需要规范化 middleware 实现并将 trace 功能迁移到 middleware 架构中，提升代码的可维护性和扩展性。

## What Changes

- 新建 `agent/middlewares/` 目录，集中管理所有 middleware 实现
- 重构 `ProgressiveToolDisclosureMiddleware`，遵循 LangChain middleware 规范（支持 node-style 和 wrap-style hooks）
- 将当前的 trace 系统（`agent/domain/trace.py`）迁移为 middleware 实现，通过 `before_model`/`after_model` hooks 自动记录执行阶段
- 统一 middleware 的注册和配置机制

## Capabilities

### New Capabilities
- `middleware-architecture`: 建立标准化的 middleware 架构，包括目录结构、基类定义和注册机制
- `trace-middleware`: 将 trace 功能实现为 middleware，自动追踪 agent 执行流程

### Modified Capabilities
- `progressive-tool-disclosure`: 重构现有的 ProgressiveToolDisclosureMiddleware，使其符合 LangChain 标准

## Impact

**代码变更：**
- 新增 `agent/middlewares/` 目录及相关模块
- 重构 `agent/capabilities.py` 中的 ProgressiveToolDisclosureMiddleware
- 修改 `agent/domain/trace.py` 或创建新的 trace middleware
- 更新 middleware 的使用方（如 `agent/runtime_agent.py`）

**依赖影响：**
- 依赖 LangChain 的 middleware 接口（已存在）
- 不影响外部 API 和用户接口

**测试影响：**
- 需要更新相关单元测试（`tests/unit/test_agent_capabilities.py`, `tests/unit/test_domain_trace.py`）
- 需要添加新的 middleware 集成测试
