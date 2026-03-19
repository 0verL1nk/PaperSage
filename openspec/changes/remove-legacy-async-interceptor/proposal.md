## Why

项目已经完成从旧 `agent.a2a` / `agent.orchestration` 路径向 `agent.application -> agent.runtime_agent -> agent.middlwares` 主链路的重构，但仓库中仍残留“异步策略拦截器”相关设置、测试导入和文档叙述。当前这些残留既误导使用者，也让测试继续依赖已经删除的模块路径，必须收敛到现行架构。

## What Changes

- 移除设置中心中已失效的“异步策略拦截”暴露项和对应误导文案。
- 清理当前运行时不再消费的异步策略拦截运行时调优映射，避免继续把无效配置注入环境。
- 将仍引用 `agent.a2a.*`、`agent.orchestration.*`、`policy_engine.intercept()` 的测试迁移到当前 middleware 主链路。
- 更新运行链路与路由职责文档，明确当前 canonical 入口是 `turn_engine` + middleware，而不是旧式预执行拦截器。
- **BREAKING**: 删除面向旧异步拦截器的用户可见设置项与相关兼容预期；不再承诺 `AGENT_POLICY_ASYNC_*` 会影响运行时行为。

## Capabilities

### New Capabilities
- `runtime-tuning-settings`: 约束设置中心和运行时调优仅暴露当前主链路实际消费的选项，不再暴露失效的异步拦截器配置。

### Modified Capabilities
- `orchestrator-core`: 明确当前执行主链路不得依赖已删除的 `agent.a2a` / `agent.orchestration` 入口，测试与调用方必须以 application + middleware 为准。
- `orchestration-middleware`: 明确复杂度分析与模式提示的 canonical 语义来自 middleware，而非请求前异步拦截器。

## Impact

- 受影响代码主要位于 `pages/2_settings.py`、`agent/application/runtime_tuning.py`、`agent/settings.py`、`agent/domain/request_context.py`、以及若干 `tests/evals` / `tests/integration` 文件。
- 受影响文档包括 `README.md` 与仍将旧编排路径当作现役主链路的计划/说明文档。
- 用户侧行为变化集中在设置中心：旧异步拦截选项被移除或降级为历史说明，避免继续制造“改了配置却不生效”的假象。
