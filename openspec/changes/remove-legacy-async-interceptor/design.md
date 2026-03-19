## Context

当前仓库已经把 Agent Center 主链路收敛到 `ui -> agent.application -> runtime_agent -> middleware`，但旧的异步策略拦截模型还以三种方式残留：

1. 用户侧残留：设置中心仍暴露“异步策略拦截”相关开关和阈值，制造“可调但不生效”的错误预期。
2. 测试侧残留：`tests/evals` 和 `tests/integration` 仍直接导入 `agent.a2a.*`、`agent.orchestration.*`，导致收集期就失败。
3. 文档侧残留：README 和部分计划文档仍把 `policy_engine.intercept()`、`agent.orchestration` 叙述为现役入口。

项目约束要求保留清晰分层、避免无语义 wrapper，并要求同一业务用例只有一个 canonical 入口。因此这次变更的重点不是恢复旧拦截器，而是让运行时、测试、设置和文档一致地承认 middleware 是现行入口。

## Goals / Non-Goals

**Goals:**
- 移除用户界面与运行时调优中已经失效的异步拦截器概念。
- 让测试基于当前 application + middleware 主链路，而不是已删除的历史模块。
- 明确复杂度分析和模式提示的 canonical 责任边界在 orchestration middleware。
- 保证变更可回滚，且不引入新的兼容 facade。

**Non-Goals:**
- 不重新实现旧的 `agent.a2a` 或 `agent.orchestration` 包。
- 不把异步策略预测器重新接入当前运行链路。
- 不在本次 change 中全面重写所有历史计划文档，只修正当前对外说明和与主链路直接相关的文档。

## Decisions

### Decision 1: 直接删除失效入口暴露，而不是补兼容 wrapper

保留 `agent.a2a.*` 或 `agent.orchestration.*` 的兼容别名看似成本低，但会继续制造双入口，违反项目关于 canonical 入口和“simple is better”的约束。  
因此本次 change 采用清理式策略：修正调用方、测试和文档，不恢复旧包路径。

备选方案：
- 兼容层：能短期止血，但会延长历史债务寿命，且测试继续锚定错误入口。
- 重新接回异步拦截器：改动更大，且与当前 middleware-first 架构冲突。

### Decision 2: 设置中心只暴露当前主链路真实消费的调优项

`pages/2_settings.py` 中的异步拦截设置目前没有被 `turn_engine`、`runtime_agent` 或 middleware builder 消费。继续保留只会让用户误以为它们参与运行时决策。  
因此设置中心和环境同步逻辑应仅保留仍有实际消费者的参数，移除或显式停用异步拦截相关项。

备选方案：
- 保留设置但标为 deprecated：仍会增加噪音，并让页面承担历史说明责任。
- 偷偷保留存储字段：可以接受作为数据兼容，但不能继续出现在 UI 和活跃运行时同步路径上。

### Decision 3: 测试迁移到当前主链路的可验证边界

测试不再 patch 已删除的 `agent.orchestration.orchestrator.intercept_policy` 或导入 `agent.a2a.router`。  
新的验证边界应放在：
- `agent.application.turn_engine.execute_turn_core`
- `agent.middlewares.orchestration.OrchestrationMiddleware`
- `agent.application.agent_center.*` facade/runtime_state

这样既符合现有分层，也更接近真实运行时行为。

备选方案：
- 通过 monkeypatch 恢复旧模块路径：测试可跑，但验证对象仍是错误架构。

### Decision 4: 文档明确 middleware 是复杂度分析与模式提示的唯一现役入口

README 和主链路说明文档需要从“policy engine pre-intercept”更新为“middleware-based guidance”。  
历史计划文档允许保留为历史记录，但不应再被当前说明引用为现役架构。

## Risks / Trade-offs

- [Risk] 用户可能依赖旧设置项判断系统行为  
  → Mitigation: 在 README/设置中心文案中明确当前模式提示来自 middleware，并移除误导性开关。

- [Risk] 删除环境同步项后，已有测试可能因为断言旧 env 映射而失败  
  → Mitigation: 先更新 spec 和测试预期，再收敛实现，确保失败是有意的行为变更。

- [Risk] 部分历史文档仍会提到旧路径，造成局部不一致  
  → Mitigation: 本次优先修正 README 和直接面向当前主链路的文档；其余历史计划文档在后续治理中逐步归档。

- [Risk] 过度清理可能误删仍被其他模块读取的存量字段  
  → Mitigation: 区分“UI 暴露 / env 应用 / 数据库存量字段”三个层次；优先停止暴露和消费，再评估存储迁移。

## Migration Plan

1. 先更新 OpenSpec 和测试预期，确认新行为边界。
2. 收敛设置页和 runtime tuning 应用层，去掉失效 async interceptor 配置的暴露与注入。
3. 修正测试到当前 middleware 主链路，并补新的回归测试。
4. 更新 README 和必要架构说明。
5. 验证核心质量门禁。

回滚策略：
- 若 UI 或测试迁移引发意外回归，可单独回滚本 change 涉及的页面、测试和文档修改，不需要恢复旧模块路径。

## Open Questions

- 是否保留数据库中的 `agent_policy_async_*` 字段仅作历史兼容，还是在后续迁移中彻底移除。
- 是否需要在设置中心增加一条只读说明，解释模式提示已由 middleware 接管，以帮助老用户理解行为变化。
