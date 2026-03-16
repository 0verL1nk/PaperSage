## ADDED Requirements

### Requirement: Middleware 目录结构
系统 SHALL 在 `agent/middlewares/` 目录下组织所有 middleware 实现，包含 `__init__.py` 和具体的 middleware 模块文件。

#### Scenario: 目录结构符合规范
- **WHEN** 开发者查看项目结构
- **THEN** 存在 `agent/middlewares/__init__.py`、`agent/middlewares/trace.py`、`agent/middlewares/progressive_tool_disclosure.py` 等模块

### Requirement: 直接继承 LangChain AgentMiddleware
系统 SHALL 让所有自定义 middleware 直接继承 `langchain.agents.middleware.AgentMiddleware` 基类，无需额外的项目级基类。

#### Scenario: TraceMiddleware 继承 AgentMiddleware
- **WHEN** 定义 TraceMiddleware 类
- **THEN** 使用 `class TraceMiddleware(AgentMiddleware)` 并重写需要的 hook 方法

### Requirement: 支持 hook_config 装饰器
系统 SHALL 支持使用 `@hook_config` 装饰器配置 hook 的行为，如 can_jump_to 参数。

#### Scenario: 配置 can_jump_to
- **WHEN** middleware 的 before_model hook 需要提前结束执行
- **THEN** 使用 `@hook_config(can_jump_to=["end"])` 装饰该方法，并在返回值中包含 `{"jump_to": "end"}`

### Requirement: Middleware 列表注册
系统 SHALL 支持将多个 middleware 实例组织为列表，按顺序传递给 agent 构造函数。

#### Scenario: 注册多个 middleware
- **WHEN** 创建 agent 时传入 middleware 列表 `[TraceMiddleware(), ProgressiveToolDisclosureMiddleware(...)]`
- **THEN** 所有 middleware 按列表顺序执行，before hooks 从前到后，wrap hooks 嵌套（第一个包裹所有后续的），after hooks 从后到前
