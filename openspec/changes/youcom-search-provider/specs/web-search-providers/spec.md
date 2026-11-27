## ADDED Requirements

### Requirement: You.com search provider

系统必须支持You.com MCP作为可选Web搜索提供者（默认关闭，显式开启）。

#### Scenario: 默认关闭
- **WHEN** AGENT_WEB_YOUCOM_ENABLED未设置
- **THEN** 系统不初始化You.com提供者，提供者链与引入前一致

#### Scenario: keyless开启
- **WHEN** AGENT_WEB_YOUCOM_ENABLED=1且YOUCOM_API_KEY未设置
- **THEN** 系统以免费档端点（keyless）初始化You.com提供者并注册到提供者链

#### Scenario: 认证模式
- **WHEN** AGENT_WEB_YOUCOM_ENABLED=1且YOUCOM_API_KEY已设置
- **THEN** 系统以认证端点初始化You.com提供者并携带Bearer头

#### Scenario: 提供者失败回退
- **WHEN** You.com请求失败、返回错误或无结果
- **THEN** 系统按优先级尝试下一个提供者（Wikipedia及之后）