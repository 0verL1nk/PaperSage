## ADDED Requirements

### Requirement: Web search provider abstraction
系统必须提供统一的Web搜索提供者抽象层，支持多个搜索引擎。

#### Scenario: 提供者优先级
- **WHEN** 系统初始化Web搜索客户端
- **THEN** 按优先级尝试：Brave API → SearXNG → Wikipedia → DuckDuckGo

#### Scenario: 提供者回退
- **WHEN** 首选提供者失败
- **THEN** 系统自动尝试下一个可用提供者

#### Scenario: 所有提供者失败
- **WHEN** 所有提供者都不可用或失败
- **THEN** 系统返回包含所有错误信息的失败消息

### Requirement: Brave Search integration
系统必须支持Brave Search API作为首选Web搜索提供者。

#### Scenario: API密钥配置
- **WHEN** BRAVE_SEARCH_API_KEY环境变量已设置
- **THEN** 系统初始化Brave搜索客户端

#### Scenario: 搜索请求
- **WHEN** 执行搜索查询
- **THEN** 系统调用Brave API并返回格式化结果

### Requirement: SearXNG integration
系统必须支持SearXNG公共实例池作为备选搜索提供者。

#### Scenario: 实例池配置
- **WHEN** AGENT_SEARXNG_BASE_URLS未配置
- **THEN** 系统使用默认实例列表

#### Scenario: 实例轮询
- **WHEN** 某个实例失败
- **THEN** 系统尝试池中的下一个实例

### Requirement: Wikipedia integration
系统必须支持Wikipedia API作为备选搜索提供者。

#### Scenario: Wikipedia搜索
- **WHEN** 执行搜索查询
- **THEN** 系统调用Wikipedia API并返回格式化结果

### Requirement: DuckDuckGo fallback
系统必须支持DuckDuckGo作为最后的回退选项。

#### Scenario: 回退启用
- **WHEN** AGENT_WEB_ENABLE_DDG_FALLBACK=1且其他提供者不可用
- **THEN** 系统初始化DuckDuckGo客户端

#### Scenario: 回退禁用
- **WHEN** AGENT_WEB_ENABLE_DDG_FALLBACK未设置
- **THEN** 系统不使用DuckDuckGo回退
