## ADDED Requirements

### Requirement: Tool builder function
系统必须提供工具构建函数，根据配置动态组装工具集。

#### Scenario: 基本工具构建
- **WHEN** 调用build_agent_tools并提供必要的回调函数
- **THEN** 系统返回已配置的工具列表

#### Scenario: 工具启用控制
- **WHEN** 通过allowed_tools参数指定允许的工具
- **THEN** 系统只构建允许列表中的工具

#### Scenario: 环境变量禁用
- **WHEN** 设置AGENT_DISABLE_<TOOL_NAME>环境变量
- **THEN** 系统跳过该工具的构建

### Requirement: Tool metadata enhancement
系统必须支持工具元数据增强，特别是技能工具的可搜索性。

#### Scenario: 技能元数据注入
- **WHEN** 发现可用技能
- **THEN** 系统将技能信息附加到use_skill工具的描述中

#### Scenario: 工具描述更新
- **WHEN** 增强工具元数据
- **THEN** 系统同步更新工具对象和元数据映射
