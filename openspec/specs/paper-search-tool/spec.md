## ADDED Requirements

### Requirement: Academic paper search tool
系统必须提供学术论文搜索工具，集成Semantic Scholar API。

#### Scenario: 成功搜索论文
- **WHEN** 用户提供学术查询和限制数量
- **THEN** 系统返回格式化的论文列表

#### Scenario: 查询清理
- **WHEN** 查询超过1200字符
- **THEN** 系统截断查询并执行搜索

#### Scenario: 危险查询阻止
- **WHEN** 查询包含危险模式
- **THEN** 系统拒绝执行并返回错误

#### Scenario: 搜索失败处理
- **WHEN** Semantic Scholar API失败
- **THEN** 系统返回友好的错误信息并建议使用search_web

#### Scenario: 限制参数验证
- **WHEN** 用户指定limit参数
- **THEN** 系统验证limit在1-20范围内
