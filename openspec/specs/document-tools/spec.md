## ADDED Requirements

### Requirement: Document search tool
系统必须提供基于RAG的文档检索工具，支持查询清理和安全检查。

#### Scenario: 成功检索文档
- **WHEN** 用户提供有效查询字符串
- **THEN** 系统返回相关的文档片段

#### Scenario: 查询被清理
- **WHEN** 查询超过1200字符
- **THEN** 系统截断查询到1200字符并执行检索

#### Scenario: 危险查询被阻止
- **WHEN** 查询包含危险模式（如系统命令、注入攻击）
- **THEN** 系统返回错误信息并拒绝执行

#### Scenario: 返回证据JSON格式
- **WHEN** 提供evidence函数且检索成功
- **THEN** 系统返回JSON格式的证据数据

### Requirement: Document reading tool
系统必须提供分块读取文档的工具，支持offset/limit分页和可选的RAG上下文。

#### Scenario: 基本分块读取
- **WHEN** 用户指定offset和limit
- **THEN** 系统返回指定位置的文档内容和总长度

#### Scenario: 包含RAG上下文
- **WHEN** 用户设置include_rag=True
- **THEN** 系统额外返回相关的RAG上下文片段

#### Scenario: 默认参数
- **WHEN** 用户不提供参数
- **THEN** 系统使用offset=0, limit=2000作为默认值

### Requirement: Document listing tool
系统必须提供列出所有已加载文档的工具，支持详细模式。

#### Scenario: 基本列表
- **WHEN** 用户调用list_document
- **THEN** 系统返回JSON格式的文档列表，包含doc_uid和doc_name

#### Scenario: 详细模式
- **WHEN** 用户设置verbose=True
- **THEN** 系统额外返回每个文档的字符长度

#### Scenario: 无文档
- **WHEN** 当前项目范围内无文档
- **THEN** 系统返回友好的提示信息
