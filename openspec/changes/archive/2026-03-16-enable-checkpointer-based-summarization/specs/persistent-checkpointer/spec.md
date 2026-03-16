## ADDED Requirements

### Requirement: Persistent checkpointer implementation

系统SHALL使用SqliteSaver实现持久化的checkpointer，用于保存和恢复agent的对话状态。

#### Scenario: Checkpointer初始化
- **WHEN** 创建agent session时
- **THEN** 系统创建SqliteSaver实例，连接到指定的SQLite数据库文件

#### Scenario: 状态保存
- **WHEN** agent执行完一轮对话后
- **THEN** checkpointer自动将agent状态（包括消息历史）保存到SQLite数据库

#### Scenario: 状态恢复
- **WHEN** 使用已存在的thread_id创建agent session时
- **THEN** checkpointer自动从SQLite数据库加载该thread的历史状态

### Requirement: Thread ID管理

系统SHALL为每个用户会话维护唯一的thread_id，并持久化到数据库中，以支持会话恢复。

#### Scenario: 首次创建会话
- **WHEN** 用户首次在某个project和session中使用agent时
- **THEN** 系统生成新的thread_id并保存到数据库（关联project_uid和session_uid）

#### Scenario: 恢复已有会话
- **WHEN** 用户重新打开已存在的project和session时
- **THEN** 系统从数据库加载对应的thread_id，用于恢复agent状态

#### Scenario: Thread ID唯一性
- **WHEN** 查询thread_id时
- **THEN** 系统根据(project_uid, session_uid)组合查询，确保每个会话有唯一的thread_id

### Requirement: 会话恢复机制

系统SHALL在程序重启后能够恢复之前的agent会话状态，包括压缩后的消息历史。

#### Scenario: 程序重启后恢复
- **WHEN** 程序关闭后重新启动，用户打开之前的会话时
- **THEN** 系统使用持久化的thread_id和checkpointer恢复完整的对话状态

#### Scenario: 压缩状态恢复
- **WHEN** 恢复的会话中包含middleware压缩过的消息时
- **THEN** 系统恢复压缩后的消息状态，而不是原始的完整消息

### Requirement: Checkpointer配置

系统SHALL支持配置checkpointer的存储位置和参数。

#### Scenario: 数据库文件路径配置
- **WHEN** 初始化checkpointer时
- **THEN** 系统使用配置的数据库文件路径（默认为项目目录下的checkpoints.db）

#### Scenario: 多用户隔离
- **WHEN** 多个用户使用系统时
- **THEN** 每个用户的thread_id独立，不会相互干扰
