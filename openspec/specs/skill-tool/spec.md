## ADDED Requirements

### Requirement: Skill template application tool
系统必须提供技能模板应用工具，支持动态加载和应用技能指导。

#### Scenario: 成功应用技能
- **WHEN** 用户提供有效的技能名称和任务描述
- **THEN** 系统返回技能的指导内容、引用和脚本

#### Scenario: 技能名称标准化
- **WHEN** 用户提供技能名称
- **THEN** 系统将其转换为小写并去除空格

#### Scenario: 未知技能处理
- **WHEN** 用户请求不存在的技能
- **THEN** 系统返回可用技能列表

#### Scenario: 危险任务阻止
- **WHEN** 任务描述包含危险模式
- **THEN** 系统拒绝执行并返回错误

#### Scenario: 技能引用限制
- **WHEN** 技能包含引用文件
- **THEN** 系统最多返回2个引用，每个限制1800字符

#### Scenario: 技能元数据
- **WHEN** 技能包含agent_metadata
- **THEN** 系统在输出中包含该元数据
