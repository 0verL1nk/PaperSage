## ADDED Requirements

### Requirement: Completed turns SHALL persist raw episodes and enqueue background memory extraction

系统必须在一轮对话完成后先持久化原始 memory episode，再异步投递长期记忆抽取任务。长期记忆保存不得阻塞用户主问答返回。

#### Scenario: Assistant turn completes successfully
- **WHEN** 系统完成一轮用户问题与 assistant 回复并准备写入长期记忆
- **THEN** 系统必须先保存该轮原始 episode 数据
- **THEN** 系统必须异步投递一个 memory extraction job
- **THEN** 用户主链路不得等待 memory extraction 完成

#### Scenario: Memory extraction job retries after transient failure
- **WHEN** 异步 memory extraction job 因临时错误失败
- **THEN** 系统必须允许基于已保存的 raw episode 重试该任务
- **THEN** 重试写入不得产生重复的 active memory item

### Requirement: Long-term memory SHALL distinguish user memory from knowledge memory

系统必须将长期记忆至少区分为 user memory 与 knowledge memory，并为两类记忆保存结构化字段，而不是仅保存聊天片段文本。

#### Scenario: User preference is extracted
- **WHEN** 本轮对话包含稳定的用户偏好、格式要求或长期指令
- **THEN** 系统必须将其保存为 user memory
- **THEN** 该记忆必须包含 canonical text、status、confidence 和来源 episode

#### Scenario: Knowledge fact is extracted
- **WHEN** 本轮对话包含论文事实、项目事实或可引用结论
- **THEN** 系统必须将其保存为 knowledge memory
- **THEN** 该记忆必须包含 canonical text、evidence、status、confidence 和来源 episode

### Requirement: Memory writes SHALL reconcile against existing memories before persistence

系统必须在持久化长期记忆前，对候选记忆与已有 active memory 执行 reconcile，并明确产生 `ADD`、`UPDATE`、`DELETE`、`NONE` 或 `SUPERSEDE` 动作。

#### Scenario: Duplicate memory candidate matches existing active memory
- **WHEN** 候选记忆与现有 active memory 在 dedup key 或 canonical identity 上匹配
- **THEN** 系统不得创建新的重复 active memory item
- **THEN** 系统必须执行 `UPDATE` 或 `NONE`

#### Scenario: New user instruction replaces old instruction
- **WHEN** 新的 user memory 与旧的 active user memory 指向同一偏好槽位但值发生变化
- **THEN** 系统必须将旧记忆标记为 superseded 或 inactive
- **THEN** 系统必须将新记忆作为当前 active 版本保存

#### Scenario: New evidence supersedes old knowledge memory
- **WHEN** 新的 knowledge memory 与旧的 active knowledge memory 冲突且新证据被接受
- **THEN** 系统必须保留新旧关系而不是简单并存为两个无关 active 记忆
- **THEN** 系统必须记录 supersede 或 equivalent lifecycle 信息

### Requirement: Long-term memory retrieval SHALL use typed retrieval with semantic recall as the canonical path

系统必须将长期记忆检索的主路径改为类型化过滤与语义召回。关键词匹配只能作为受控降级路径，不能继续作为长期记忆的主检索算法。

#### Scenario: User memory is loaded for prompt policy context
- **WHEN** 系统为一次新请求构建 prompt
- **THEN** active user memory 必须按类型和状态过滤后加载
- **THEN** 该类记忆不得依赖关键词全文检索才能生效

#### Scenario: Knowledge memory is recalled for a semantically related query
- **WHEN** 用户查询与已有 knowledge memory 语义相关但不共享显式关键词
- **THEN** 系统必须能够通过语义召回返回相关 active knowledge memory

#### Scenario: Rejected or superseded memories are excluded
- **WHEN** 系统执行长期记忆检索
- **THEN** 被标记为 rejected、deleted 或 superseded 的记忆不得作为默认注入结果返回

### Requirement: Prompt construction SHALL inject user memory and knowledge memory through separate channels

系统必须将 user memory 和 knowledge memory 以不同的 prompt 通道注入，保证行为约束与证据上下文分离。

#### Scenario: User memory influences response style
- **WHEN** 存在 active user memory 指定语言、格式或回答偏好
- **THEN** 系统必须将该记忆注入到 system 或 policy 语义位置

#### Scenario: Knowledge memory remains subordinate to current evidence
- **WHEN** active knowledge memory 被注入到 prompt
- **THEN** 系统必须明确当前问题与当前证据优先于长期 knowledge memory

### Requirement: The new memory pipeline SHALL replace and retire the legacy memory path

系统必须将新的异步结构化记忆 pipeline 作为唯一 canonical 长期记忆主链路，并替换、删除或降级旧的同步片段写入与关键词主检索代码。

#### Scenario: Legacy synchronous Q/A fragment persistence is retired
- **WHEN** 一轮对话结束并触发长期记忆保存
- **THEN** 系统不得再以同步方式将截断后的 `Q/A` 文本直接作为 canonical 长期记忆写入

#### Scenario: Legacy keyword search is not used as the default long-term memory retrieval path
- **WHEN** 系统为主 prompt 构建长期记忆上下文
- **THEN** 系统不得默认使用旧的关键词打分路径作为 canonical 检索实现

#### Scenario: Legacy facade layers do not remain as canonical entrypoints
- **WHEN** 新的长期记忆 pipeline 上线
- **THEN** 无独立语义的兼容 wrapper 和 facade 必须被删除，或明确降级为非主路径兼容接口
