## Context

当前项目的记忆系统由运行态消息缓存、会话摘要表和 `memory_items` 组成，但长期记忆的主链路仍然是“同步写入截断后的 `Q/A` 文本 + 关键词检索 + 字符串注入”。这导致长期记忆更像对话片段缓存，而不是可维护的知识层：

- 写入发生在用户主链路上，任何扩展都直接增加回合延迟
- 存储对象缺少 canonical form、证据、状态、去重键和替代关系
- 用户偏好与知识事实混存，覆盖规则和冲突处理不清晰
- 关键词主检索对同义改写、跨轮抽象和事实更新不稳定
- 旧代码在 `agent/memory/*`、`utils/utils.py`、prompt 构造链路中分散，已与目标架构不一致

项目已经具备 SQLite 持久化和异步任务队列基础设施，因此本次改造重点不是引入新层级，而是在现有边界内建立新的 canonical memory pipeline，并替换掉旧的同步片段写入与关键词主检索路径。

## Goals / Non-Goals

**Goals:**
- 将每轮对话后的长期记忆保存改为异步执行，避免阻塞主问答链路
- 将长期记忆的主存储形态从聊天片段改为结构化 memory item
- 区分用户记忆与知识记忆，并为两类记忆定义不同的更新和注入规则
- 为长期记忆提供 reconcile 机制，支持新增、更新、删除、跳过和替代
- 引入可追溯的数据模型，保留 raw episode 与 evidence 关联
- 替换并移除旧的同步片段写入、关键词主检索和无语义 facade 入口

**Non-Goals:**
- 不在本次变更中引入外部专用 memory service
- 不将整个系统重构为图数据库或完整知识图谱方案
- 不追求一次迁移所有历史数据为高质量结构化记忆
- 不改变短期会话消息缓存与中期 compact summary 的基本职责

## Decisions

### Decision 1: 采用 “episode + memory item + evidence” 三段式长期记忆模型

**选择:** 将长期记忆拆为原始 episode、结构化 memory item 和 evidence/link 关系，而不是继续把 `Q/A` 文本直接当长期记忆主体。

**理由:**
- episode 层负责审计、回放和重新抽取
- memory item 层负责检索、注入和状态管理
- evidence 层负责建立记忆与原始来源之间的可追溯关系
- 这种结构能支持后续去重、替代和误提取修复

**备选方案:**
- 继续扩展单表 `memory_items`: 简单，但会把事件日志、知识条目和注入对象混成一层，后续维护成本高
- 只保存结构化 memory item，不保留原始 episode: 实现更轻，但丢失回溯与重算能力

### Decision 2: 使用专用异步 memory writer worker，而不是复用主对话 leader agent

**选择:** 每轮对话完成后落 raw episode，并投递专用 memory writer 任务；由 worker 读取本轮对话和必要上下文，输出结构化 memory candidates。

**理由:**
- memory 保存职责与主问答职责不同，分离后更便于测试、限流和失败重试
- 专用 worker 的输入输出边界更稳定，避免复用大而全的对话 agent
- 项目已有 `utils/task_queue.py` 与 worker 任务模式，可直接复用

**备选方案:**
- 在主链路同步保存记忆: 实现简单，但直接增加用户时延
- 复用 leader agent 二次运行: 上下文更重、成本更高，行为也更难约束

### Decision 3: 长期记忆按类型拆分为 user memory 和 knowledge memory

**选择:** 定义至少两类顶层记忆类型：
- `user_memory`: 偏好、格式要求、长期指令
- `knowledge_memory`: 论文事实、项目事实、可引用结论

**理由:**
- 用户记忆默认“最新有效”，允许覆盖和替代
- 知识记忆必须带 evidence，并对冲突保持审慎，不适合简单覆盖
- 检索和 prompt 注入对两类记忆的需求不同

**备选方案:**
- 单一 memory_kind + 统一规则: 实现最省事，但生命周期和冲突策略会混乱
- 进一步细分为更多类型: 更精细，但首轮改造复杂度高，可在后续演进

### Decision 4: reconcile 采用动作驱动模型，而不是直接 upsert

**选择:** memory worker 在写入前必须对候选记忆执行 reconcile，动作集合为 `ADD`、`UPDATE`、`DELETE`、`NONE`、`SUPERSEDE`。

**理由:**
- 去重不是唯一问题，更关键的是“旧事实是否被新事实替代”
- 同一偏好、同一知识点可能出现更新和冲突，简单 upsert 不足以表达状态变化
- 动作驱动模型更适合测试和审计

**备选方案:**
- 仅靠 `dedup_key` 精确去重: 无法处理近似重复和事实替代
- 纯 LLM 自由写入: 无法提供稳定生命周期与幂等保证

### Decision 5: 检索改为类型化过滤 + 语义召回优先，关键词仅作降级

**选择:** 长期记忆的 canonical 检索路径改为：
- `user_memory`: 高优先级 active profile 直接加载或轻量过滤
- `knowledge_memory`: 基于 metadata filter 的语义召回，必要时再 rerank
- 关键词匹配仅作为兼容降级策略，不再作为主检索算法

**理由:**
- 用户偏好不需要每次全文搜索
- 知识记忆必须解决同义改写和抽象表达问题
- 兼容降级可以降低迁移风险

**备选方案:**
- 保留关键词检索为主: 不能满足长期记忆质量要求
- 全量改成图检索: 架构收益存在，但本次范围过大

### Decision 6: Prompt 注入按记忆类型分流，并显式退役旧注入方式

**选择:**
- `user_memory` 注入到 system/policy 侧，作为行为约束
- `knowledge_memory` 注入到 context 侧，附带 evidence 优先规则
- 现有“将所有长期记忆拼成一段文本”的主链路注入方式退役

**理由:**
- 用户偏好和知识证据在 prompt 中的语义角色不同
- 分流后更容易控制长度、优先级和冲突处理
- 可以让旧的 `inject_long_term_memory` 与关键词结果拼接逻辑退出主链路

**备选方案:**
- 继续统一拼接成 `[长期记忆]` 段落: 迁移成本低，但表达能力和可控性差

### Decision 7: 老代码采用“替换后删除”的治理策略，而不是长期双写双读

**选择:** 新 pipeline 达到可用后，移除旧的同步片段写入、关键词主检索和仅做转发的 facade；仅在短期迁移窗口内保留必要兼容读取。

**理由:**
- 项目已有明确约束：不长期保留历史兼容 facade 和无独立语义的 wrapper
- 双主链路会让行为不确定，测试与文档也会持续分裂
- 长期 memory 系统必须存在单一 canonical 入口

**备选方案:**
- 长期双写双读: 回滚轻松，但会让技术债长期固化
- 立即硬切删除所有旧路径: 风险过高，不利于平滑验证

## Risks / Trade-offs

- [异步任务失败导致记忆缺失] → 为 memory job 增加任务状态、重试和幂等写入约束，并允许后续补偿重跑
- [结构化抽取质量不稳定] → 对 memory writer 输出建立 schema 校验、低置信度丢弃和 evidence 必填规则
- [旧数据与新模型并存期间语义不一致] → 明确迁移窗口内的兼容读取边界，并禁止继续沿用旧写入主链路
- [语义检索引入额外依赖和成本] → 优先复用项目已有 embedding 能力，关键词检索仅作为受控降级
- [去重与替代逻辑过于复杂] → 先支持最关键的动作集合和两类顶层记忆，避免首版过度抽象
- [清理老代码时引发回归] → 为旧路径退役补充回归测试和文档更新，按模块逐步删除而非一次性大改

## Migration Plan

### Phase 1: 数据模型和事件落盘
1. 新增 episode、memory item、evidence 所需表或字段
2. 将当前对话完成后的同步 `persist_turn_memory` 改为 raw episode 持久化
3. 在对话完成后投递异步 memory writer job

### Phase 2: 异步抽取与 reconcile
1. 实现 memory writer worker 输入/输出 schema
2. 实现 user memory 与 knowledge memory 的候选抽取
3. 实现 dedup key、近似匹配、状态迁移和 supersede 逻辑

### Phase 3: 检索与 prompt 注入切换
1. 将 controller 的长期记忆检索切到 typed retrieval
2. 将 prompt 注入改为 user memory / knowledge memory 分流
3. 将关键词检索降级为非主路径 fallback

### Phase 4: 旧代码退役
1. 删除旧的同步片段写入逻辑
2. 删除关键词主检索与旧注入主路径
3. 清理 `store` / `utils` 中无独立语义的 memory facade
4. 更新 README、架构说明和测试

### Rollback 策略
- 保留数据库迁移的向后兼容读取能力
- 在切换期通过 feature flag 或配置开关临时回退到旧读取链路
- 若异步 worker 稳定性不足，可先保留 raw episode 落盘并暂停结构化写入步骤

## Open Questions

- 初版语义召回是直接复用现有 embedding 组件，还是单独为 memory item 建立更轻量的向量索引
- 是否需要为用户提供显式的记忆查看、删除和禁用入口
- 历史 `memory_items` 是否做离线迁移，还是仅保留只读兼容并让新数据走新模型
