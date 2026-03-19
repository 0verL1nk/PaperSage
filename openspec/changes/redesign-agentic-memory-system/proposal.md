## Why

当前项目的长期记忆仍以截断后的 `Q/A` 文本片段为主存储形态，并通过关键词重叠与时间衰减进行检索。这种实现无法稳定支持用户偏好、论文事实、项目知识等长期记忆场景，也缺少结构化更新、冲突处理和可靠去重能力。

随着项目向多 Agent、长会话和项目级知识沉淀演进，记忆系统需要从“对话片段缓存”升级为“异步、结构化、可对账的长期记忆管线”，避免主链路阻塞，并让记忆具备可追溯、可更新、可检索的工程属性。

## What Changes

- 将现有同步 `persist_turn_memory` 逻辑改造为“原始 episode 落盘 + 异步 memory job 投递”的双阶段流程
- 新增长期记忆结构化抽取流程，区分用户偏好/指令类记忆与知识类记忆
- 新增记忆对账与生命周期动作，支持 `ADD`、`UPDATE`、`DELETE`、`NONE`、`SUPERSEDE`
- 为长期记忆补充 canonical text、dedup key、evidence、confidence、status 等结构化字段
- 将长期记忆检索从关键词优先改为类型化检索与语义召回优先
- 调整 prompt 注入策略，对用户记忆与知识记忆采用不同注入位置与优先级
- 替换并移除旧的同步片段写入、关键词主检索与仅透传式兼容入口，避免新旧主链路长期并存
- **BREAKING**: 长期记忆的主链路语义将从“聊天片段存储/搜索”切换为“结构化 memory item 管理/检索”

## Capabilities

### New Capabilities

- `agentic-memory-system`: 定义长期记忆的异步抽取、结构化存储、去重对账、状态迁移与类型化检索能力

### Modified Capabilities

<!-- 无现有 capability 需要修改 -->

## Impact

**代码影响**:
- `agent/application/agent_center/memory.py` - 从同步对话片段写入切换到 episode 持久化与异步任务投递
- `agent/application/agent_center/page_orchestrator.py` - 对话完成后触发 memory pipeline
- `agent/memory/repository.py` - 扩展记忆表结构，新增 episode / evidence / lifecycle 支持
- `agent/memory/service.py` - 重构长期记忆检索与对账逻辑
- `agent/memory/store.py` / `utils/utils.py` - 清理旧的 memory facade 与兼容入口
- `agent/application/agent_center/controller.py` - 按记忆类型调整 prompt 注入策略
- `utils/task_queue.py` / `utils/tasks.py` - 新增异步 memory writer 任务

**数据影响**:
- 新增长期记忆相关结构化字段与原始 episode 数据
- 现有 `memory_items` 数据模型需要迁移或兼容读取
- 记忆检索索引和 dedup key 需要持久化维护

**依赖与运行时影响**:
- 复用现有异步任务基础设施，不要求新增外部队列服务
- 需要引入或复用 embedding / rerank 能力支撑语义召回
- 异步 memory job 的失败、重试、幂等需要纳入运行时治理
