# Base Agent + Profile/Capability 重构方案

日期：2026-03-20

作者：Codex

## 1. 背景与问题

当前项目中的主入口 agent 事实上是 `agent/paper_agent.py`。它同时承担了以下职责：

1. 定义论文问答主 system prompt
2. 组装文档、计划、team 等工具
3. 创建 session / thread / checkpointer
4. 作为 leader 的运行时入口
5. 间接决定 subagent 可以继承哪些能力

这种结构在早期迭代中能快速推进，但在当前 `leader + teammate` 模式下已经出现明显问题：

1. `paper_agent` 同时承担“基础 agent 能力”和“leader 专属能力”，角色边界不清
2. leader 与 teammate 的工具集合没有物理隔离，更多依赖 prompt 约束
3. 运行时容易把“当前 agent 有哪些工具”沿链路传播，产生隐式耦合
4. prompt 中混合了通用问答、检索策略、复杂任务处理、团队协作等多种心智，互相竞争
5. 扩展新角色时，往往只能继续堆参数或增加条件分支，架构持续变重

从当前目标看，我们需要的不是“继续把 paper_agent 做大”，而是把它下沉成一个可复用的基础运行时，再把 leader、worker、reviewer 等角色显式建模。

## 2. 重构目标

本次重构的目标不是引入更多抽象层，而是建立清晰、最小且可落地的角色装配机制：

1. 形成一个 `Base Agent Runtime`，只负责通用 session 创建与运行时拼装
2. 通过 `AgentProfile` 显式定义不同 agent 的角色、prompt、能力边界
3. 通过 `Capability Pack` 管理工具与 middleware 的成组装配
4. 让 leader 与 teammate 在“创建时”就完成权限隔离，而不是在对话中靠提示词约束
5. 避免把 tool 列表、tool ownership 在多个链路之间来回传递
6. 保持当前 `leader + teammate` 模式，不演化成自动 swarm
7. 保留现有 `thread_id` / checkpoint / middleware 体系，避免一次性推翻

## 3. 非目标

以下内容不在本次重构的第一阶段目标内：

1. 不引入全新的 agent 框架
2. 不把 `turn_engine` 改成自动 orchestrator
3. 不做“所有角色都可继续 spawn 子 agent”的递归团队模型
4. 不为了整齐增加无语义 wrapper 或 alias 层
5. 不在第一阶段重写所有 prompt，只拆出基础层和角色增量层

## 4. 设计原则

### 4.1 Simple Is Better

本次设计遵循现有工程规范中的“simple is better”：

1. 一个 canonical 创建入口
2. 少量 profile
3. 少量 capability pack
4. runtime 负责“怎么跑”，profile 负责“能做什么”

### 4.2 权限边界优先于提示词边界

如果某个角色不应该拥有某项能力，优先做“物理剥离”，而不是只写提示词。

例如：

1. leader 可以有 team tools
2. teammate 默认不应拥有 team tools
3. worker 默认不应拥有全局计划治理能力

### 4.3 profile 是角色语义，不是参数集合

profile 不是简单的 `enable_team=True`、`enable_plan=False` 参数包。它应该表达清晰的角色语义，例如：

1. `leader`
2. `paper_worker`
3. `reviewer`
4. `researcher`

### 4.4 capability pack 是装配单元，不是传输载荷

tool ownership 应在 agent 创建时确定，而不是通过消息在 leader 和 worker 之间传递。

## 5. 目标架构

### 5.1 新的核心模型

建议引入三个核心概念：

1. `AgentSessionFactory`
2. `AgentProfile`
3. `CapabilityPack`

它们的职责如下。

#### AgentSessionFactory

职责：

1. 创建 `thread_id`
2. 创建或接入 checkpointer
3. 构建 model
4. 合并 middleware
5. 合并工具
6. 返回统一的 `AgentSession`

不负责：

1. 决定具体角色
2. 决定是否允许 team、plan、web
3. 决定 prompt 业务语义

#### AgentProfile

职责：

1. 定义角色身份
2. 定义 base prompt 之上的角色增量 prompt
3. 定义允许的 capability packs
4. 定义允许的 middleware packs
5. 定义运行策略与约束

示例字段：

```python
@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str
    prompt_builder: PromptBuilder
    capability_ids: tuple[str, ...]
    middleware_ids: tuple[str, ...]
    allow_team: bool = False
    allow_global_planning: bool = False
    allow_web: bool = False
```

#### CapabilityPack

职责：

1. 提供一组有共同语义的工具
2. 在必要时提供对应 middleware
3. 封装该能力所需的依赖

典型 capability pack：

1. `document_pack`
2. `planning_pack`
3. `team_pack`
4. `web_pack`
5. `skill_pack`

## 6. 推荐目录结构

建议在 `agent/` 内逐步收敛到如下结构：

```text
agent/
  session_factory.py
  profiles.py
  prompts/
    base.py
    leader.py
    worker.py
    reviewer.py
  capabilities/
    document.py
    planning.py
    team.py
    web.py
    skill.py
  runtime/
    agent_session.py
    middleware_builder.py
    tool_builder.py
```

如果不希望一次性新增太多目录，可以采用渐进式方案：

1. 先增加 `agent/session_factory.py`
2. 先增加 `agent/profiles.py`
3. 先增加 `agent/prompts/`
4. `capabilities/` 先只整理装配，不要求迁移所有工具文件

也就是说，第一阶段可以允许 `capabilities/document.py` 只是“对现有 document tool builder 的有语义装配”，但不能只是简单透传而没有边界说明。

## 7. 角色拆分方案

### 7.1 Base Agent

Base Agent 不是面向用户直接对话的角色，而是一个“运行时装配基类”。

它具备的只是通用能力：

1. session 生命周期
2. checkpoint
3. thread_id
4. 统一日志上下文
5. middleware / tools 的装配管线

它不直接拥有：

1. team tools
2. 复杂任务调度语义
3. 具体角色提示词

### 7.2 Leader Agent

Leader 是当前对话的 owner。

职责：

1. 理解用户意图
2. 判断是否需要检索、计划、团队协作
3. 决定是否创建 todos
4. 决定是否 dispatch teammate
5. 负责 review / replan / final answer

建议能力：

1. `document_pack`
2. `planning_pack`
3. `team_pack`
4. `skill_pack`
5. 可选 `web_pack`

Leader 特点：

1. 有 team tools
2. 有 todo / planning 能力
3. 对当前用户对话拥有唯一 ownership

### 7.3 Worker Agent

Worker 是由 leader 派生出的执行角色，不拥有用户对话主导权。

职责：

1. 执行单个任务
2. 返回结构化结果
3. 不接管当前会话

建议能力：

1. `document_pack`
2. 可选 `skill_pack`

不建议能力：

1. `team_pack`
2. 全局 `planning_pack`

Worker 特点：

1. 默认不可继续分派新的 teammate
2. 默认不可管理全局 todo
3. 结果是中间产物，不是最终答复

### 7.4 Reviewer Agent

如果需要独立评审角色，建议定义为单独 profile，而不是继续复用 worker。

职责：

1. 基于 leader 或 worker 的产出做审查
2. 聚焦发现缺失、冲突、证据不足
3. 输出 review 结果而非最终面向用户的答复

建议能力：

1. `document_pack`
2. 可选 `skill_pack`

不建议能力：

1. `team_pack`
2. 全局 `planning_pack`

## 8. Prompt 分层方案

当前问题之一是主 prompt 中混合了通用问答、检索、复杂任务处理、团队协作等多重规则。建议拆成三层：

### 8.1 Base Prompt

放所有通用约束：

1. 输出语言跟随用户
2. 结论要有证据
3. 安全与基础行为约束

### 8.2 Domain Prompt

放论文阅读/文档问答领域约束：

1. 优先使用文档检索
2. 如何引用证据
3. 文献相关能力说明

### 8.3 Role Prompt Addon

按角色附加：

`leader`：

1. 你负责调度与最终回答
2. 你决定是否需要团队分工
3. 你决定何时 plan / todo / review / replan

`worker`：

1. 你负责完成单个任务
2. 不接管用户对话
3. 不创建下级 agent
4. 输出结构化任务结果

`reviewer`：

1. 你负责评审现有产出
2. 重点识别风险、缺证、冲突
3. 不负责重新组织用户对话

## 9. Capability Pack 设计

### 9.1 document_pack

包含：

1. `search_document`
2. `read_document`
3. `list_document`

约束：

1. 文档工具对 leader 与 worker 都可复用
2. 防止循环搜索的策略也应归属在这一 pack 内

### 9.2 planning_pack

包含：

1. plan tools
2. todo tools
3. todo scheduler hints

只建议给 leader。

原因：

1. todo 是全局流程状态，不应由任意 worker 改写
2. worker 只需拿任务执行，不应持有全局治理权

### 9.3 team_pack

包含：

1. `spawn_agent`
2. `send_message`
3. `get_agent_result`
4. `list_agents`
5. `close_agent`

只建议给 leader。

原因：

1. 当前架构是 `leader + teammate`
2. 不需要 teammate 再递归扩散团队

### 9.4 skill_pack

包含：

1. `use_skill`
2. 可能的 skill 选择逻辑

可按角色细分：

1. leader 可用全部通用 skill
2. worker 只开放与执行任务直接相关的 skill

### 9.5 web_pack

仅给需要外部信息的 leader 或特定 researcher。

默认不建议所有 worker 都开放 web。

## 10. Session 与 Thread 设计

### 10.1 Leader Session

leader 持有主会话 thread，例如：

```text
paper-qa-<uuid>
```

### 10.2 Worker Session

worker 在 leader 当前 team runtime 下生成独立 thread，例如：

```text
team:{leader_thread}:{agent_id}
```

这样可以保证：

1. worker 有独立上下文与 checkpoint
2. worker 不污染 leader 的完整消息历史
3. 仍然可以从属于 leader 当前 team runtime

### 10.3 返回结果原则

worker 返回给 leader 的应该是结构化结果，而不是完整会话历史。

建议格式：

```json
{
  "task_id": "todo-2",
  "status": "completed",
  "summary": "方法A 精度更高，方法B 延迟更低",
  "artifacts": [],
  "evidence": [],
  "risks": []
}
```

这有助于：

1. 减少 leader 上下文污染
2. 让 leader 更容易 review 与 replan
3. 保持最终答复 ownership 在 leader

## 11. 创建入口设计

建议最终保留一个 canonical 入口：

```python
create_agent_session(
    profile=leader_profile,
    deps=AgentDependencies(...),
    options=AgentRuntimeOptions(...),
)
```

其中：

`profile` 决定：

1. 角色语义
2. prompt
3. capability packs
4. middleware packs

`deps` 提供：

1. 文档检索函数
2. evidence retriever
3. list/read document
4. project/session/user 上下文

`options` 提供：

1. model
2. checkpointer
3. schema level
4. 是否启用某些可选能力

这能避免现在 `create_paper_agent_session(...)` 不断新增参数，最终变成“大而全构造器”。

## 12. 与当前代码的映射关系

### 12.1 `paper_agent.py`

当前建议拆解为：

1. Base session factory
2. paper leader profile
3. base prompt + paper domain prompt

第一阶段不要求彻底删除 `paper_agent.py`，可以让它退化成兼容入口：

```python
def create_paper_agent_session(...):
    return create_agent_session(profile=paper_leader_profile, ...)
```

这样可以减少调用点一次性爆炸。

### 12.2 `runtime_agent.py`

建议保留其“创建 runtime agent”的职责，但去掉隐式角色假设。

它应只负责：

1. 接收 tools
2. 接收 middleware
3. 接收 system prompt
4. 创建 agent

不应负责：

1. 决定某角色有哪些工具
2. 决定某角色是否允许 team

### 12.3 `team/runtime.py` 与 `tools/team.py`

保留其进程内 team runtime 的实现，但把调用入口显式绑定到 leader profile。

重点不是重写 team runtime，而是停止把 team tools 暴露给所有 agent。

## 13. 迁移路径

建议按四个阶段推进。

### 阶段 1：抽取装配骨架

目标：

1. 新增 `create_agent_session`
2. 新增 `AgentProfile`
3. 新增 `paper_leader_profile`
4. 让 `create_paper_agent_session` 转为兼容 facade

收益：

1. 先把“如何创建 agent”与“创建什么 agent”拆开

### 阶段 2：引入 capability packs

目标：

1. 抽出 `document_pack`
2. 抽出 `planning_pack`
3. 抽出 `team_pack`
4. 由 profile 决定装哪些 pack

收益：

1. tool ownership 明确
2. 不再需要跨链路传 tool 套件

### 阶段 3：引入 worker/reviewer profiles

目标：

1. 新增 `paper_worker_profile`
2. 新增 `reviewer_profile`
3. `spawn_agent` 时显式指定 profile

收益：

1. teammate 能力边界从“prompt 约束”升级为“创建时隔离”

### 阶段 4：收敛 leader 协作流

目标：

1. 让 leader 更稳定地从 `team_handoff` 进入 todo / review / replan 路径
2. 补充重复检索保护与任务阶段收敛

收益：

1. `leader + teammate` 体验更稳定
2. live 下不再轻易进入工具循环

## 14. 测试策略

本次重构至少需要以下测试：

### 14.1 单元测试

1. profile 装配出的 tool 集合正确
2. leader 包含 team pack，worker 不包含
3. prompt 分层合并正确
4. session factory 为不同 profile 生成正确 session 配置

### 14.2 集成测试

1. leader 创建 worker 后，worker 使用独立 thread
2. worker 无法调用 team tools
3. leader 能正常读取 worker 结果
4. 当前 `turn_engine` 与 agent center 主链路不回归

### 14.3 live / E2E 验证

1. 复杂任务命中 `team_handoff` 后，leader 能进入更合理的执行路径
2. 重复相同 query 不会无限循环

## 15. 风险与控制

### 风险 1：抽象过多

控制方式：

1. 保持一个 canonical 创建入口
2. profile 数量从少量角色开始
3. capability pack 只在真正有语义边界时抽取

### 风险 2：兼容链路回归

控制方式：

1. 第一阶段保留 `create_paper_agent_session` facade
2. 优先在内部切换实现，不一次性修改所有调用点

### 风险 3：worker 权限切太狠导致任务能力不足

控制方式：

1. 先定义最小 worker
2. 按角色增补必要 capability
3. 不直接开放 team/global planning

### 风险 4：prompt 拆分后行为漂移

控制方式：

1. 先保持 base/domain 内容不变
2. 只把角色增量部分拆出
3. 用回归测试锁住关键行为

## 16. 推荐落地顺序

如果只按最小可行路径推进，建议严格按以下顺序实施：

1. 抽 `session_factory`
2. 引入 `paper_leader_profile`
3. 让 `paper_agent.py` 退化为兼容入口
4. 把 team tools 只挂到 leader profile
5. 引入 `worker_profile`
6. 改 `spawn_agent` 以 profile 创建 worker
7. 再优化 prompt 收敛与搜索防重

## 17. 结论

本次重构的核心不是“把 paper_agent 改名”，而是明确三件事：

1. Base runtime 负责运行时拼装
2. Profile 负责角色语义与权限边界
3. Capability pack 负责能力装配与 ownership

在这个结构下：

1. leader 能拥有 team / planning 能力
2. teammate 不需要继承 leader 全部工具
3. tool 不再需要在多条链路中隐式传递
4. `leader + teammate` 会比当前结构更清晰、更稳、更容易扩展

这条路线与当前仓库的分层规范一致，也能最小化对现有 turn engine、middleware、team runtime 的破坏。
