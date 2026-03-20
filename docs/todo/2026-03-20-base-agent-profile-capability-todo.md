# Base Agent + Profile/Capability 重构 Todo

关联设计文档：`docs/plan/2026-03-20-BaseAgent-ProfileCapability-重构方案.md`

## 总目标

将当前以 `agent/paper_agent.py` 为中心的 leader 入口，重构为：

1. 一个 base runtime / session factory
2. 一组显式 profile
3. 一组按语义装配的 capability pack
4. leader 与 teammate 在创建时完成权限隔离

---

## Phase 1：抽取基础装配骨架

### 目标

把“如何创建 agent”与“创建什么 agent”拆开，同时保持现有调用链兼容。

### Todo

- [x] 新增 `agent/session_factory.py`
- [x] 定义统一的 `create_agent_session(...)` 入口
- [x] 定义 `AgentSession` / `AgentDependencies` / `AgentRuntimeOptions` 基础数据结构
- [x] 把 `thread_id`、checkpointer、tool schema level、model 装配逻辑迁移到 session factory
- [x] 新增 `agent/profiles.py`
- [x] 定义最小 `AgentProfile` 结构
- [x] 定义 `paper_leader_profile`
- [x] 让 `agent/paper_agent.py` 退化为兼容 facade，内部调用 `create_agent_session(profile=paper_leader_profile, ...)`
- [x] 保持现有 `create_paper_agent_session(...)` 外部调用点不变

### 验收

- [x] 现有 `paper_agent` 相关单测可继续通过
- [x] 现有主链路无需大规模改调用点
- [x] `paper_agent.py` 不再承担完整装配职责

---

## Phase 2：拆出 Prompt 分层

### 目标

把当前混合在 `paper_agent.py` 中的大 prompt 拆成 base / domain / role 三层。

### Todo

- [x] 新增 `agent/prompts/base.py`
- [x] 提取通用基础约束
- [x] 新增 `agent/prompts/paper_domain.py`
- [x] 提取论文阅读 / 文档问答领域约束
- [x] 新增 `agent/prompts/leader.py`
- [x] 定义 leader 的角色增量 prompt
- [x] 新增 `agent/prompts/worker.py`
- [x] 定义 worker 的角色增量 prompt
- [x] 新增 `agent/prompts/reviewer.py`
- [x] 定义 reviewer 的角色增量 prompt
- [x] 在 profile 中接入 prompt builder，而不是直接内联超长字符串

### 验收

- [ ] base/domain 内容语义与当前主 prompt 保持一致
- [x] leader prompt 明确对话 ownership 与调度责任
- [x] worker prompt 明确“不接管用户对话、不创建下级 agent”

---

## Phase 3：引入 Capability Pack

### 目标

把工具与中间件从“默认通吃”改为“按角色装配”。

### Todo

- [x] 新增 `agent/capabilities/document.py`
- [x] 封装 `search_document`、`read_document`、`list_document` 的装配逻辑
- [x] 新增 `agent/capabilities/planning.py`
- [x] 封装 plan / todos / scheduler hint 的装配逻辑
- [x] 新增 `agent/capabilities/team.py`
- [x] 封装 team tools 的装配逻辑
- [x] 新增 `agent/capabilities/skill.py`
- [x] 封装 `use_skill` 相关装配逻辑
- [x] 视情况新增 `agent/capabilities/web.py`
- [x] 让 profile 通过 capability id 列表决定工具集合
- [x] 避免在 leader 和 worker 之间显式传递 tool 列表

### 验收

- [ ] `document_pack` 可供 leader 与 worker 复用
- [ ] `planning_pack` 默认只给 leader
- [ ] `team_pack` 默认只给 leader
- [ ] `skill_pack` 可按角色受限开放

---

## Phase 4：收敛 Middleware 装配边界

### 目标

让 middleware 不再隐式假设“所有 agent 都是 leader”。

### Todo

- [x] 盘点 `agent/middlewares/builder.py` 当前默认挂载的 middleware
- [x] 区分 base middleware 与 role-specific middleware
- [x] 把 team middleware 调整为仅 leader profile 挂载
- [x] 评估 plan / todolist middleware 是否仅 leader 挂载
- [x] 保留 trace / llm logger / retry 等基础 middleware 给所有 profile
- [ ] 明确 orchestration middleware 在 leader 与 worker 上的行为差异

### 验收

- [x] worker 默认没有 team middleware 暴露的工具
- [x] worker 不会被注入 leader 导向的调度提示
- [x] leader 仍能获得 handoff / scheduler convenience

---

## Phase 5：引入 Worker / Reviewer Profile

### 目标

让 teammate 从“共享一套通用 agent”升级为“显式角色实例”。

### Todo

- [x] 在 `agent/profiles.py` 中新增 `paper_worker_profile`
- [x] 在 `agent/profiles.py` 中新增 `reviewer_profile`
- [ ] 如有需要新增 `researcher_profile`
- [x] 为不同 profile 定义 capability pack 组合
- [x] 为不同 profile 定义 prompt builder
- [ ] 明确每个 profile 的运行约束和输出格式

### 验收

- [x] leader 拥有 team/planning/document 等能力
- [x] worker 只有执行任务所需的最小能力
- [x] reviewer 聚焦 review，不拥有 team/global planning 权限

---

## Phase 6：改造 Team Spawn 路径

### 目标

让 `spawn_agent` 基于 profile 创建子 agent，而不是隐式复用 leader 能力面。

### Todo

- [x] 审视 `agent/tools/team.py` 当前 `spawn_agent(name, system_prompt)` 接口
- [x] 设计新的 spawn 输入结构，至少包含 `profile` 或 `role`
- [x] 让 team runtime 通过 profile 创建子 agent session
- [x] 保持 worker 使用独立 `thread_id`
- [x] 确保 worker 默认 `tools=[]` 的旧策略升级为“按 worker profile 装配”，而不是彻底无工具
- [x] 避免 worker 获取 team pack
- [x] 让 `send_message / get_agent_result / close_agent` 仍兼容当前 team runtime

### 验收

- [x] `spawn_agent` 生成的 worker 是独立 session 实例
- [x] worker 具备最小执行能力，而不是空壳
- [x] worker 无法继续递归分派 teammate

---

## Phase 7：定义 Worker 输出契约

### 目标

防止 worker 把整段会话历史返回给 leader，污染 leader 上下文。

### Todo

- [ ] 定义统一 worker result schema
- [ ] 明确 `task_id / status / summary / evidence / risks / artifacts` 等字段
- [ ] 在 team runtime 中统一整理 worker 返回值
- [ ] 让 leader 消费结构化结果，而不是自由文本历史
- [ ] 为 reviewer 返回结构定义单独约束

### 验收

- [ ] leader 获取的是结构化中间产物
- [ ] worker 的内部工具调用历史不会直接灌回 leader 上下文
- [ ] leader 更容易做 review / replan / final synthesis

---

## Phase 8：补重复检索与收敛机制

### 目标

解决当前 live 下重复 `search_document` 同 query 的问题，并提升 leader 协作收敛性。

### Todo

- [ ] 在 `document_pack` 或 `agent/tools/document.py` 中增加相同 query 防重复策略
- [ ] 对“同 query 连续命中”输出明确短路提示
- [ ] 记录详细日志，便于判断是否进入重复检索
- [ ] 明确 evidence 不足时的改写 query / 转入计划 / 转入总结规则
- [ ] 当 `needs_team=True` 时，提升 leader 进入 todo / team 路径的优先级

### 验收

- [ ] 不会无限重复搜索相同 query
- [ ] leader 在复杂任务下更容易收敛到 todo / dispatch / review 路径
- [ ] live 日志可明确看出阶段转换

---

## Phase 9：测试与回归保护

### 单元测试

- [ ] `session_factory` 创建不同 profile 时工具集合正确
- [x] leader profile 包含 team pack
- [x] worker/reviewer profile 不包含 team pack
- [x] prompt 分层合并结果正确
- [ ] worker result schema 正常
- [ ] document 防重复搜索策略有回归测试

### 集成测试

- [x] leader 创建 worker 后使用独立 thread
- [ ] worker 无法调用 team tools
- [ ] leader 能读取 worker 结构化结果
- [ ] 当前 `turn_engine` 不回归
- [ ] 当前 `agent_center` 主链路不回归

### Live / E2E

- [ ] 复杂任务命中 `team_handoff` 后能看到更合理的执行轨迹
- [ ] 相同 query 不会重复 search
- [ ] leader 最终能正常收敛输出

---

## Phase 10：文档与迁移收口

### Todo

- [ ] 更新 README 中对 leader / teammate 架构的描述
- [ ] 更新 README_EN 中对应描述
- [ ] 如实现方案与现有 OpenSpec 有偏差，更新 `openspec/changes/upgrade-leader-teammate-orchestration/` 相关文档
- [ ] 补充迁移说明：旧入口如何兼容、新入口如何使用

### 验收

- [ ] 文档描述与代码真实能力一致
- [ ] 不再把 `paper_agent.py` 描述成唯一形态的主 agent

---

## 建议执行顺序

建议按以下优先级推进：

1. Phase 1：抽装配骨架
2. Phase 2：拆 prompt 分层
3. Phase 3：引入 capability pack
4. Phase 4：收敛 middleware 装配边界
5. Phase 5：引入 worker/reviewer profile
6. Phase 6：改造 team spawn 路径
7. Phase 7：定义 worker 输出契约
8. Phase 8：补重复检索与收敛机制
9. Phase 9：补全测试
10. Phase 10：文档收口

---

## Definition of Done

- [ ] `paper_agent.py` 不再承担完整 leader 运行时装配职责
- [ ] leader 与 worker 的能力边界在创建时完成隔离
- [ ] team tools 不再暴露给 worker
- [ ] worker 具备独立 session / thread
- [ ] 不再需要沿链路传递 tool 套件
- [ ] 重复相同 query 的搜索被抑制
- [ ] 关键单测、集成测试、live 验证通过
- [ ] 文档与 README 更新完成
