# AGENTS.md

PaperSage 项目开发规范与工程化约束（团队约定版）

最后更新：2026-03-08

---

## 1. 目标

本规范用于约束后续开发，避免架构继续劣化，确保：

1. 分层清晰
2. 可测试、可维护
3. 变更可回滚
4. 质量门禁可执行

---

## 2. 目录与分层边界

当前主目录职责：

1. `pages/`：Streamlit 页面入口（薄层，仅处理页面交互与编排调用）
2. `ui/`：可复用 UI 组件与页面渲染函数
3. `agent/`：
   - `domain/` 领域模型与契约
   - `application/` 用例编排
   - `adapters/` 外部依赖适配
   - `orchestration/` 调度策略
   - `a2a/` 协作协议/状态机
4. `utils/`：遗留兼容与通用能力（逐步收敛）
5. `tests/`：单元 / 集成 / eval
6. `docs/plan/`：重构与治理计划

依赖方向（必须遵守）：

1. `pages/ui -> agent.application -> agent.domain`
2. `agent.application -> agent.adapters`
3. `agent.adapters -> infra/repository`
4. 禁止 `domain` 依赖 `ui/pages`

---

## 3. 强制约束（MUST）

1. 新业务逻辑禁止写入 `utils/utils.py`。
2. 新增数据库访问必须放在 repository/adapter 层，不允许页面直接写 SQL。
3. 页面层禁止直接调用 LLM SDK（如 `OpenAI(...)`），统一走 adapter/service。
4. 不允许在业务代码中使用 `sys.path.insert` 修补导入路径。
5. 任何功能改动必须包含最小测试或现有测试更新。
6. 涉及配置、行为变更必须更新文档（`README.md` 或 `docs/`）。
7. 不允许硬编码 API Key、Token、密钥。
8. 所有新增函数必须有类型注解（至少参数与返回值）。
9. 错误处理禁止静默 `except Exception: pass`。
10. 所有跨层调用要有清晰命名，禁止“万能工具函数”继续扩散。

---

## 4. 禁止事项（MUST NOT）

1. 不允许新增“巨型文件”：
   - 单文件超过 800 行时，默认必须拆分并在 PR 描述说明原因。
2. 不允许在 `pages/` 重复初始化逻辑（DB、用户、session_state）。
3. 不允许在 adapter 中仅做无意义透传（直接 `return utils.xxx`）而不定义清晰接口边界。
4. 不允许新增全局可变状态，除非明确封装在 session/context 对象内。
5. 不允许提交与当前任务无关的大规模格式化噪音。

---

## 5. 代码风格与实践（SHOULD）

1. 单函数建议控制在 60-80 行以内，超出时拆成私有辅助函数。
2. 复杂逻辑优先“先定义输入输出 schema，再实现”。
3. 使用早返回减少嵌套层级。
4. I/O、LLM 调用、DB 调用与纯逻辑分离。
5. 日志采用结构化信息（至少包含关键上下文：`uuid/project_uid/session_uid`）。
6. 命名清晰表达语义，避免 `data/temp/obj`。
7. 优先组合而非继承，优先协议/接口而非硬耦合实现。

---

## 6. 测试策略

默认策略：

1. 纯逻辑改动：至少补 1 个单元测试。
2. 跨模块编排改动：单元测试 + 至少 1 个集成测试。
3. 修复 bug：必须新增能复现并防回归的测试。

建议命令：

```bash
# 核心质量门禁（必跑）
bash scripts/quality_gate.sh core

# 目标测试
uv run --extra dev python -m pytest tests/unit -q

# 指定模块测试
uv run --extra dev python -m pytest tests/unit/test_turn_engine.py -q
```

---

## 7. 提交与评审规范

PR 描述至少包含：

1. 背景与问题
2. 改动范围（文件列表）
3. 风险点
4. 回滚方式
5. 测试结果（命令 + 结果）

评审检查清单：

1. 是否破坏分层边界
2. 是否引入跨层反向依赖
3. 是否新增重复初始化或重复逻辑
4. 是否补齐测试与文档
5. 是否存在潜在安全泄露

---

## 8. 架构演进要求（针对当前项目）

1. `utils/utils.py` 只减不增：新能力必须进新模块。
2. 页面初始化统一收敛到 bootstrap helper。
3. `ui/agent_center_page.py` 继续拆分为 controller / view / state。
4. worker 任务导入路径标准化，去除路径 hack。
5. 质量门禁逐步扩围到 `ui/pages/utils`。

详见：

1. `docs/plan/2026-03-08-项目架构治理与重构计划.md`

---

## 9. 安全与配置

1. 仅从环境变量、用户配置或安全存储读取密钥。
2. 日志中禁止打印完整密钥与敏感凭据。
3. 对外部输入（文件、prompt、路径）做基本校验。
4. 涉及文件读写时校验路径合法性，避免越权访问。

---

## 10. Definition of Done（完成标准）

一个任务被视为“完成”，必须同时满足：

1. 功能达成且无已知阻塞 bug
2. 通过对应质量门禁与测试
3. 文档已更新
4. 无新增架构债务（或已在 PR 中明确登记和偿还计划）
5. 评审意见闭环

---

## 11. 快速决策规则

当遇到“快修 vs 工程化”冲突时：

1. 线上故障先止血，但必须在同 PR 或后续紧邻 PR 补工程化修复。
2. 非紧急需求优先按分层规范实现，不接受一次性脚本式堆叠。
3. 无法一次做完的大改动，按“可回滚的小步提交”推进。
