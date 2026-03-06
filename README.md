# 文献阅读助手

[English](README_EN.md) | 简体中文

> 面向科研阅读与写作场景的 AI 智能体应用，基于 **Streamlit + LangChain + LangGraph** 构建。  
> 以"项目化论文问答工作台"为核心：按项目组织文档、限定检索范围、自动路由 Agent 工作流、输出可追溯证据。

---

## ✨ 核心能力

### 1. 多模式 Agent 工作流（自动路由）

| 模式 | 适用场景 |
|------|---------|
| `ReAct`（Tool + Memory） | 简单问答、单跳检索、快速事实确认 |
| `Plan-Act`（Orchestration） | 中等复杂任务，先 LLM 规划再多角色协同执行 |
| `Plan-Act-RePlan`（Orchestration） | 高复杂度任务，规划→团队执行→复核→重规划 |

- **智能路由**：优先关键词快速路由，回退 LLM 路由，再兜底 `ReAct`；复杂度评分由多维度信号（句子数、问号数、并列结构等）驱动，阈值全部可通过环境变量调整。
- **结构化 Orchestration**：独立 `agent/orchestration/` 子模块，含 `planning_service`（LLM 生成执行计划）、`policy_engine`（复杂度评分与策略决策）、`team_runtime`（动态角色分配与多轮执行）、`orchestrator`（统一对外入口）。
- **Leader 中心调度**：默认 `team=false` 时，本轮对话只有 `Leader` 执行；开启团队后采用“`Leader` 中心双向网状”协作，角色只与 `Leader` 交互，不做角色间直接串联，便于治理与回放。
- **Team Todo 依赖调度**：团队执行阶段会自动生成 todo 任务图（含 `assignee/dependencies/plan_id/step_ref/history`），按依赖关系逐步派发给子角色；结果回填到 `team_execution.todo_records` 供 UI 与追踪使用。
- **调度拓扑（当前主链路）**：

```text
user -> leader(request)
      -> policy_engine(policy)
      -> planner(plan, optional)
      -> leader <-> role_1..role_n (dispatch/review, multi-round, optional)
      -> leader -> user(final)
```

- **协议分层**：内部编排事件统一走 `sender/receiver/performative` + `run_id/task_id/span_id`；同时镜像 `a2aMessage`（SDK `Message`）用于观测与后续桥接。对外 A2A 接口由 `agent/a2a/standard.py` 提供。
- **统一对话轮次服务**：`agent/application/turn_engine.py` 封装单轮完整执行逻辑，`agent/turn_service.py` 提供兼容入口，包含证据归一化、阶段标签聚合、方法对比解析。
- **会话上下文治理**：超 Token 阈值自动压缩历史消息，支持 LLM 压缩与关键事实锚点提取。
- **指标追踪**：每个会话记录 `total_queries / workflow_counts / latency_ms / replan_rounds` 等指标。

### 2. 长短期记忆系统

- **长期记忆存储**：`agent/memory/repository.py` 基于 SQLite 持久化，支持按项目和用户隔离的 `project_memory_items` 表。
- **记忆类型分类**：`agent/memory/policy.py` 自动将对话内容分类为 `episodic`（事件记忆）、`semantic`（语义知识）、`procedural`（操作习惯）三类，并按类型设置差异化 TTL。
- **语义检索记忆**：`agent/memory/service.py` 实现基于词项重叠 + 时效衰减评分的混合记忆检索，优先返回近期相关记忆条目。
- **记忆层门面**：`agent/memory/store.py` 统一暴露 `upsert / search / compact_memory` 等操作接口，`agent/memory/policy.py` 提供 `query_long_term_memory` 高层调用。

### 3. 本地 Hybrid RAG

- **混合检索**：Dense（FastEmbed）+ BM25 + RRF 融合 + 可选 flashrank rerank。
- **结构化 Chunking**：Markdown 标题感知切分，保留 `section_path / heading_level / prev/next chunk_id`。
- **邻域扩展**：自动补充相邻 Chunk，增强上下文完整性。
- **结构化证据输出**：每条证据含 `doc_uid / chunk_id / score / offset_start / offset_end / page_no`。
- 本地嵌入模型默认 `BAAI/bge-small-zh-v1.5`，自动缓存至 `./models/embeddings/`。

### 4. 工具体系（Tools）

| 工具 | 说明 |
|------|------|
| `search_document` | 在当前文档/项目范围内 RAG 检索 |
| `read_document` | 按字符偏移直读文档原文，可选 RAG 上下文 |
| `read_file` | 读取工作区文件（带 offset/limit） |
| `write_file` | 写入/追加工作区文件 |
| `edit_file` | 按文本片段精确替换 |
| `update_file` | 按行区间替换 |
| `write_todo` | 创建/更新结构化 todo（支持 assignee/dependencies） |
| `edit_todo` | 按 id 编辑 todo（状态/依赖/备注） |
| `bash` | 在工作区内执行受限 bash 命令 |
| `ask_human` | 生成人工确认请求（前端聊天区+侧边栏可见） |
| `search_papers` | 调用 Semantic Scholar API 检索学术论文 |
| `search_web` | DuckDuckGo + SearXNG 双引擎网络检索（可按需关闭） |
| `use_skill` | 按需调用结构化技能模板 |

- **渐进式自动加载**：与 Skills 类似，Tools 先进行元信息发现，再按需构建；`search_web` provider 在首次调用时初始化，避免会话启动阶段的无效加载。
- **前端协同可视化**：左侧固定展示 pinned todo；`ask_human` 请求会在聊天区卡片和侧边栏 `人工确认（Pinned）` 同步显示。
- **Team Todo 自动落盘**：团队执行生成的 todo 任务图会自动 upsert 到 `AGENT_TODO_FILE`（默认 `.agent/todo.json`），便于侧边栏实时查看。

### 5. 技能（Skills）

从 `agent/skills/*/SKILL.md` 动态发现与加载，当前内置：

- `summary`（论文总结）
- `critical_reading`（批判性阅读）
- `method_compare`（方法对比）
- `translation`（翻译）
- `mindmap`（思维导图，输出标准 JSON 树结构）

### 6. 项目化工作区

- 支持项目创建、归档、重命名。
- 文档按项目绑定与激活，问答时严格限定在当前项目作用域检索。
- 多项目独立维护会话历史、上下文摘要、Agent 实例。

### 7. 工程化基础

- **SQLite 自动建表与迁移**：`agent_outputs` 与 `project_memory_items` 等表按 `uuid / project_uid / session_uid / doc_uid` 多维度归档。
- **Redis + RQ 异步任务队列**：无 worker 时自动回退同步执行。
- **A2A SDK 双栈兼容**：`agent/a2a/standard.py` 使用官方 `a2a-sdk` 类型，兼容旧方法 `message/send / tasks/get / tasks/cancel` 与 v1 方法 `SendMessage / GetTask / CancelTask / SendStreamingMessage`，支持 `A2A-Version` 协商与流式事件。
- **LLM Provider 适配**：自动识别 OpenAI / 阿里云百炼（DashScope），按 provider 开启 `reasoning_effort` 或 `enable_thinking`。
- **输出净化**：自动过滤 CoT 推理过程，仅向用户呈现结构化答案。

---

## 📄 页面与导航

运行后侧边栏页面：

| 页面 | 文件 | 说明 |
|------|------|------|
| 🤖 Agent 中心（默认） | `pages/0_🤖_Agent中心.py` | 页面入口（委托 `ui/agent_center_page.py`） |
| 📁 文件中心 | `pages/1_📁_文件中心.py` | 文档上传与管理 |
| ⚙️ 设置中心 | `pages/2_⚙️_设置中心.py` | API Key、模型、RAG 参数配置 |
| 🗂️ 项目中心 | `pages/3_🗂️_项目中心.py` | 项目与文档绑定管理 |

---

## 🗂️ 项目结构

```text
.
├── main.py                        # Streamlit 导航入口
├── pages/
│   ├── 0_🤖_Agent中心.py         # 页面入口（薄封装）
│   ├── 1_📁_文件中心.py
│   ├── 2_⚙️_设置中心.py
│   └── 3_🗂️_项目中心.py
├── agent/
│   ├── domain/                    # 领域契约（Policy/Team/Turn/Trace）
│   ├── application/               # 应用用例编排（turn engine）
│   │   └── agent_center/          # Agent 中心应用逻辑（key/prompt/memory）
│   ├── adapters/                  # 外部依赖适配层（LLM/RAG/文档）
│   ├── a2a/                       # A2A 协调与协议层
│   │   ├── coordinator.py         # Planner/Researcher/Reviewer 协调器
│   │   ├── router.py              # 三档模式自动路由（react/plan_act/plan_act_replan）
│   │   ├── replan_policy.py       # RePlan 预算与复核策略
│   │   ├── state_machine.py       # 协调状态机定义与转换
│   │   └── standard.py            # A2A JSON-RPC 双栈接口（含 streaming）
│   ├── orchestration/             # Leader 中心结构化编排（主链路）
│   │   ├── orchestrator.py        # request/policy/plan/team/final 统一执行
│   │   ├── planning_service.py    # LLM 生成执行计划
│   │   ├── policy_engine.py       # 复杂度评分与策略决策
│   │   ├── team_runtime.py        # 动态角色分配与多轮双向协作
│   │   └── contracts.py           # 数据契约（OrchestratedTurn / TeamExecution）
│   ├── rag/                       # RAG 子模块（chunking/evidence/hybrid/local）
│   ├── memory/                    # 长期记忆子模块（policy/repository/service/store）
│   ├── agent_center_runner.py     # 单轮执行、轨迹与证据渲染
│   ├── paper_agent.py             # ReAct 会话构建
│   ├── turn_service.py            # 统一对话轮次执行（证据归一化/阶段标签）
│   ├── context_governance.py      # 上下文治理与自动压缩
│   ├── capabilities.py            # Tool 聚合与发现（含渐进式加载）
│   ├── tools/                     # 本地工具实现（file/todo/bash/ask_human）
│   ├── llm_provider.py            # LLM Provider 适配（OpenAI / DashScope）
│   ├── output_cleaner.py          # 输出净化（过滤 CoT 推理）
│   ├── scholarly_search.py        # Semantic Scholar 学术检索
│   ├── session_state.py           # Streamlit session_state 初始化
│   ├── settings.py                # 环境变量与默认配置（含 orchestration/policy 字段）
│   ├── metrics.py                 # 会话指标统计
│   ├── archive.py                 # 回答归档（SQLite）
│   ├── logging_utils.py           # 结构化日志上下文
│   ├── stream.py                  # LangGraph 流式输出解析
│   ├── ui_helpers.py              # 轨迹/证据/思维导图 UI 组件
│   └── skills/                    # 技能模板
│       ├── loader.py
│       ├── summary/
│       ├── critical_reading/
│       ├── method_compare/
│       ├── translation/
│       └── mindmap/
├── utils/
│   ├── utils.py                   # DB 初始化、文件处理、内容缓存
│   ├── task_queue.py              # RQ 任务队列（含同步回退）
│   ├── tasks.py                   # 后台任务定义
│   ├── page_helpers.py            # 页面通用辅助函数
│   ├── schemas.py                 # 共享 Pydantic Schema
│   └── compare_parser.py          # 方法对比结果解析
├── ui/
│   ├── agent_center_page.py      # Agent 中心页面实现
│   ├── project_workspace.py       # 项目工作区 UI 组件
│   └── theme.py                   # 主题配置
├── tests/
│   ├── unit/                      # 单元测试（30+ 测试文件）
│   ├── integration/               # 集成测试 + Live API E2E
│   └── evals/                     # 评估集
├── docs/                          # 设计文档与开发记录
├── models/
│   └── embeddings/                # 本地嵌入模型缓存
├── uploads/                       # 用户上传文件
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 🚀 快速开始

### 环境要求

- Python `>= 3.10`
- [uv](https://github.com/astral-sh/uv)（推荐包管理器）

### 本地启动

```bash
# 1. 安装依赖
uv sync --no-install-project

# 2. 启动
streamlit run main.py
```

浏览器访问 `http://localhost:8501`。

### 运行配置

在"⚙️ 设置中心"页面填写：

| 配置项 | 说明 |
|--------|------|
| API Key | LLM 服务密钥 |
| 模型名称 | 如 `qwen-plus`、`gpt-4o` 等 |
| Base URL | OpenAI Compatible 接入点（默认阿里云百炼） |

---

## ⚙️ 环境变量参考

```bash
# LLM 接入
OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# RAG
LOCAL_RAG_HYBRID_ENABLED=true        # 是否启用混合检索（默认 true）
LOCAL_RAG_TOP_K=8                     # 最终返回 Chunk 数量（默认 8）
LOCAL_RAG_RERANK_ENABLED=false        # 是否启用 rerank（默认 false）
RAG_PROJECT_MAX_CHARS=300000          # 项目级 RAG 最大字符数
RAG_PROJECT_MAX_CHUNKS=1200           # 项目级 RAG 最大 Chunk 数

# Agent
AGENT_TEMPERATURE=0.1
AGENT_ENABLE_THINKING=false           # 是否启用 Chain-of-Thought 推理模式
AGENT_REASONING_EFFORT=               # OpenAI reasoning_effort（low/medium/high）

# Orchestration 团队执行
AGENT_TEAM_MAX_MEMBERS=3              # 最大动态角色数
AGENT_TEAM_MAX_ROUNDS=2               # 最大执行轮次
AGENT_TEAM_MEMBERS_HARD_CAP=6        # 角色数硬上限
AGENT_TEAM_ROUNDS_HARD_CAP=4         # 轮次数硬上限
AGENT_PLANNER_MIN_STEPS=2             # 规划最小步数
AGENT_PLANNER_MAX_STEPS=4             # 规划最大步数

# 路由策略复杂度评分阈值（均可按需调整）
AGENT_POLICY_TEXT_LEN_MEDIUM=140
AGENT_POLICY_TEXT_LEN_HIGH=240
AGENT_POLICY_SENTENCE_THRESHOLD=3
AGENT_POLICY_COMMA_THRESHOLD=4
AGENT_POLICY_QUESTION_THRESHOLD=2
AGENT_POLICY_SCORE_PLAN=2             # 触发 Plan-Act 的最低分
AGENT_POLICY_SCORE_TEAM=4             # 触发多角色团队的最低分

# 工具开关
AGENT_DISABLE_SEARCH_WEB=false        # 禁用 search_web 工具
AGENT_SEARXNG_BASE_URLS=              # SearXNG 实例地址（逗号分隔，留空使用内置公共节点）
AGENT_FILE_TOOLS_ROOT=                # 文件类工具根目录（默认当前工作目录）
AGENT_TODO_FILE=.agent/todo.json      # pinned todo 默认文件位置

# 日志
APP_LOG_LEVEL=INFO
APP_LOG_FILE=                         # 日志输出文件路径（空则仅 stdout）

# 任务队列
RQ_WORKER_COUNT=2                     # RQ Worker 数量
```

---

## 🧪 测试

```bash
# 安装开发依赖
uv sync --extra dev --no-install-project

# 单元测试
uv run --extra dev python -m pytest tests/unit -q

# 集成测试
uv run --extra dev python -m pytest tests/integration -q

# 本地真实论文语料场景 E2E（不依赖 API，需要先准备 tests/fixtures/papers）
RUN_REAL_SCENARIO_E2E=1 uv run --extra dev python -m pytest tests/integration/test_real_scene_e2e.py -q

# Live API E2E（需配置真实 API Key）
uv run --extra dev python -m pytest tests/integration/test_live_api_e2e.py -q

# 真实论文项目 E2E（下载真实论文 + 多智能体链路）
# 需设置 RUN_LIVE_REAL_PAPERS_E2E=1 及 OPENAI_* 环境变量
uv run --extra dev python -m pytest tests/integration/test_real_papers_project_e2e.py -q
```

---

## �� Docker 部署

```bash
# 构建并启动
docker-compose up --build
```

`docker-compose.yml` 说明：
- 监听端口 `8501`
- 持久化挂载 `./database.sqlite` 和 `./uploads/`
- 支持 `DASHSCOPE_API_KEY` 和 `RQ_WORKER_COUNT` 环境变量注入

> **注意**：镜像内置 LibreOffice + Tesseract OCR（支持中英文）用于文档解析，镜像体积较大（~1GB+）。

---

## 📦 主要依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `streamlit` | `~1.54` | Web UI 框架 |
| `langchain` / `langchain-openai` | `~1.x` | LLM 编排 |
| `langgraph` | — | Agent 状态机与流式执行 |
| `fastembed` | `~0.7` | 本地嵌入模型 |
| `flashrank` | `~0.2` | 本地 rerank 模型 |
| `duckduckgo-search` | `~6.3` | 网络搜索工具 |
| `pyecharts` | `~2.0` | 思维导图可视化 |
| `redis` / `rq` | `~5.x / ~1.15` | 异步任务队列 |
| `textract` | `~1.6` | 多格式文档解析 |

---

## 🤝 贡献

欢迎提交 Issue / PR ❤️。
