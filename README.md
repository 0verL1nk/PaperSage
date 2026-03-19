<div align="center">

# 📚 PaperSage

**面向科研阅读与写作的 AI 智能体工作台**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.0.5-informational)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-blueviolet?logo=langchain)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54%2B-red?logo=streamlit)](https://streamlit.io/)
[![A2A](https://img.shields.io/badge/A2A-Compatible-brightgreen)](https://google.github.io/A2A/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![uv](https://img.shields.io/badge/uv-managed-6E40C9)](https://github.com/astral-sh/uv)

[English](README_EN.md) · [简体中文](#) · [CHANGELOG](CHANGELOG.md) · [文档](docs/)

</div>

---

<div align="center">

![PaperSage 主界面](images/main.jpg)

> 基于 **Streamlit + LangChain + LangGraph** 构建。  
> 以"项目化论文问答工作台"为核心：按项目组织文档、限定检索范围、自动路由 Agent 工作流、输出可追溯证据。

</div>

---

## ✨ 核心能力一览

| 能力 | 说明 |
|------|------|
| 🔀 **中间件驱动编排** | `OrchestrationMiddleware` 基于对话复杂度注入计划或 Team 提示，主链路由 `turn_engine + runtime_agent` 驱动 |
| 🤝 **会话级 Team 运行时** | `TeamMiddleware + TeamRuntime` 提供 `spawn_agent / send_message / get_agent_result / list_agents / close_agent` 协作能力 |
| 🔍 **项目级 Hybrid RAG** | 作用域文档切分、Dense + BM25 + RRF、可选 FlashRank Rerank、邻域 Chunk 扩展、结构化证据回传 |
| 💾 **可持久化向量存储** | `AGENT_VECTORSTORE_BACKEND=auto` 优先使用 Chroma，本地不可用时自动回退 `InMemoryVectorStore` |
| 🧠 **上下文治理与记忆** | `SqliteSaver` 会话记忆、自动压缩摘要、项目级长期记忆（episodic / semantic / procedural） |
| 🛠️ **运行时工具集** | 文档检索/阅读、学术搜索、联网搜索、技能调用、计划/Todo、Team 工具按运行时装配 |
| 📝 **可插拔技能体系** | 内置 `summary` / `critical_reading` / `method_compare` / `translation` / `mindmap` / `agentic_search` 六类技能 |
| 🗂️ **项目化工作区** | 多项目隔离、文档绑定、独立会话、会话消息与线程 ID 持久化 |

---

## 🖼️ 功能截图

### Agent 中心 — 智能问答
![alt text](images/agent中心.png)

### Agent 中心 — Team 运行时协作
当任务需要拆解为多个子任务时，系统会先由 `OrchestrationMiddleware` 分析复杂度，再向主 Agent 注入规划或 Team 提示。当前 Team 能力不是硬编码的 DAG 编排器，而是由 `TeamMiddleware + TeamRuntime` 暴露一组会话级协作工具。

- **复杂度判断与提示注入**：对多步骤分析、调研或写作任务，middleware 会建议主 Agent 使用 `write_plan`、`write_todos` 或 Team 工具，而不是直接切换到固定流程。
- **会话隔离的子 Agent 生命周期**：通过 `spawn_agent` 创建子 Agent，使用 `send_message` 派发任务，并结合 `get_agent_result` / `list_agents` / `close_agent` 管理执行状态。
- **与检索链路和技能协同**：主 Agent 仍可组合 `search_document`、`search_papers`、`search_web` 与 `use_skill` 汇总结果并生成最终回答。

**💡 当前代码链路示例：**
1. 用户提出需要分工处理的多步骤任务。
2. `OrchestrationMiddleware` 将任务标记为复杂任务，必要时设置 `needs_team`。
3. 主 Agent 调用 `spawn_agent` 创建研究或写作子 Agent。
4. 通过 `send_message` 派发子任务，并用 `get_agent_result` 汇总结果。
5. 最终回答仍由主 Agent 整合输出，并保留 trace、证据和 Todo 状态。

![team](images/team调度.png)

### 文件中心 — 文档管理

![alt text](images/文件中心.png)

### 论文问答 — 证据追溯

![alt text](images/证据链.png)

### 思维导图 — 可视化

![alt text](images/思维导图1.png)

### 论文总结

![alt text](images/总结.png)

### 上下文治理 — 可视化

![上下文可视化](images/上下文可视化.png)


---

## 🏗️ 架构设计

### 当前分层执行链路

```mermaid
flowchart TD
    A[用户提问] --> B[pages/0_agent_center.py]
    B --> C[ui.agent_center_page]
    C --> D[ui.page_bootstrap]
    C --> E[agent.application.agent_center.facade]
    E --> F[agent.application.turn_engine]
    F --> G[agent.runtime_agent.create_runtime_agent]
    G --> H[LangChain Agent Runtime]
    H --> I[Middleware 链]
    I --> I1[Trace / Retry / Orchestration]
    I --> I2[SubAgent / Team / Todo / Plan]
    I --> I3[Tool Selector / Summarization]
    H --> J[运行时工具集]
    J --> J1[search_document / search_papers / search_web]
    J --> J2[use_skill / write_plan / read_plan]
    J --> J3[spawn_agent / send_message / list_agents]
    J --> K[RAG / Skills / SQLite / Project Store]
    I --> L[最终回答 + trace_payload + evidence_items + todos]
    K --> L
```

当前 canonical 入口是 `pages -> ui -> agent.application -> runtime_agent + middlewares`。规划提示、Team 提示、Todo、trace 与自动摘要都位于 middleware 链中。

### Hybrid RAG 检索管线

```mermaid
flowchart LR
    A[用户 Query] --> B[作用域内项目文档]
    B --> C[切分 + project index cache]
    A --> D[Dense 检索]
    A --> E[BM25 稀疏检索]
    C --> D
    C --> E
    C --> J{向量库后端}
    J -->|auto / chroma| K[(Chroma 持久化)]
    J -->|fallback| L[(InMemoryVectorStore)]
    K --> D
    L --> D
    D --> F[RRF 融合]
    E --> F
    F -->|可选| G[FlashRank Rerank]
    G --> H[邻域 Chunk 扩展]
    F --> I[EvidenceItem 结构化证据]
    H --> I
    I --> M[doc_uid / chunk_id / page_no / offset / score]
```

### 长短期记忆架构

```mermaid
flowchart TB
    A[会话消息] --> B[LangGraph SqliteSaver 短期记忆]
    A --> C{上下文超过阈值}
    C --> D[auto_compact_messages]
    D --> E[LLM 摘要 + 事实锚点]
    E --> F[(session_compact_memory)]
    G[当前用户问题] --> H[search_project_memory_items]
    H --> I[episodic<br/>TTL 30 天]
    H --> J[semantic<br/>长期保留]
    H --> K[procedural<br/>TTL 90 天]
    I --> L[词项匹配 + 时效分数]
    J --> L
    K --> L
    F --> M[build_hinted_prompt]
    L --> M
    M --> N[注入 execute_turn_core]
```

---

## 📄 页面导航

| 页面 | 说明 |
|------|------|
| 🤖 **Agent 中心**（默认） | 智能问答主界面，工作流可视化，证据展示 |
| 📁 **文件中心** | 文档上传、格式转换、内容预览 |
| ⚙️ **设置中心** | API Key、模型、RAG 参数、Agent 行为配置 |
| 🗂️ **项目中心** | 项目管理、文档绑定、工作区切换 |

---

## 🚀 快速开始

### 方式一：PyPI 安装（推荐）

> 适合直接使用，无需克隆仓库。

**Linux / macOS**

```bash
# 安装（推荐 uv tool，自动注册全局命令）
uv tool install paper-sage

# 启动
paper-sage
```

**Windows（PowerShell）**

```powershell
# 安装
uv tool install paper-sage

# 启动（Windows 下用不带连字符的命令）
papersage
```

> ⚠️ **不要用 `uv pip install`**：该方式不会将命令写入全局 PATH，需手动激活虚拟环境后才能使用。

浏览器访问 `http://localhost:8501`，在 **⚙️ 设置中心** 填写 API Key 和模型名称即可开始使用。

---

### 方式二：克隆源码本地启动

```bash
# 克隆仓库
git clone https://github.com/0verL1nk/PaperSage.git
cd PaperSage

# 安装依赖
uv sync --no-install-project

# 启动应用
streamlit run main.py
```

### 方式三：Docker 部署

```bash
docker-compose up --build
```

- `docker-compose` 模式默认启用 MinerU 解析（`DOC_PARSE_BACKEND=mineru`）。
- 直接本地 `streamlit run main.py` 不会启用 MinerU，仍使用本地解析链路（MarkItDown / PyMuPDF）。
- 若本地没有 `mineru:latest` 镜像，请先按 MinerU 官方文档构建或在 `.env` 中改 `MINERU_IMAGE`。

---

### 环境要求

- Python `>= 3.11`
- [uv](https://github.com/astral-sh/uv)（推荐包管理器）

---

## 🗂️ 项目结构

```text
.
├── main.py                     # Streamlit 导航与 CLI 入口
├── pages/                      # 薄页面入口（Agent / 文件 / 设置 / 项目）
├── ui/                         # UI 组件、页面控制与 bootstrap
│   ├── agent_center/           #   Agent 中心 controller / state / view
│   └── page_bootstrap.py       #   页面公共初始化
├── agent/                      # 🧠 Agent 核心
│   ├── application/            #   用例编排与 turn 执行
│   ├── domain/                 #   领域契约、trace、请求上下文
│   ├── adapters/               #   SQLite / LLM / 项目 / Session 适配
│   ├── middlewares/            #   编排、Team、Todo、Plan、Trace、摘要
│   ├── team/                   #   会话级 TeamRuntime
│   ├── rag/                    #   Hybrid RAG（切分/检索/证据/向量库）
│   ├── memory/                 #   压缩摘要与长期记忆
│   ├── tools/                  #   文档/搜索/技能/计划/Team 工具
│   ├── subagent/               #   文件式子 Agent 配置
│   ├── skills/                 #   内置技能模板
│   └── a2a/                    #   A2A 兼容与协议对象
├── utils/                      # 遗留兼容与通用工具
├── tests/                      # 单元 / 集成 / eval
├── docs/                       # 设计文档与开发记录
├── models/embeddings/          # 本地嵌入模型缓存
├── pyproject.toml              # 项目配置（hatch + uv）
├── Dockerfile                  # 容器构建
└── docker-compose.yml          # 容器编排
```

---

## ⚙️ 主要环境变量

<details>
<summary>点击展开完整配置</summary>

```bash
# LLM 接入
OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# RAG
LOCAL_RAG_HYBRID_ENABLED=true
LOCAL_RAG_TOP_K=8
LOCAL_RAG_RERANK_ENABLED=false
AGENT_VECTORSTORE_BACKEND=auto
AGENT_VECTORSTORE_PERSIST_DIR=./.cache/vector_db

# Agent 行为
AGENT_TEMPERATURE=0.1
AGENT_ENABLE_THINKING=false
AGENT_REASONING_EFFORT=
AGENT_POLICY_ROUTER_MODEL_NAME=
AGENT_POLICY_ROUTER_BASE_URL=
AGENT_POLICY_ROUTER_API_KEY=
AGENT_POLICY_ROUTER_TEMPERATURE=0.0

# 编排与团队
AGENT_TEAM_MAX_MEMBERS=3
AGENT_TEAM_MAX_ROUNDS=2
AGENT_PLANNER_MIN_STEPS=2
AGENT_PLANNER_MAX_STEPS=4

# 工具开关
AGENT_DISABLE_SEARCH_WEB=false
AGENT_TODO_FILE=.agent/todo.json
AGENT_HISTORY_PAGE_SIZE=40
AGENT_PROJECT_INDEX_CACHE_DIR=./.cache/project_indexes

# 日志
APP_LOG_LEVEL=INFO
```

</details>

---

## 🧭 Plan 术语约定

为避免主链路、A2A、观测统计之间语义漂移，约定如下：

| 术语 | 定义 |
|------|------|
| `plan` | planner 产出的结构化执行计划（`ExecutionPlan`） |
| `replan` | 真实计划修订事件（revised plan） |
| `policy_switch` | 路由策略切换事件，不代表计划修订 |
| `step` | 单 agent 计划中的最小可执行动作（可带 `depends_on`） |
| `review` | A2A reviewer 回路事件（不等于单 agent verifier） |
| `verifier` | 单 agent step 校验阶段（对应 `step_verify`） |

更多约束见：`docs/single-agent-plan.md`

---

## 🧩 工具加载与 Schema 暴露

为降低工具数量增长带来的上下文开销，运行时采用“工具已注册 + Schema 按级别暴露”的策略。

| 项目 | 说明 |
|------|------|
| `tool_load` 事件 | 仅输出摘要（`registered/schema_ready/schema_lazy + tools preview`），避免一次性展开全部工具详情 |
| `AGENT_TOOL_SCHEMA_LEVEL=manifest`（默认） | 仅暴露工具元信息（`name/description`），不注入参数 JSON Schema |
| `AGENT_TOOL_SCHEMA_LEVEL=compact` | 暴露轻量参数摘要（字段名 + required） |
| `AGENT_TOOL_SCHEMA_LEVEL=full` | 暴露完整 JSON Schema（调试/开发场景建议按需开启） |

示例：

```bash
# 默认推荐：最小上下文占用
AGENT_TOOL_SCHEMA_LEVEL=manifest
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

# Live API E2E（需配置真实 API Key）
uv run --extra dev python -m pytest tests/integration/test_live_api_e2e.py -q
```

---

## ✅ 质量检查（Lint / Typecheck）

```bash
# Core（阻塞门禁，建议本地与 CI 必跑）
bash scripts/quality_gate.sh core

# Full（全量扫描，当前用于持续收敛，可逐步升级为阻塞）
bash scripts/quality_gate.sh full
```

说明：
- `core`：覆盖主入口与关键模块（`main.py`、`agent/domain`、`agent/tools`、`agent/application/contracts.py`）。
- `full`：覆盖全仓 `ruff` 与全量 `agent` 的 `mypy`。
- CI 已配置分层门禁：`core` 阻塞、`full` 当前非阻塞（progressive rollout）。

---

## 📦 技术栈

| 技术 | 用途 |
|------|------|
| **Streamlit** | Web UI 框架 |
| **LangChain / LangGraph** | LLM 编排与 Agent 状态机 |
| **FastEmbed** (bge-small-zh) | 本地向量嵌入 |
| **FlashRank** | 本地 Rerank |
| **rank_bm25** | 稀疏检索 |
| **a2a-sdk** | Google A2A 协议兼容 |
| **SQLite** | 记忆与数据持久化 |
| **Redis + RQ** | 异步任务队列 |
| **pyecharts** | 思维导图可视化 |
| **Docker** | 容器化部署 |

---

## 📄 License

[MIT](LICENSE)

---

## 🤝 贡献

欢迎提交 Issue / PR ❤️
