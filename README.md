<div align="center">

# 📚 PaperSage

**面向科研阅读与写作的 AI 智能体工作台**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-informational)](CHANGELOG.md)
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

![PaperSage 主界面](images/main.png)

> 基于 **Streamlit + LangChain + LangGraph** 构建。  
> 以"项目化论文问答工作台"为核心：按项目组织文档、限定检索范围、自动路由 Agent 工作流、输出可追溯证据。

</div>

---

## ✨ 核心能力一览

| 能力 | 说明 |
|------|------|
| 🔀 **多模式 Agent 工作流** | ReAct / Plan-Act / Plan-Act-RePlan 三级工作流，智能路由自动选择 |
| 🤝 **Multi-Agent 团队协作** | Leader 中心调度，LLM 动态生成角色，依赖拓扑派发，多轮 review-replan |
| 🔍 **本地 Hybrid RAG** | Dense + BM25 + RRF + Rerank 四阶检索，结构化证据可追溯至原文 |
| 🧠 **长短期记忆系统** | episodic / semantic / procedural 三类记忆，差异化 TTL，时效衰减检索 |
| 🛠️ **14+ 内置工具** | RAG 检索、文件读写、学术搜索、网络检索、Todo 管理、人工确认等 |
| 📝 **可插拔技能体系** | 论文总结、批判性阅读、方法对比、翻译、思维导图，从 SKILL.md 动态加载 |
| 🗂️ **项目化工作区** | 多项目隔离、文档绑定、独立会话与上下文 |

---

## 🖼️ 功能截图

### Agent 中心 — 智能问答
![alt text](images/agent中心.png)

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

### 工作流路由与调度

```text
用户提问
  │
  ├─→ 智能路由（关键词快速路由 → LLM 结构化路由 → 兜底 ReAct）
  │
  ├─ ReAct 模式 ──────→ 单 Agent + Tool 直接回答
  ├─ Plan-Act 模式 ───→ Planner 生成计划 → Leader 执行
  └─ Plan-Act-RePlan ─→ Planner → Leader ⇄ Team（多角色） → Reviewer → RePlan
                                                                    ↓
                                                              质量门控循环
```

### Hybrid RAG 检索管线

```text
用户 Query
  │
  ├─→ Dense 检索（FastEmbed bge-small-zh）
  ├─→ BM25 稀疏检索
  │         │
  │         ├─→ RRF 融合排序
  │         │         │
  │         │         ├─→ FlashRank Rerank（可选）
  │         │         │         │
  │         │         │         └─→ 邻域 Chunk 扩展
  │         │         │                   │
  └─────────┴─────────┴───────────────────┴─→ 结构化 EvidenceItem
                                                (doc_uid / chunk_id / score / page_no / offset)
```

### 长短期记忆架构

```text
┌─────────────────────────────────────────────┐
│                 记忆三层架构                    │
├─────────────────────────────────────────────┤
│  短期：LangGraph InMemorySaver（当前会话）     │
├─────────────────────────────────────────────┤
│  中期：上下文自动压缩                          │
│  （超 Token 阈值 → LLM 摘要 + 事实锚点提取）    │
├─────────────────────────────────────────────┤
│  长期：SQLite 持久化（按项目/用户隔离）          │
│  ├─ episodic（事件）  TTL 30 天               │
│  ├─ semantic（知识）  永久保留                  │
│  └─ procedural（偏好）TTL 90 天               │
│  检索：词项匹配 + 时效衰减评分                   │
│  注入：容量熔断 + 冲突消解（证据优先于记忆）       │
└─────────────────────────────────────────────┘
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

### 环境要求

- Python `>= 3.10`
- [uv](https://github.com/astral-sh/uv)（推荐包管理器）

### 本地启动

```bash
# 1. 安装依赖
uv sync --no-install-project

# 2. 启动应用
streamlit run main.py
```

浏览器访问 `http://localhost:8501`，在"⚙️ 设置中心"填写 API Key 和模型配置即可使用。

### Docker 部署

```bash
docker-compose up --build
```

---

## 🗂️ 项目结构

```text
.
├── main.py                     # Streamlit 导航入口
├── pages/                      # 四个功能页面
├── agent/                      # 🧠 Agent 核心（77 个模块 / 12,500+ 行）
│   ├── a2a/                    #   A2A 协调与协议层（状态机/路由/RePlan）
│   ├── orchestration/          #   Leader 中心编排（策略引擎/规划/团队执行）
│   ├── rag/                    #   Hybrid RAG（切分/检索/证据/融合）
│   ├── memory/                 #   长期记忆（分类/检索/存储/注入）
│   ├── skills/                 #   可插拔技能（summary/critical_reading/...）
│   ├── tools/                  #   内置工具（文件/todo/bash/ask_human）
│   ├── domain/                 #   领域契约
│   ├── application/            #   应用用例编排
│   └── adapters/               #   外部依赖适配层
├── ui/                         # UI 组件层
├── utils/                      # 共享工具函数
├── tests/                      # 单元测试 53 个 + 集成测试 + Eval
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

# Agent 行为
AGENT_TEMPERATURE=0.1
AGENT_ENABLE_THINKING=false
AGENT_REASONING_EFFORT=

# 编排与团队
AGENT_TEAM_MAX_MEMBERS=3
AGENT_TEAM_MAX_ROUNDS=2
AGENT_PLANNER_MIN_STEPS=2
AGENT_PLANNER_MAX_STEPS=4

# 路由策略阈值
AGENT_POLICY_SCORE_PLAN=2
AGENT_POLICY_SCORE_TEAM=4

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
