<div align="center">

# 📚 PaperSage

**AI-Powered Research Reading & Writing Workbench**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-informational)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-blueviolet?logo=langchain)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54%2B-red?logo=streamlit)](https://streamlit.io/)
[![A2A](https://img.shields.io/badge/A2A-Compatible-brightgreen)](https://google.github.io/A2A/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![uv](https://img.shields.io/badge/uv-managed-6E40C9)](https://github.com/astral-sh/uv)

[简体中文](README.md) · [English](#) · [CHANGELOG](CHANGELOG.md) · [Docs](docs/)

</div>

---

<div align="center">

![PaperSage Main Interface](images/main.png)

> Built with **Streamlit + LangChain + LangGraph**.  
> A project-based paper Q&A workbench: organise documents by project, scope retrieval to the active context, auto-route agent workflows, and surface traceable evidence.

</div>

---

## ✨ Feature Overview

| Feature | Description |
|---------|-------------|
| 🔀 **Multi-mode Agent Workflows** | ReAct / Plan-Act / Plan-Act-RePlan — auto-routed by query complexity |
| 🤝 **Multi-Agent Team Collaboration** | Leader-centric dispatch, LLM-generated roles, dependency-topological execution, multi-round review-replan |
| 🔍 **Local Hybrid RAG** | Dense + BM25 + RRF + Rerank four-stage retrieval with structured, traceable evidence |
| 🧠 **Long/Short-term Memory** | Episodic / semantic / procedural memory, differentiated TTL, recency-decay retrieval |
| 🛠️ **14+ Built-in Tools** | RAG search, file I/O, academic search, web search, Todo management, human-in-the-loop confirmation |
| 📝 **Pluggable Skills** | Paper summary, critical reading, method comparison, translation, mind map — dynamically loaded from `SKILL.md` |
| 🗂️ **Project Workspaces** | Multi-project isolation, document binding, independent sessions and context |

---

## 🖼️ Screenshots

### Agent Center — Intelligent Q&A
![Agent Center](images/agent中心.png)

### File Center — Document Management
![File Center](images/文件中心.png)

### Paper Q&A — Evidence Tracing
![Evidence](images/证据链.png)

### Mind Map — Visualization
![Mind Map](images/思维导图1.png)

### Paper Summary
![Summary](images/总结.png)

### Context Governance — Visualization
![Context Governance](images/上下文可视化.png)

---

## 🏗️ Architecture

### Workflow Routing & Dispatch

```text
User Query
  │
  ├─→ Smart Router (keyword fast-path → LLM structured routing → fallback ReAct)
  │
  ├─ ReAct ──────────→ Single Agent + Tool loop
  ├─ Plan-Act ────────→ Planner generates plan → Leader executes
  └─ Plan-Act-RePlan ─→ Planner → Leader ⇄ Team (multi-role) → Reviewer → RePlan
                                                                      ↓
                                                               Quality gate loop
```

### Hybrid RAG Pipeline

```text
User Query
  │
  ├─→ Dense retrieval (FastEmbed bge-small-zh)
  ├─→ BM25 sparse retrieval
  │         │
  │         ├─→ RRF fusion ranking
  │         │         │
  │         │         ├─→ FlashRank Rerank (optional)
  │         │         │         │
  │         │         │         └─→ Neighbour chunk expansion
  │         │         │                   │
  └─────────┴─────────┴───────────────────┴─→ Structured EvidenceItem
                                                (doc_uid / chunk_id / score / page_no / offset)
```

### Memory Architecture

```text
┌────────────────────────────────────────────────┐
│               Three-tier Memory                 │
├────────────────────────────────────────────────┤
│  Short-term: LangGraph InMemorySaver (session)  │
├────────────────────────────────────────────────┤
│  Mid-term: Auto context compression             │
│  (token threshold → LLM summary + fact anchors) │
├────────────────────────────────────────────────┤
│  Long-term: SQLite persistence (per project)    │
│  ├─ episodic   TTL 30 days                      │
│  ├─ semantic   permanent                        │
│  └─ procedural TTL 90 days                      │
│  Retrieval: term overlap + recency decay score  │
│  Injection: capacity circuit-breaker + conflict │
│             resolution (evidence > memory)       │
└────────────────────────────────────────────────┘
```

---

## 📄 Pages

| Page | Description |
|------|-------------|
| 🤖 **Agent Center** (default) | Main Q&A interface — workflow visualisation, evidence panel |
| 📁 **File Center** | Document upload, format conversion, content preview |
| ⚙️ **Settings** | API key, model, RAG params, agent behaviour config |
| 🗂️ **Project Center** | Project management, document binding, workspace switching |

---

## 🚀 Quick Start

### Option 1: Install from PyPI (Recommended)

> No need to clone the repository.

**Linux / macOS**

```bash
# Install via uv tool (registers the command globally)
uv tool install paper-sage

# Launch
paper-sage
```

**Windows (PowerShell)**

```powershell
# Install
uv tool install paper-sage

# Launch (use the no-hyphen alias on Windows)
papersage
```

> ⚠️ **Avoid `uv pip install`**: it does not register the command in your global PATH. You would need to activate the virtual environment manually first.

Open `http://localhost:8501` and configure your API key in **⚙️ Settings**.

---

### Option 2: Clone & Run Locally

```bash
git clone https://github.com/0verL1nk/PaperSage.git
cd PaperSage

# Install dependencies
uv sync --no-install-project

# Start the app
streamlit run main.py
```

### Option 3: Docker

```bash
docker-compose up --build
```

---

### Requirements

- Python `>= 3.10`
- [uv](https://github.com/astral-sh/uv) (recommended)

---

## 🗂️ Project Structure

```text
.
├── main.py                     # Streamlit navigation entry
├── pages/                      # Four feature pages
├── agent/                      # 🧠 Agent core (77 modules / 12,500+ lines)
│   ├── a2a/                    #   A2A coordination & protocol layer
│   ├── orchestration/          #   Leader-centric orchestration
│   ├── rag/                    #   Hybrid RAG (chunking / retrieval / evidence / fusion)
│   ├── memory/                 #   Long-term memory (classify / retrieve / store / inject)
│   ├── skills/                 #   Pluggable skills (summary / critical_reading / ...)
│   ├── tools/                  #   Built-in tools (file / todo / bash / ask_human)
│   ├── domain/                 #   Domain contracts
│   ├── application/            #   Application use-case orchestration
│   └── adapters/               #   External dependency adapters
├── ui/                         # UI component layer
├── utils/                      # Shared utilities
├── tests/                      # 53 unit tests + integration + eval
├── docs/                       # Design docs & dev notes
├── models/embeddings/          # Local embedding model cache
├── pyproject.toml              # Package config (hatch + uv)
├── Dockerfile
└── docker-compose.yml
```

---

## ⚙️ Environment Variables

<details>
<summary>Click to expand full configuration</summary>

```bash
# LLM
OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# RAG
LOCAL_RAG_HYBRID_ENABLED=true
LOCAL_RAG_TOP_K=8
LOCAL_RAG_RERANK_ENABLED=false

# Agent behaviour
AGENT_TEMPERATURE=0.1
AGENT_ENABLE_THINKING=false
AGENT_REASONING_EFFORT=            # low / medium / high (OpenAI)

# Orchestration & team
AGENT_TEAM_MAX_MEMBERS=3
AGENT_TEAM_MAX_ROUNDS=2
AGENT_PLANNER_MIN_STEPS=2
AGENT_PLANNER_MAX_STEPS=4

# Routing thresholds
AGENT_POLICY_SCORE_PLAN=2
AGENT_POLICY_SCORE_TEAM=4

# Tools
AGENT_DISABLE_SEARCH_WEB=false
AGENT_TODO_FILE=.agent/todo.json
AGENT_HISTORY_PAGE_SIZE=40
AGENT_PROJECT_INDEX_CACHE_DIR=./.cache/project_indexes

# Logging
APP_LOG_LEVEL=INFO
```

</details>

---

## 🧪 Testing

```bash
# Install dev dependencies
uv sync --extra dev --no-install-project

# Unit tests
uv run --extra dev python -m pytest tests/unit -q

# Integration tests
uv run --extra dev python -m pytest tests/integration -q

# Live API E2E (requires real API key)
uv run --extra dev python -m pytest tests/integration/test_live_api_e2e.py -q
```

---

## 📦 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web UI framework |
| **LangChain / LangGraph** | LLM orchestration & agent state machine |
| **FastEmbed** (bge-small-zh) | Local vector embeddings |
| **FlashRank** | Local reranking |
| **rank_bm25** | Sparse retrieval |
| **a2a-sdk** | Google A2A protocol compatibility |
| **SQLite** | Memory & data persistence |
| **Redis + RQ** | Async task queue |
| **pyecharts** | Mind map visualisation |
| **Docker** | Containerised deployment |

---

## 📄 License

[MIT](LICENSE)

---

## 🤝 Contributing

Issues and PRs are welcome ❤️
