# Literature Reading Assistant
[简体中文](README.md) | English

An AI-powered literature reading platform built with **Streamlit + LangChain + LangGraph** for research workflows.  
Core concept: a project-based paper Q&A workbench — organise documents by project, scope retrieval to the active project, auto-route agent workflows, and surface traceable evidence.

## Features

- User & file management
- Literature analysis:
  - Hybrid RAG document search
  - Paper summarization
  - Text rewriting / paraphrasing / translation
  - Agent-based paper Q&A with evidence
  - Method comparison
- Mind map visualization (pyecharts)
- Long-term & short-term memory system

## Agent Workflows

The main entry (`main.py`, Agent Center) supports three workflow modes:

| Mode | When |
|------|------|
| `ReAct (Tool+Memory)` | Simple Q&A, single-hop retrieval |
| `Plan-Act (Orchestration)` | Medium-complexity tasks, LLM-generated plan + multi-role team execution |
| `Plan-Act-RePlan (Orchestration)` | High-complexity tasks with review and replanning loop |

Key points:
- **Smart routing**: keyword fast-path → LLM routing → fallback `ReAct`. Complexity scoring driven by text length, sentence count, punctuation, etc. — all thresholds configurable via env vars.
- **Structured Orchestration** (`agent/orchestration/`): `planning_service` (LLM plan generation), `policy_engine` (complexity scoring & policy decision), `team_runtime` (dynamic role assignment & multi-round execution), `orchestrator` (unified entry point).
- **Turn service**: `turn_service.py` wraps a full single-round execution, including evidence normalisation, phase label aggregation, and method-compare parsing.

## Memory System

- **Long-term memory** (`memory_repository.py`): SQLite-backed `project_memory_items` table, isolated by user and project.
- **Memory type classification** (`memory_policy.py`): auto-classifies turns as `episodic`, `semantic`, or `procedural`, with type-specific TTL.
- **Memory retrieval** (`memory_service.py`): hybrid scoring — term-overlap + recency decay.
- **Memory facade** (`memory_store.py`): exposes `upsert / search / compact_memory` operations; `memory_policy.py` provides a `query_long_term_memory` high-level API.

## Project Structure

```text
.
├── main.py
├── pages/
│   ├── 0_🤖_Agent中心.py          # Thin page entrypoint
│   ├── 1_📁_文件中心.py
│   ├── 2_⚙️_设置中心.py
│   └── 3_🗂️_项目中心.py
├── ui/
│   ├── agent_center_page.py       # Agent Center page implementation
│   ├── project_workspace.py
│   └── theme.py
├── agent/
│   ├── domain/                    # Domain contracts (policy/team/turn/trace)
│   ├── application/               # Application use-cases (turn engine)
│   │   └── agent_center/          # Agent Center application logic
│   ├── adapters/                  # External adapters (LLM/RAG/document)
│   ├── paper_agent.py             # ReAct session builder
│   ├── multi_agent_a2a.py         # A2A multi-agent coordination
│   ├── turn_service.py            # Backward-compatible facade to turn engine
│   ├── capabilities.py            # Tools (DuckDuckGo + SearXNG)
│   ├── rag_hybrid.py              # Hybrid RAG (Dense + BM25 + RRF)
│   ├── memory_store.py            # Memory facade
│   ├── memory_service.py          # Memory retrieval
│   ├── memory_policy.py           # Memory classification & TTL
│   ├── memory_repository.py       # Memory persistence (SQLite)
│   ├── orchestration/
│   │   ├── orchestrator.py
│   │   ├── planning_service.py
│   │   ├── policy_engine.py
│   │   ├── team_runtime.py
│   │   └── contracts.py
│   └── ...
├── utils/
│   └── ...
├── tests/
│   ├── unit/                      # 30+ unit test files
│   ├── integration/               # Integration + Live API E2E
│   └── evals/
├── pyproject.toml
└── docker-compose.yml
```

## Requirements

- Python `>=3.10`
- [uv](https://github.com/astral-sh/uv) (recommended)

## Quick Start

1. Clone repo

```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies

```bash
uv sync --no-install-project
```

3. Run app

```bash
streamlit run main.py
```

4. Open browser

Visit `http://localhost:8501`

## Runtime Settings

Configure these in the sidebar (⚙️ Settings):

- `API Key`
- `Model Name`
- `OpenAI Compatible Base URL` (optional, defaults to Aliyun DashScope)

## Environment Variables

```bash
# LLM
OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# RAG
LOCAL_RAG_HYBRID_ENABLED=true
LOCAL_RAG_TOP_K=8
LOCAL_RAG_RERANK_ENABLED=false
RAG_PROJECT_MAX_CHARS=300000
RAG_PROJECT_MAX_CHUNKS=1200

# Agent
AGENT_TEMPERATURE=0.1
AGENT_ENABLE_THINKING=false
AGENT_REASONING_EFFORT=              # low / medium / high (OpenAI)

# Orchestration
AGENT_TEAM_MAX_MEMBERS=3
AGENT_TEAM_MAX_ROUNDS=2
AGENT_PLANNER_MIN_STEPS=2
AGENT_PLANNER_MAX_STEPS=4

# Complexity scoring thresholds
AGENT_POLICY_TEXT_LEN_MEDIUM=140
AGENT_POLICY_TEXT_LEN_HIGH=240
AGENT_POLICY_SCORE_PLAN=2
AGENT_POLICY_SCORE_TEAM=4

# Tools
AGENT_DISABLE_SEARCH_WEB=false
AGENT_SEARXNG_BASE_URLS=             # Comma-separated SearXNG instances (optional)

# Logging
APP_LOG_LEVEL=INFO
APP_LOG_FILE=                        # Path for log file (stdout only if empty)

# Task queue
RQ_WORKER_COUNT=2
```

## Logging (Dev Debugging)

Agent Center end-to-end logs are written to `./logs/agent_center.log` by default
(rotating file: 10MB each, keep 7 backups).

Key trace fields:

- `run`: per-query run id
- `uid`: user id
- `doc`: document uid
- `workflow`: routed mode (`react` / `plan_act` / `plan_act_replan`)
- `session`: agent session id (A2A session or ReAct thread)

## Testing & Quality

Install dev dependencies:

```bash
uv sync --extra dev --no-install-project
```

Run unit tests:

```bash
uv run --extra dev python -m pytest tests/unit -q
```

Run integration / E2E tests:

```bash
uv run --extra dev python -m pytest tests/integration -q
```

Run live API E2E (requires env vars):

```bash
RUN_LIVE_E2E=1 \
OPENAI_BASE_URL=... \
OPENAI_MODEL_NAME=... \
OPENAI_API_KEY=... \
uv run --extra dev python -m pytest tests/integration/test_live_api_e2e.py -q
```

Run coverage gate:

```bash
uv run --extra dev python -m pytest \
  --cov=utils \
  --cov=agent \
  --cov=pages \
  --cov-report=term-missing \
  --cov-fail-under=80
```

## Docker (Optional)

```bash
docker-compose up --build
```

## Contributing

Issues and PRs are welcome.

- `run`: per-query run id
- `uid`: user id
- `doc`: document uid
- `workflow`: routed mode (`react` / `plan_act` / `plan_act_replan`)
- `session`: agent session id (A2A session or ReAct thread)

## Testing & Quality

Install dev dependencies:

```bash
uv sync --extra dev --no-install-project
```

Run unit tests:

```bash
uv run --extra dev python -m pytest tests/unit -q
```

Run integration / E2E tests:

```bash
uv run --extra dev python -m pytest tests/integration -q
```

Run live API E2E (requires `.env`):

Only these keys are supported in `.env`:

```bash
RUN_LIVE_E2E=1
OPENAI_BASE_URL=...
OPENAI_MODEL_NAME=...
OPENAI_API_KEY=...
AGENT_ENABLE_THINKING=0
AGENT_REASONING_EFFORT=
```

Command:

```bash
uv run --extra dev python -m pytest tests/integration/test_live_api_e2e.py -q
```

Run coverage gate:

```bash
uv run --extra dev python -m pytest \
  --cov=utils \
  --cov=agent \
  --cov=pages \
  --cov-report=term-missing \
  --cov-fail-under=80
```

## Docker (Optional)

```bash
docker-compose up --build
```

## Screenshots

### Login
![Login](images/登录.png)

### File Center
![File Center](images/%E6%96%87%E4%BB%B6%E4%B8%AD%E5%BF%83.png)

### Extraction
![Extraction](images/%E5%8E%9F%E6%96%87%E6%8F%90%E5%8F%96.png)

### Rewriting
![Rewrite Example](images/文段优化1.png)
![Rewrite Example](images/文段优化3.png)
![Rewrite Result](images/文段优化4.png)

### Q&A
![Q&A](images/论文问答.png)
![Q&A Example](images/论文问答2.png)

### Mind Map
![Mind Map](images/思维导图.png)

## Contributing

Issues and PRs are welcome.
