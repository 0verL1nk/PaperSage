# Literature Reading Assistant
[简体中文](README.md) | English

An AI-powered literature reading platform built with Streamlit + LangChain for research workflows: extraction, summarization, Q&A, rewriting, and mind map visualization.

## Features

- User & file management
- Literature analysis:
  - Key-text extraction
  - Paper summarization
  - Text rewriting/paraphrasing/translation
  - Agent-based paper Q&A
- Mind map visualization with pyecharts

## Agent Workflows (Q&A Page)

The main entry (`main.py`, Agent Center) supports:

1. `ReAct (Tool+Memory)`
2. `Plan-Act (A2A Coordination)`
3. `Plan-Act-RePlan (A2A Coordination)`

Notes:
- A2A mode uses Planner / Researcher / Reviewer roles.
- ACP-compatible naming is kept in code for backward compatibility.

## Project Structure

```text
.
├── main.py
├── pages/
│   ├── 1_📁_文件中心.py
│   └── 2_⚙️_设置中心.py
├── agent/
│   ├── paper_agent.py
│   ├── multi_agent_a2a.py
│   ├── capabilities.py
│   ├── local_rag.py
│   └── ...
├── utils/
│   └── ...  # generic utilities
├── tests/
│   ├── unit/
│   └── integration/
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

Configure these in the sidebar:

- `API Key`
- `Model Name`
- `OpenAI Compatible Base URL` (optional)

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
