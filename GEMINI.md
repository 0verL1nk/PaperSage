# PaperSage: AI-Powered Research Reading & Writing Workbench

PaperSage is a sophisticated, project-based AI workbench designed for researchers to manage, read, and write scientific literature. It leverages a multi-agent orchestration system, hybrid RAG (Retrieval-Augmented Generation), and a flexible skill system to provide traceable and high-quality assistance.

## Project Overview

- **Core Technologies:** Python 3.10+, Streamlit, LangChain, LangGraph, Chroma/SQLite, FastEmbed/FlashRank.
- **Architecture:** 
    - **Orchestration:** Multi-mode workflows (ReAct, Plan-Act, Plan-Act-RePlan) managed by a central orchestrator.
    - **Hybrid RAG:** Four-stage retrieval pipeline (Dense + BM25 + RRF + Rerank) with structured evidence.
    - **Memory:** Three-tier memory system (Short-term session memory, Mid-term context compression, Long-term SQLite persistence).
    - **Skill System:** Pluggable skills dynamically loaded from `SKILL.md` files in `agent/skills/`.
    - **Project-Centric:** Documents and sessions are organized into projects for isolated context and management.

## Project Structure

- `main.py`: Entry point for the Streamlit application and CLI.
- `agent/`: Core intelligent logic.
    - `a2a/`: Coordination and protocol layer.
    - `orchestration/`: Multi-agent planning and execution.
    - `rag/`: Hybrid retrieval and chunking.
    - `memory/`: Long-term memory management.
    - `skills/`: Dynamically loaded specialized capabilities.
    - `tools/`: Built-in tools for agents (web search, file I/O, etc.).
- `ui/`: Shared Streamlit components and themes.
- `pages/`: Streamlit page definitions (Agent Center, File Center, Settings, Project Center).
- `utils/`: Common utilities and shared schemas.
- `tests/`: Comprehensive test suite (unit, integration, and evaluations).

## Building and Running

### Prerequisites
- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- Docker (optional, for containerized deployment)

### Setup
```bash
# Install dependencies using uv
uv sync --no-install-project
```

### Running the Application
```bash
# Using Streamlit directly
streamlit run main.py

# Or via the CLI entry point defined in pyproject.toml
paper-sage
```

### Testing
```bash
# Run unit tests
uv run --extra dev python -m pytest tests/unit -q

# Run integration tests
uv run --extra dev python -m pytest tests/integration -q
```

### Quality Gate (Linting & Typing)
```bash
# Run core quality checks (ruff and mypy)
bash scripts/quality_gate.sh core
```

## Development Conventions

- **Tool Schemas:** Uses a tiered approach for tool schema disclosure (`manifest`, `compact`, `full`) to manage context window efficiency.
- **Skill Addition:** To add a new skill, create a directory in `agent/skills/` with a `SKILL.md` file following the existing templates.
- **Configuration:** Managed via environment variables (see `.env.example` or README for details).
- **Typing:** Strict typing is encouraged in the `agent/` core; use `scripts/quality_gate.sh core` to verify.
- **Testing:** New features should include corresponding unit or integration tests in the `tests/` directory.

## Key Files
- `pyproject.toml`: Defines project metadata, dependencies, and build scripts.
- `agent/orchestration/orchestrator.py`: The heart of the agent dispatch and workflow management.
- `agent/rag/hybrid.py`: Implementation of the multi-stage hybrid RAG pipeline.
- `agent/skills/loader.py`: Responsible for dynamic loading and discovery of skills.
- `pages/0_agent_center.py`: Main interactive Q&A page logic.
