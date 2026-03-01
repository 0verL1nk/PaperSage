# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-26
**Commit:** c890ab1
**Branch:** main

## OVERVIEW

AI-powered literature reading assistant (文献阅读助手) built with Streamlit. Supports paper summarization, Q&A, text rewriting, and mind map visualization.

## STRUCTURE
```
./
├── 文件中心.py           # Main Streamlit entry point
├── pages/               # Streamlit multi-page app (5 pages)
│   ├── 1_🤓_原文提取.py
│   ├── 2_😶‍🌫️_论文总结.py
│   ├── 4_🤖_论文问答.py
│   ├── 5_✒️_文段改写.py
│   └── 6_🤯_思维导图.py
├── utils/               # Shared utilities
├── src/llm_app/         # FastAPI backend (UNUSED - not packaged)
├── tests/               # Empty test directories
├── pyproject.toml      + hatch # uvling config
├── Dockerfile          # Docker build
└── docker-compose.yml  # Container orchestration
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Main app | `文件中心.py` | Auth, file upload, sidebar config |
| Page logic | `pages/*.py` | Each feature as separate page |
| Utilities | `utils/utils.py` | Language detection, LLM calls |
| Config | `pyproject.toml` | Dependencies, Python 3.9+ |
| Docker | `Dockerfile`, `docker-compose.yml` | Container deployment |

## CONVENTIONS (DEVIATIONS)

- **uv package manager**: Use `uv sync --no-install-project` instead of pip
- **Chinese filenames**: `文件中心.py`, `pages/1_🤓_原文提取.py` - may cause encoding issues
- **No page 3**: Pages numbered 1,2,4,5,6 (3 intentionally skipped)
- **Unused backend**: `src/llm_app/` exists but NOT packaged in pyproject.toml

## ANTI-PATTERNS (THIS PROJECT)

- **No tests**: `tests/` directories empty, no pytest/config
- **No CI/CD**: No GitHub Actions, no Makefile (uses `start.sh`)
- **Dual databases**: `database.sqlite` at both root AND `src/`
- **Redis in container**: Runs inside app container (violates one-process-per-container)
- **Heavy Docker image**: Includes LibreOffice (~700MB) for textract PDF processing
- **No multi-stage Dockerfile**: All build artifacts in single image

## UNIQUE STYLES

- Streamlit with emoji page names (🤓, 😶‍🌫️, 🤖, ✒️, 🤯)
- LangChain integration for LLM workflows
- pyecharts for mind map visualization
- Redis + RQ for background task queuing

## COMMANDS

```bash
# Development
uv sync --no-install-project
streamlit run 文件中心.py

# Docker
docker-compose up --build

# Install dependencies
uv add <package>
uv remove <package>
```

## NOTES

- Configure API keys in Streamlit sidebar at runtime
- Two SQLite databases exist (root + src/) - root is active
- FastAPI backend (`src/llm_app/`) appears abandoned - not integrated with Streamlit app
