# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Tool Search Mechanism:** Replaced the hardcoded `activate_tool` mechanism with a dynamic `search_tools` capability based on the "Just-in-Time Retrieval" design.
- **Hybrid Tool Registry:** Introduced a new `ToolRegistry` (`agent/tools/registry.py`) that utilizes a 3-way hybrid retrieval engine (Regex intersection, BM25 sparse search, and FastEmbed dense vector search) to discover relevant tools and skills dynamically.
- **Skill Ecosystem Integration:** Updated `SkillLoader` to parse and index `keywords` from `SKILL.md` frontmatter, making dynamically loaded skills discoverable through the central tool search via the `use_skill` proxy.

### Changed

- Progressive Tool Disclosure Middleware now extracts discovered tools from the `search_tools` JSON response to un-defer their schema definitions instead of relying on explicit activation names.
- Added a shared runtime agent builder so paper agent, team member agent, and A2A agents use one assembly path for tools, middleware, and checkpointer setup.
- Team member agents now load the shared tool system with `start_plan` / `start_team` removed at load time, preventing nested mode spawning inside team execution.

## [1.0.3] - 2026-03-07

## [1.0.2] - 2026-03-07

## [1.0.1] - 2026-03-07

## [1.0.0] - 2026-03-07

### Added

- Async policy interception loop with lightweight router model, periodic context refresh, and in-loop mode switching.
- User-level policy router configuration in Settings Center (model / base URL / API key).
- User-level runtime tuning in Settings Center for async policy and memory controls:
  `AGENT_POLICY_ASYNC_*`, `RAG_INDEX_BATCH_SIZE`, `AGENT_DOCUMENT_TEXT_CACHE_MAX_CHARS`,
  `LOCAL_RAG_PROJECT_MAX_CHARS`, `LOCAL_RAG_PROJECT_MAX_CHUNKS`.
- `.env.example` with router, async-policy, RAG, memory, and tooling configuration examples.
- New unit tests covering logging configuration, user-setting migrations, tool-load tracing,
  runtime cache pruning, and provider thinking flags.

### Changed

- Project-level RAG vector index build path now uses batched insertion to reduce peak memory usage.
- Agent Center now applies per-user runtime tuning overrides at startup.
- High-frequency async interceptor decision logs are now `DEBUG` (keeps `INFO` focused on main-path decisions).
- Repository metadata and release links switched to `PaperSage`.

### Fixed

- Concurrent project retriever build race by introducing per-project build lock.
- OOM-prone document text cache growth by pruning with total-char budget.
- Thinking-related provider tests after settings schema expansion.

## [0.1.0] - 2026-03-06

### Added

- Multi-mode Agent workflow with automatic routing (ReAct / Plan-Act / Plan-Act-RePlan)
- Leader-centric Multi-Agent team orchestration with dependency-based task dispatch
- Local Hybrid RAG pipeline (Dense + BM25 + RRF + FlashRank Rerank)
- Structured evidence output with document-level traceability (chunk_id / page_no / offset)
- Long-term memory system with episodic / semantic / procedural classification and TTL
- Context governance with automatic compression and fact-anchor extraction
- 14+ built-in tools (search_document, read_file, write_file, bash, search_papers, search_web, etc.)
- 6 pluggable skills loaded from SKILL.md (summary, critical_reading, method_compare, translation, mindmap, agentic_search)
- Project-based workspace with document binding and scoped retrieval
- A2A SDK dual-stack compatibility (v0.3 + v1, streaming support)
- LLM provider adapter (OpenAI / DashScope with reasoning_effort / enable_thinking)
- Output sanitizer to filter CoT reasoning from user-facing answers
- Session metrics tracking (queries, latency, workflow counts, replan rounds)
- SQLite auto-migration for agent_outputs and memory tables
- Redis + RQ async task queue with synchronous fallback
- Docker deployment with docker-compose
- 53 unit tests + 6 integration tests + eval baselines
- CLI entry point: `paper-sage`

[Unreleased]: https://github.com/0verL1nk/PaperSage/compare/v1.0.3...HEAD
[1.0.0]: https://github.com/0verL1nk/PaperSage/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/0verL1nk/PaperSage/releases/tag/v0.1.0
[1.0.1]: https://github.com/0verL1nk/PaperSage/compare/v1.0.0...v1.0.1
[1.0.2]: https://github.com/0verL1nk/PaperSage/compare/v1.0.1...v1.0.2
[1.0.3]: https://github.com/0verL1nk/PaperSage/compare/v1.0.2...v1.0.3
