# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/0verL1nk/LLM_App_Final/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/0verL1nk/LLM_App_Final/releases/tag/v0.1.0
