# Agent Refactor Phase Plan

## Implemented in this iteration

- Upgraded the core stack to LangChain 1.x (`langchain`, `langchain-core`, `langchain-community`, `langchain-openai`).
- Replaced legacy `ChatTongyi` call paths with a unified OpenAI-Compatible wrapper in `utils/llm_provider.py`.
- Introduced centralized runtime config in `utils/agent_settings.py`.
- Rebuilt the QA page as a single agent entry in `pages/4_🤖_论文问答.py` using `langchain.agents.create_agent`.
- Added agentic capabilities in `utils/agent_capabilities.py`:
  - local document retrieval tool,
  - web search tool,
  - skill invocation tool.
- Migrated RAG embeddings to local CPU-first FastEmbed (ONNX) in `utils/local_rag.py`.
- Enabled first-run automatic model download via `ensure_local_embedding_model_downloaded()`.

## Architectural direction

- Keep non-RAG generation model-agnostic through OpenAI-Compatible endpoint abstraction.
- Keep RAG embedding local-only for privacy/cost control and predictable latency.
- Expose capabilities as explicit tools to support agentic planning and future policy controls.

## Next phases

1. Split `utils/utils.py` into smaller domain modules to reduce coupling and static-analysis noise.
2. Add deterministic integration tests for:
   - local model auto-download behavior,
   - document retrieval tool quality,
   - skill routing behavior.
3. Introduce tool policy and audit logging for skill/tool execution.
4. Add persistent local vector index to avoid rebuilding embeddings every session.
