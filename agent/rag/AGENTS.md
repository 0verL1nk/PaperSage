# agent/rag/ AGENTS.md

**Generated:** 2026-03-22

## OVERVIEW
Hybrid RAG pipeline: Dense (vector) + Sparse (BM25) + RRF fusion + optional FlashRank rerank + evidence extraction.

## STRUCTURE
```
agent/rag/
├── __init__.py
├── chunking.py           # Document chunking strategies
├── evidence.py           # EvidenceItem dataclass + extraction
├── hybrid.py             # Main hybrid retrieval orchestrator (~800 lines)
├── local.py              # Local document indexing
└── vector_store.py       # Vector DB abstraction (Chroma / InMemory)
```

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `HybridRetriever` | class | `hybrid.py` | Dense+sparse+RRF fusion |
| `EvidenceItem` | dataclass | `evidence.py` | Structured evidence (chunk_id, page_no, offset, score) |
| `SemanticAwareSplitter` | class | `chunking.py` | Markdown-aware + length-aware chunking |
| `LocalVectorStore` | class | `vector_store.py` | Chroma or InMemory backend |

## HYBRID RETRIEVAL FLOW

```
Query → Dense (vector) ─┐
         Sparse (BM25) ─┴─→ RRF Fusion → [FlashRank Rerank] → EvidenceItem
```

Key: RRF (Reciprocal Rank Fusion) merges dense + sparse rankings without training.

## CONVENTIONS

1. **Chunk strategy**: Scope by project_doc_uid. Chunk size/overlap configurable per project.
2. **Evidence extraction**: Always returns `EvidenceItem` with doc_uid/chunk_id/page_no/offset/score — never raw text alone.
3. **Vector store backend**: `AGENT_VECTORSTORE_BACKEND=auto` (Chroma first, InMemory fallback). Never hardcode backend.
4. **BM25**: Uses `rank_bm25` for sparse retrieval. Tokenizer must match chunking tokenizer.
5. **Neighborhood expansion**: After retrieval, expand context by including adjacent chunks.
6. **No RAG for short contexts**: Hybrid retrieval only when session context exceeds threshold.

## ANTI-PATTERNS

1. Calling vector DB directly instead of through `HybridRetriever` — bypasses sparse + fusion.
2. Returning raw text instead of `EvidenceItem` — breaks evidence traceability.
3. Hardcoding chunk size (magic numbers) — use settings (`rag_chunk_size`) or `SemanticAwareSplitter` config.
4. Bypassing project scope — always filter by `project_uid`/`doc_uid`.

## TYPING

- All functions must have type annotations.
- `HybridRetriever.search()` returns `RetrievalResult | list[str]` (not `list[EvidenceItem]`).
- Evidence is extracted via `_retrieval_result_to_evidence_payload()` → `EvidencePayload`.
- `EvidenceItem` is in `evidence.py` — do not duplicate or redefine.

## TESTING
```bash
uv run --extra dev python -m pytest tests/unit/test_rag_hybrid.py tests/unit/test_rag_chunking.py tests/unit/test_rag_vector_store.py -q
```
RAG-specific tests live in `tests/unit/`: `test_rag_hybrid.py`, `test_rag_chunking.py`, `test_rag_vector_store.py`.
