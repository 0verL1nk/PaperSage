from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.vectorstores import InMemoryVectorStore

try:
    from langchain_chroma import Chroma as _LangchainChroma
except Exception:
    try:
        # fmt: off
        from langchain_community.vectorstores import Chroma as _LangchainChroma  # type: ignore[assignment]
        # fmt: on
    except Exception:
        _LangchainChroma = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_SAFE_COLLECTION_RE = re.compile(r"[^a-z0-9_-]+")
_WARNED_CHROMA_UNAVAILABLE = False


def _resolve_index_batch_size() -> int:
    try:
        raw = int(os.getenv("RAG_INDEX_BATCH_SIZE", "256"))
    except Exception:
        raw = 256
    return max(16, raw)


def _vectorstore_backend() -> str:
    raw = str(os.getenv("AGENT_VECTORSTORE_BACKEND", "auto") or "").strip().lower()
    if raw in {"auto", "chroma", "inmemory"}:
        return raw
    return "auto"


def _vectorstore_persist_dir() -> Path:
    raw = str(os.getenv("AGENT_VECTORSTORE_PERSIST_DIR", "./.cache/vector_db") or "").strip()
    if not raw:
        raw = "./.cache/vector_db"
    return Path(raw)


def _safe_collection_name(prefix: str, key: str) -> str:
    normalized_prefix = _SAFE_COLLECTION_RE.sub("_", str(prefix or "rag").lower()).strip("_")
    if not normalized_prefix:
        normalized_prefix = "rag"
    raw_key = _SAFE_COLLECTION_RE.sub("_", str(key or "").lower()).strip("_")
    if not raw_key:
        raw_key = "default"
    # Chroma collection names have length limits; keep deterministic + compact.
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]
    compact = raw_key[:32]
    return f"{normalized_prefix}_{compact}_{digest}"[:63]


def stable_vectorstore_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _add_texts_in_batches(
    *,
    vectorstore: Any,
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    if not texts:
        return
    batch_size = _resolve_index_batch_size()
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_metas = (
            metadatas[start:end]
            if isinstance(metadatas, list) and len(metadatas) >= end
            else None
        )
        vectorstore.add_texts(batch_texts, metadatas=batch_metas)


def _build_inmemory_vectorstore(
    *,
    texts: list[str],
    embedding: Any,
    metadatas: list[dict[str, Any]] | None = None,
) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=embedding)
    _add_texts_in_batches(vectorstore=store, texts=texts, metadatas=metadatas)
    return store


def _build_chroma_vectorstore(
    *,
    texts: list[str],
    embedding: Any,
    metadatas: list[dict[str, Any]] | None = None,
    collection_prefix: str,
    collection_key: str,
) -> Any:
    if _LangchainChroma is None:
        raise RuntimeError("langchain Chroma integration is unavailable")

    persist_root = _vectorstore_persist_dir()
    persist_root.mkdir(parents=True, exist_ok=True)
    collection_name = _safe_collection_name(collection_prefix, collection_key)
    store = _LangchainChroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(persist_root),
    )
    existing_count = 0
    try:
        existing_count = int(store._collection.count())
    except Exception:
        existing_count = 0
    if existing_count <= 0:
        _add_texts_in_batches(vectorstore=store, texts=texts, metadatas=metadatas)
    return store


def build_vectorstore(
    *,
    texts: list[str],
    embedding: Any,
    metadatas: list[dict[str, Any]] | None = None,
    collection_prefix: str = "rag",
    collection_key: str = "",
) -> tuple[Any, str]:
    backend = _vectorstore_backend()
    if backend == "inmemory":
        return _build_inmemory_vectorstore(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
        ), "inmemory"

    effective_key = str(collection_key or "").strip()
    if not effective_key:
        effective_key = stable_vectorstore_key(
            {
                "prefix": collection_prefix,
                "size": len(texts),
                "sample": texts[: min(len(texts), 8)],
            }
        )

    try:
        store = _build_chroma_vectorstore(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            collection_prefix=collection_prefix,
            collection_key=effective_key,
        )
        return store, "chroma"
    except Exception as exc:
        global _WARNED_CHROMA_UNAVAILABLE
        if not _WARNED_CHROMA_UNAVAILABLE:
            _WARNED_CHROMA_UNAVAILABLE = True
            logger.warning(
                "Persistent vector DB unavailable, fallback to InMemoryVectorStore: %s",
                exc,
            )
        return _build_inmemory_vectorstore(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
        ), "inmemory"
