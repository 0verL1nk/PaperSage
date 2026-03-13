import threading
from collections.abc import Callable
from typing import Any

from agent.domain.openviking_contracts import OpenVikingSearchRequest
from agent.settings import load_agent_settings

from .openviking_runtime import get_openviking_adapter

EvidenceRetriever = Callable[[str], dict[str, Any]]
_PROJECT_RETRIEVER_LOCKS: dict[str, threading.Lock] = {}
_PROJECT_RETRIEVER_LOCK_GUARD = threading.Lock()
_RAG_NAMESPACE = "rag"


def _project_lock(project_uid: str) -> threading.Lock:
    normalized = str(project_uid or "").strip() or "__default__"
    with _PROJECT_RETRIEVER_LOCK_GUARD:
        lock = _PROJECT_RETRIEVER_LOCKS.get(normalized)
        if lock is None:
            lock = threading.Lock()
            _PROJECT_RETRIEVER_LOCKS[normalized] = lock
        return lock


def create_project_evidence_retriever(
    *,
    documents: list[dict[str, str]],
    project_uid: str,
) -> EvidenceRetriever:
    adapter = get_openviking_adapter()
    settings = load_agent_settings()
    with _project_lock(project_uid):
        _ingest_project_documents(
            documents=documents,
            project_uid=project_uid,
            adapter=adapter,
        )

    def _retrieve(query: str) -> dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {"evidences": []}
        hits = adapter.search(
            OpenVikingSearchRequest(
                project_uid=project_uid,
                query=normalized_query,
                top_k=max(1, int(settings.rag_top_k)),
            )
        )
        evidences: list[dict[str, Any]] = []
        for hit in hits:
            metadata = hit.metadata
            if str(metadata.get("namespace") or "").strip() != _RAG_NAMESPACE:
                continue
            evidences.append(
                {
                    "text": hit.content,
                    "score": hit.score,
                    "chunk_id": hit.resource_id,
                    "doc_uid": str(metadata.get("doc_uid") or ""),
                    "doc_name": str(metadata.get("doc_name") or ""),
                }
            )
        return {"evidences": evidences}

    return _retrieve


def _ingest_project_documents(
    *, documents: list[dict[str, str]], project_uid: str, adapter: Any
) -> None:
    for document in documents:
        text = str(document.get("text") or "").strip()
        if not text:
            continue
        metadata: dict[str, object] = {
            "namespace": _RAG_NAMESPACE,
            "doc_uid": str(document.get("doc_uid") or ""),
            "doc_name": str(document.get("doc_name") or ""),
        }
        _ = adapter.add_resource(
            project_uid=project_uid,
            content=text,
            metadata=metadata,
        )
