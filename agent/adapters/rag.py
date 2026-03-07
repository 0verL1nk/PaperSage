import threading
from collections.abc import Callable
from typing import Any

from ..rag.hybrid import (
    build_local_evidence_retriever_with_settings,
    build_project_evidence_retriever_with_settings,
)

EvidenceRetriever = Callable[[str], dict[str, Any]]
_PROJECT_RETRIEVER_LOCKS: dict[str, threading.Lock] = {}
_PROJECT_RETRIEVER_LOCK_GUARD = threading.Lock()


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
    with _project_lock(project_uid):
        if len(documents) == 1:
            only = documents[0]
            return build_local_evidence_retriever_with_settings(
                document_text=only["text"],
                doc_uid=only["doc_uid"],
                doc_name=only["doc_name"],
                project_uid=project_uid,
            )
        return build_project_evidence_retriever_with_settings(
            documents=documents,
            project_uid=project_uid,
        )
