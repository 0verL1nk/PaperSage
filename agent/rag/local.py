import hashlib
import os
from typing import Any, Callable

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..settings import load_agent_settings
from .evidence import EvidenceItem, EvidencePayload
from .vector_store import build_vectorstore, stable_vectorstore_key


def _clamp_relevance_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _min_relevance_score(settings: Any) -> float:
    raw = getattr(settings, "rag_min_relevance_score", None)
    if raw is None:
        raw = os.getenv("LOCAL_RAG_MIN_RELEVANCE_SCORE", "0.2")
    return _clamp_relevance_score(raw)


def ensure_local_embedding_model_downloaded() -> str:
    settings = load_agent_settings()
    os.makedirs(settings.local_embedding_cache_dir, exist_ok=True)
    try:
        embeddings = FastEmbedEmbeddings(
            model_name=settings.local_embedding_model,
            cache_dir=settings.local_embedding_cache_dir,
        )
        embeddings.embed_query("warmup")
        return settings.local_embedding_model
    except Exception:
        fallback_embeddings = FastEmbedEmbeddings(
            model_name=settings.local_embedding_fallback_model,
            cache_dir=settings.local_embedding_cache_dir,
        )
        fallback_embeddings.embed_query("warmup")
        return settings.local_embedding_fallback_model


def _build_local_reranker(model_name: str) -> Any | None:
    try:
        from flashrank import Ranker

        return Ranker(model_name=model_name)
    except Exception:
        return None


def _rerank_docs(
    query: str,
    docs: list[Any],
    reranker: Any | None,
    top_k: int,
) -> list[str]:
    if not docs:
        return []

    if reranker is None:
        return [doc.page_content for doc in docs[:top_k]]

    try:
        from flashrank import RerankRequest

        passages = [{"id": str(index), "text": doc.page_content} for index, doc in enumerate(docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = reranker.rerank(rerank_request)
        selected: list[str] = []
        for item in reranked[:top_k]:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str):
                selected.append(text)
        if selected:
            return selected
    except Exception:
        pass

    return [doc.page_content for doc in docs[:top_k]]


def _build_evidence_payload(
    doc_uid: str,
    ranked_chunks: list[str],
    source_docs: list[Any],
    source_scores: list[float] | None = None,
    doc_name: str = "",
    project_uid: str = "",
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidences: list[EvidenceItem] = []
    used_indices: set[int] = set()

    for rank, text in enumerate(ranked_chunks):
        matched_idx: int | None = None
        for index, doc in enumerate(source_docs):
            if index in used_indices:
                continue
            if doc.page_content == text:
                matched_idx = index
                used_indices.add(index)
                break

        metadata: dict[str, Any] = {}
        if matched_idx is not None and isinstance(source_docs[matched_idx].metadata, dict):
            metadata = source_docs[matched_idx].metadata

        metadata_doc_uid = metadata.get("doc_uid")
        metadata_doc_name = metadata.get("doc_name")
        metadata_project_uid = metadata.get("project_uid")

        chunk_index = metadata.get("chunk_index")
        chunk_id = (
            f"chunk_{chunk_index}"
            if isinstance(chunk_index, int)
            else f"chunk_{matched_idx if matched_idx is not None else rank}"
        )
        offset_start = metadata.get("start_index")
        if not isinstance(offset_start, int):
            offset_start = None

        evidence = EvidenceItem(
            project_uid=(
                metadata_project_uid
                if isinstance(metadata_project_uid, str)
                else project_uid
            ),
            doc_uid=metadata_doc_uid if isinstance(metadata_doc_uid, str) else doc_uid,
            doc_name=metadata_doc_name if isinstance(metadata_doc_name, str) else doc_name,
            chunk_id=chunk_id,
            text=text,
            score=(
                _clamp_relevance_score(source_scores[matched_idx])
                if (
                    isinstance(source_scores, list)
                    and matched_idx is not None
                    and matched_idx < len(source_scores)
                )
                else float(1.0 / (rank + 1))
            ),
            page_no=metadata.get("page_no") if isinstance(metadata.get("page_no"), int) else None,
            offset_start=offset_start,
            offset_end=(offset_start + len(text)) if isinstance(offset_start, int) else None,
        )
        evidences.append(evidence)

    payload = EvidencePayload(evidences=evidences, trace=trace or {})
    return payload.model_dump()


def build_local_evidence_retriever(
    document_text: str,
    doc_uid: str = "",
    doc_name: str = "",
    project_uid: str = "",
) -> Callable[[str], dict[str, Any]]:
    settings = load_agent_settings()
    model_name = ensure_local_embedding_model_downloaded()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
        add_start_index=True,
    )
    documents = splitter.create_documents([document_text])
    chunks = [doc.page_content for doc in documents]
    metadatas: list[dict[str, Any]] = []
    for index, doc in enumerate(documents):
        metadata = dict(doc.metadata) if isinstance(doc.metadata, dict) else {}
        metadata["chunk_index"] = index
        metadata["doc_uid"] = doc_uid
        metadata["doc_name"] = doc_name
        metadata["project_uid"] = project_uid
        metadatas.append(metadata)

    embeddings = FastEmbedEmbeddings(
        model_name=model_name,
        cache_dir=settings.local_embedding_cache_dir,
    )
    key_payload = {
        "mode": "local_doc_dense",
        "project_uid": str(project_uid or ""),
        "doc_uid": str(doc_uid or ""),
        "doc_name": str(doc_name or ""),
        "model_name": model_name,
        "chunk_size": int(settings.rag_chunk_size),
        "chunk_overlap": int(settings.rag_chunk_overlap),
        "text_sha1": hashlib.sha1(document_text.encode("utf-8")).hexdigest(),
    }
    collection_key = stable_vectorstore_key(key_payload)
    vectorstore, vector_backend = build_vectorstore(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        collection_prefix="local_doc",
        collection_key=collection_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.rag_dense_candidate_k})
    reranker = (
        _build_local_reranker(settings.rag_rerank_model)
        if settings.rag_rerank_enabled
        else None
    )
    min_relevance_score = _min_relevance_score(settings)

    def search_document_evidence(query: str) -> dict[str, Any]:
        scored_search_failed = False
        raw_candidate_count = 0
        docs: list[Any] = []
        scores: list[float] = []
        try:
            scored_docs = vectorstore.similarity_search_with_relevance_scores(
                query,
                k=settings.rag_dense_candidate_k,
            )
            if isinstance(scored_docs, list):
                raw_candidate_count = len(scored_docs)
                for item in scored_docs:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    doc = item[0]
                    score = _clamp_relevance_score(item[1])
                    if score < min_relevance_score:
                        continue
                    docs.append(doc)
                    scores.append(score)
        except Exception:
            scored_search_failed = True

        if scored_search_failed:
            docs = retriever.invoke(query)
            scores = [0.0] * len(docs)
            raw_candidate_count = len(docs)

        ranked_chunks = _rerank_docs(
            query=query,
            docs=docs,
            reranker=reranker,
            top_k=settings.rag_top_k,
        )
        trace = {
            "mode": "dense",
            "candidate_count": len(docs),
            "candidate_count_raw": raw_candidate_count,
            "min_relevance_score": min_relevance_score,
            "scored_search_failed": scored_search_failed,
            "top_k": settings.rag_top_k,
            "vector_backend": vector_backend,
            "vector_key": collection_key[:16],
        }
        return _build_evidence_payload(
            doc_uid=doc_uid,
            ranked_chunks=ranked_chunks,
            source_docs=docs,
            source_scores=scores,
            doc_name=doc_name,
            project_uid=project_uid,
            trace=trace,
        )

    return search_document_evidence


def build_local_vector_retriever(document_text: str) -> Callable[[str], str]:
    evidence_retriever = build_local_evidence_retriever(document_text=document_text)

    def search_document(query: str) -> str:
        payload = evidence_retriever(query)
        evidences = payload.get("evidences")
        if not isinstance(evidences, list):
            return ""
        chunks: list[str] = []
        for item in evidences:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks)

    return search_document
