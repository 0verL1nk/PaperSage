import os
from typing import Any, Callable

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .settings import load_agent_settings


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


def build_local_vector_retriever(document_text: str) -> Callable[[str], str]:
    settings = load_agent_settings()
    model_name = ensure_local_embedding_model_downloaded()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
    )
    chunks = splitter.split_text(document_text)

    embeddings = FastEmbedEmbeddings(
        model_name=model_name,
        cache_dir=settings.local_embedding_cache_dir,
    )
    vectorstore = InMemoryVectorStore.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.rag_candidate_k})
    reranker = (
        _build_local_reranker(settings.rag_rerank_model)
        if settings.rag_rerank_enabled
        else None
    )

    def search_document(query: str) -> str:
        docs = retriever.invoke(query)
        ranked_chunks = _rerank_docs(
            query=query,
            docs=docs,
            reranker=reranker,
            top_k=settings.rag_top_k,
        )
        return "\n".join(ranked_chunks)

    return search_document
