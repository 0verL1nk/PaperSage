"""Hybrid 检索模块

实现 BM25 稀疏检索 + Dense 密集检索 + RRF 融合
支持配置化开关和失败回退
支持邻域扩展和证据来源输出
"""
import logging
import os
import hashlib
import json
import math
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .evidence import EvidenceItem, EvidencePayload
from .vector_store import build_vectorstore, stable_vectorstore_key
from ..settings import load_agent_settings

logger = logging.getLogger(__name__)
PROJECT_INDEX_SCHEMA_VERSION = 1


# BM25 依赖可用性检查
def _check_bm25_available() -> bool:
    """检查 rank_bm25 库是否可用"""
    try:
        import rank_bm25  # noqa: F401
        return True
    except ImportError:
        return False


BM25_AVAILABLE = _check_bm25_available()


@dataclass
class ChunkMetadata:
    """Chunk 元数据"""
    chunk_id: str
    section_path: str = ""
    source_page: int | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    heading_level: int = 0


@dataclass
class RetrievalChunk:
    """检索结果 Chunk"""
    content: str
    score: float
    source: str  # "dense" or "sparse"
    metadata: ChunkMetadata | None = None


@dataclass
class RetrievalResult:
    """检索结果（包含证据来源）"""
    chunks: list[str]
    sources: list[dict]  # 每个 chunk 的来源信息
    metadata: dict = field(default_factory=dict)  # 检索轨迹元数据


@dataclass
class RetrievalTrace:
    """检索轨迹"""
    dense_candidate_count: int = 0
    sparse_candidate_count: int = 0
    rrf_candidate_count: int = 0
    rerank_input_count: int = 0
    final_count: int = 0
    neighbor_expanded: bool = False
    fallback_reason: str | None = None


def _project_index_cache_root() -> Path:
    raw = str(os.getenv("AGENT_PROJECT_INDEX_CACHE_DIR", "./.cache/project_indexes") or "").strip()
    if not raw:
        raw = "./.cache/project_indexes"
    return Path(raw)


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _settings_signature_for_project_index(settings: Any) -> str:
    payload = {
        "v": PROJECT_INDEX_SCHEMA_VERSION,
        "embedding_model": str(getattr(settings, "local_embedding_model", "") or ""),
        "chunk_size": int(getattr(settings, "rag_chunk_size", 500) or 500),
        "chunk_overlap": int(getattr(settings, "rag_chunk_overlap", 80) or 80),
    }
    digest_input = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return _sha1_text(digest_input)[:16]


def _project_doc_index_path(
    *,
    project_uid: str,
    doc_uid: str,
    settings_signature: str,
    text_hash: str,
) -> Path:
    root = _project_index_cache_root()
    project_key = _sha1_text(str(project_uid or "__default__"))[:16]
    doc_key = _sha1_text(str(doc_uid or "__doc__"))[:16]
    return root / project_key / f"{doc_key}.{settings_signature}.{text_hash}.pkl"


def _normalize_embedding_vectors(values: Any) -> list[list[float]]:
    normalized: list[list[float]] = []
    if not isinstance(values, list):
        return normalized
    for row in values:
        if hasattr(row, "tolist"):
            try:
                row = row.tolist()
            except Exception:
                pass
        if not isinstance(row, list):
            continue
        vector: list[float] = []
        for item in row:
            try:
                vector.append(float(item))
            except Exception:
                vector.append(0.0)
        normalized.append(vector)
    return normalized


def _normalize_embedding_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if not isinstance(value, list):
        return []
    normalized: list[float] = []
    for item in value:
        try:
            normalized.append(float(item))
        except Exception:
            normalized.append(0.0)
    return normalized


def _build_project_doc_index_artifact(
    *,
    project_uid: str,
    doc_uid: str,
    doc_name: str,
    normalized_text: str,
    settings_signature: str,
    text_hash: str,
    splitter: RecursiveCharacterTextSplitter,
    embeddings: FastEmbedEmbeddings,
) -> dict[str, Any]:
    doc_docs = splitter.create_documents(
        [normalized_text],
        metadatas=[
            {
                "doc_uid": doc_uid,
                "doc_name": doc_name,
                "project_uid": project_uid,
            }
        ],
    )
    chunks = [doc.page_content for doc in doc_docs]
    metadatas = [
        dict(doc.metadata) if isinstance(doc.metadata, dict) else {}
        for doc in doc_docs
    ]
    embedding_vectors = _normalize_embedding_vectors(
        embeddings.embed_documents(chunks) if chunks else []
    )
    return {
        "schema_version": PROJECT_INDEX_SCHEMA_VERSION,
        "project_uid": project_uid,
        "doc_uid": doc_uid,
        "doc_name": doc_name,
        "text_hash": text_hash,
        "settings_signature": settings_signature,
        "chunks": chunks,
        "metadatas": metadatas,
        "embeddings": embedding_vectors,
    }


def _load_project_doc_index_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("schema_version", 0) or 0) != PROJECT_INDEX_SCHEMA_VERSION:
        return None
    chunks = payload.get("chunks")
    metadatas = payload.get("metadatas")
    embeddings = payload.get("embeddings")
    if not isinstance(chunks, list) or not isinstance(metadatas, list) or not isinstance(embeddings, list):
        return None
    if len(chunks) != len(metadatas) or len(chunks) != len(embeddings):
        return None
    payload["embeddings"] = _normalize_embedding_vectors(embeddings)
    if len(payload["embeddings"]) != len(chunks):
        return None
    return payload


def _save_project_doc_index_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        pickle.dump(payload, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = Path(tmp.name)
    os.replace(str(tmp_path), str(path))


def _load_or_build_project_doc_index_artifact(
    *,
    project_uid: str,
    doc_uid: str,
    doc_name: str,
    normalized_text: str,
    settings_signature: str,
    splitter: RecursiveCharacterTextSplitter,
    embeddings: FastEmbedEmbeddings,
) -> tuple[dict[str, Any], bool]:
    text_hash = _sha1_text(normalized_text)
    path = _project_doc_index_path(
        project_uid=project_uid,
        doc_uid=doc_uid,
        settings_signature=settings_signature,
        text_hash=text_hash,
    )
    cached = _load_project_doc_index_artifact(path)
    if isinstance(cached, dict):
        return cached, True
    built = _build_project_doc_index_artifact(
        project_uid=project_uid,
        doc_uid=doc_uid,
        doc_name=doc_name,
        normalized_text=normalized_text,
        settings_signature=settings_signature,
        text_hash=text_hash,
        splitter=splitter,
        embeddings=embeddings,
    )
    try:
        _save_project_doc_index_artifact(path, built)
    except Exception as exc:
        logger.warning("Failed to persist project index artifact: project=%s doc=%s err=%s", project_uid, doc_uid, exc)
    return built, False


def _cosine_topk(
    *,
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    k: int,
) -> list[tuple[int, float]]:
    if not query_vector or not candidate_vectors or k <= 0:
        return []
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    if query_norm <= 0.0:
        return []
    scored: list[tuple[int, float]] = []
    for index, vector in enumerate(candidate_vectors):
        if not vector:
            continue
        dim = min(len(query_vector), len(vector))
        if dim <= 0:
            continue
        dot = 0.0
        norm_v = 0.0
        for i in range(dim):
            qv = query_vector[i]
            vv = vector[i]
            dot += qv * vv
            norm_v += vv * vv
        if norm_v <= 0.0:
            continue
        score = dot / (query_norm * math.sqrt(norm_v))
        scored.append((index, float(score)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def _retrieval_result_to_evidence_payload(
    result: RetrievalResult,
    doc_uid: str = "",
    doc_name: str = "",
    project_uid: str = "",
) -> dict[str, Any]:
    evidences: list[EvidenceItem] = []
    for index, chunk in enumerate(result.chunks):
        source = result.sources[index] if index < len(result.sources) else {}
        source_index = source.get("index")
        offset_start = source_index if isinstance(source_index, int) and source_index >= 0 else None
        source_doc_uid = source.get("doc_uid")
        source_doc_name = source.get("doc_name")
        source_project_uid = source.get("project_uid")
        evidence = EvidenceItem(
            project_uid=(
                source_project_uid if isinstance(source_project_uid, str) else project_uid
            ),
            doc_uid=source_doc_uid if isinstance(source_doc_uid, str) else doc_uid,
            doc_name=source_doc_name if isinstance(source_doc_name, str) else doc_name,
            chunk_id=str(source.get("chunk_id") or f"chunk_{index}"),
            text=chunk,
            score=float(source.get("score", 0.0)) if isinstance(source, dict) else 0.0,
            page_no=source.get("page_no") if isinstance(source.get("page_no"), int) else None,
            offset_start=offset_start,
            offset_end=(offset_start + len(chunk)) if isinstance(offset_start, int) else None,
        )
        evidences.append(evidence)

    trace_payload: dict[str, Any] = {}
    if isinstance(result.metadata, dict):
        raw_trace = result.metadata.get("trace")
        if isinstance(raw_trace, dict):
            trace_payload = raw_trace

    payload = EvidencePayload(evidences=evidences, trace=trace_payload)
    return payload.model_dump()


def _build_bm25_retriever(
    chunks: list[str],
    k: int = 30,
):
    """构建 BM25 检索器

    Args:
        chunks: 文本块列表
        k: 返回的候选数量

    Returns:
        BM25 检索函数
    """
    if not BM25_AVAILABLE:
        return None

    try:
        from rank_bm25 import BM25Okapi
        import jieba

        # Tokenize chunks
        tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        def search(query: str, top_k: int = k) -> list[tuple[int, float]]:
            """BM25 搜索

            Returns:
                List of (chunk_index, score) tuples, sorted by score descending
            """
            tokenized_query = list(jieba.cut(query))
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [
                (idx, score)
                for idx, score in top_indices[:top_k]
                if score > 0
            ]

        return search

    except Exception as e:
        logger.warning(f"BM25 初始化失败: {e}")
        return None


def _reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k: int = 60,
    rrf_constant: float = 60.0,
) -> list[tuple[int, float]]:
    """RRF (Reciprocal Rank Fusion) 融合

    RRF score = sum(1 / (k + rank)) for each retrieval method

    Args:
        dense_results: Dense 检索结果 [(chunk_index, score), ...]
        sparse_results: Sparse 检索结果 [(chunk_index, score), ...]
        k: RRF 常数，通常设置为候选数量的默认值
        rrf_constant: RRF 公式中的常数 k_rff

    Returns:
        融合后的结果 [(chunk_index, rrf_score), ...]，按 score 降序
    """
    rrf_scores: dict[int, float] = {}

    # 处理 Dense 结果
    for rank, (idx, score) in enumerate(dense_results):
        rrf_score = rrf_constant / (k + rank + 1)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + rrf_score

    # 处理 Sparse 结果
    for rank, (idx, score) in enumerate(sparse_results):
        rrf_score = rrf_constant / (k + rank + 1)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + rrf_score

    # 按 RRF score 排序
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return sorted_results


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-Max 归一化 scores"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [
        (s - min_score) / (max_score - min_score)
        for s in scores
    ]


def _rerank_docs(
    query: str,
    docs: list[Any],
    reranker: Any | None,
    top_k: int,
) -> list[str]:
    """重排序文档（与 local_rag.py 一致）"""
    if not docs:
        return []

    if reranker is None:
        return [doc.page_content for doc in docs[:top_k]]

    try:
        from flashrank import Ranker, RerankRequest

        passages = [
            {"id": str(index), "text": doc.page_content}
            for index, doc in enumerate(docs)
        ]
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


def _build_local_reranker(model_name: str) -> Any | None:
    """构建本地重排序模型（与 local_rag.py 一致）"""
    try:
        from flashrank import Ranker
        return Ranker(model_name=model_name)
    except Exception:
        return None


class HybridRetriever:
    """混合检索器

    结合 Dense（向量）和 Sparse（BM25）检索结果，使用 RRF 融合
    支持邻域扩展和证据来源输出
    支持 Query 预处理（同义改写、子问题拆分）
    """

    def __init__(
        self,
        chunks: list[str],
        embedding_model: str,
        embedding_cache_dir: str,
        vectorstore_collection_key: str = "",
        dense_k: int = 30,
        sparse_k: int = 30,
        rrf_k: int = 40,
        rerank_enabled: bool = True,
        rerank_model: str = "ms-marco-MiniLM-L-12-v2",
        neighbor_expansion: bool = True,
        neighbor_count: int = 1,
        query_preprocess_enabled: bool = False,
        query_rewrite_enabled: bool = True,
        query_split_enabled: bool = True,
    ):
        self.chunks = chunks
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k
        self.rerank_enabled = rerank_enabled
        self.rerank_model = rerank_model
        self.neighbor_expansion = neighbor_expansion
        self.neighbor_count = neighbor_count
        self.query_preprocess_enabled = query_preprocess_enabled
        self.query_rewrite_enabled = query_rewrite_enabled
        self.query_split_enabled = query_split_enabled
        self.vectorstore_collection_key = str(vectorstore_collection_key or "").strip()

        # LLM 客户端（用于 query 预处理）
        self.llm_client: Any = None

        # 初始化 Dense 检索（向量检索）
        embeddings = FastEmbedEmbeddings(
            model_name=embedding_model,
            cache_dir=embedding_cache_dir,
        )
        chunk_metadatas = [{"chunk_index": idx} for idx in range(len(chunks))]
        self.vectorstore, self.vector_backend = build_vectorstore(
            texts=chunks,
            embedding=embeddings,
            metadatas=chunk_metadatas,
            collection_prefix="hybrid_doc",
            collection_key=self.vectorstore_collection_key,
        )

        # 初始化 Sparse 检索（BM25）
        self.bm25_search = _build_bm25_retriever(chunks, k=sparse_k)

        # 初始化重排序器
        self.reranker = _build_local_reranker(rerank_model) if rerank_enabled else None

    def _expand_neighbors(
        self,
        indices: list[int],
    ) -> list[int]:
        """邻域扩展：包含选中 chunk 的前后邻居

        Args:
            indices: 原始检索到的 chunk 索引列表

        Returns:
            扩展后的索引列表（包含邻居）
        """
        if not self.neighbor_expansion or self.neighbor_count <= 0:
            return indices

        expanded = set(indices)
        for idx in indices:
            # 添加前面的邻居
            for i in range(1, self.neighbor_count + 1):
                if idx - i >= 0:
                    expanded.add(idx - i)
            # 添加后面的邻居
            for i in range(1, self.neighbor_count + 1):
                if idx + i < len(self.chunks):
                    expanded.add(idx + i)

        return sorted(expanded)

    def search(
        self,
        query: str,
        top_k: int = 8,
        include_sources: bool = False,
    ) -> RetrievalResult | list[str]:
        """执行混合检索

        Args:
            query: 查询字符串
            top_k: 最终返回的结果数量
            include_sources: 是否包含证据来源信息

        Returns:
            如果 include_sources=True，返回 RetrievalResult
            否则返回 list[str]（兼容旧接口）
        """
        trace = RetrievalTrace()

        # 0. Query 预处理（同义改写 + 子问题拆分）
        processed_queries = [query]
        if self.query_preprocess_enabled and self.llm_client is not None:
            # 同义改写
            rewritten_query = query_rewrite(
                query,
                self.llm_client,
                rewrite_enabled=self.query_rewrite_enabled,
            )
            # 子问题拆分
            processed_queries = query_split(
                rewritten_query,
                self.llm_client,
                split_enabled=self.query_split_enabled,
            )

        # 如果有多个查询，使用多查询检索并合并结果
        if len(processed_queries) > 1:
            return self._search_multiple_queries(
                processed_queries,
                query,  # 原始查询用于 rerank
                top_k,
                include_sources,
                trace,
            )

        # 单查询检索（原有逻辑）
        return self._search_single_query(
            query,
            top_k,
            include_sources,
            trace,
        )

    def _search_single_query(
        self,
        query: str,
        top_k: int,
        include_sources: bool,
        trace: RetrievalTrace,
    ) -> RetrievalResult | list[str]:
        """执行单查询检索"""
        # 1. Dense 检索
        dense_docs = self.vectorstore.similarity_search(
            query,
            k=self.dense_k,
        )
        trace.dense_candidate_count = len(dense_docs)
        dense_results = [
            (i, 1.0 / (i + 1))
            for i in range(len(dense_docs))
        ]

        # 2. Sparse 检索（BM25）
        sparse_results: list[tuple[int, float]] = []
        if self.bm25_search is not None:
            try:
                sparse_results = self.bm25_search(query, top_k=self.sparse_k)
                trace.sparse_candidate_count = len(sparse_results)
                # 归一化 BM25 scores
                if sparse_results:
                    scores = [s for _, s in sparse_results]
                    normalized = _normalize_scores(scores)
                    sparse_results = [
                        (idx, normalized[i])
                        for i, (idx, _) in enumerate(sparse_results)
                    ]
            except Exception as e:
                logger.warning(f"BM25 检索失败: {e}")
                trace.fallback_reason = "BM25 failed"

        # 3. RRF 融合
        if sparse_results:
            fused_results = _reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=self.rrf_k,
            )
        else:
            if self.bm25_search is None:
                trace.fallback_reason = "BM25 not available"
            fused_results = dense_results

        trace.rrf_candidate_count = len(fused_results)

        # 4. 邻域扩展
        fused_indices = [idx for idx, _ in fused_results[:top_k]]
        if self.neighbor_expansion:
            fused_indices = self._expand_neighbors(fused_indices)
            trace.neighbor_expanded = True

        trace.final_count = len(fused_indices)

        # 5. 获取文档
        retrieved_docs = [dense_docs[i] for i in fused_indices if i < len(dense_docs)]

        # 6. 重排序
        trace.rerank_input_count = len(retrieved_docs)
        ranked_chunks = _rerank_docs(
            query=query,
            docs=retrieved_docs,
            reranker=self.reranker,
            top_k=top_k,
        )

        # 7. 构建返回结果
        if include_sources:
            sources = []
            for i, chunk in enumerate(ranked_chunks):
                # 找到 chunk 在原始列表中的索引
                original_idx = None
                selected_doc = None
                for idx, doc in enumerate(retrieved_docs):
                    if doc.page_content == chunk:
                        original_idx = fused_indices[idx] if idx < len(fused_indices) else idx
                        selected_doc = doc
                        break
                metadata = (
                    dict(selected_doc.metadata)
                    if selected_doc is not None and isinstance(selected_doc.metadata, dict)
                    else {}
                )
                sources.append({
                    "chunk_id": f"chunk_{original_idx}" if original_idx is not None else f"chunk_{i}",
                    "index": original_idx if original_idx is not None else i,
                    "score": 1.0 / (i + 1) if i < len(fused_results) else 0,
                    "doc_uid": metadata.get("doc_uid", ""),
                    "doc_name": metadata.get("doc_name", ""),
                    "project_uid": metadata.get("project_uid", ""),
                    "page_no": metadata.get("page_no"),
                })
            return RetrievalResult(
                chunks=ranked_chunks,
                sources=sources,
                metadata={
                    "trace": {
                        **trace.__dict__,
                        "vector_backend": self.vector_backend,
                        "vector_key": self.vectorstore_collection_key[:16],
                    }
                },
            )

        return ranked_chunks

    def _search_multiple_queries(
        self,
        queries: list[str],
        original_query: str,
        top_k: int,
        include_sources: bool,
        trace: RetrievalTrace,
    ) -> RetrievalResult | list[str]:
        """执行多查询检索并合并结果

        对每个子查询分别检索，然后使用 RRF 合并结果。
        """
        all_results: dict[int, float] = {}  # chunk_index -> accumulated score

        for sub_query in queries:
            # 对每个子查询执行单查询检索
            sub_trace = RetrievalTrace()

            # Dense 检索
            dense_docs = self.vectorstore.similarity_search(
                sub_query,
                k=self.dense_k,
            )
            sub_trace.dense_candidate_count = len(dense_docs)

            # Sparse 检索
            sparse_results: list[tuple[int, float]] = []
            if self.bm25_search is not None:
                try:
                    sparse_results = self.bm25_search(sub_query, top_k=self.sparse_k)
                    sub_trace.sparse_candidate_count = len(sparse_results)
                    if sparse_results:
                        scores = [s for _, s in sparse_results]
                        normalized = _normalize_scores(scores)
                        sparse_results = [
                            (idx, normalized[i])
                            for i, (idx, _) in enumerate(sparse_results)
                        ]
                except Exception:
                    pass

            # 累积结果
            for rank, (idx, _) in enumerate(dense_docs):
                score = 1.0 / (self.rrf_k + rank + 1)
                all_results[idx] = all_results.get(idx, 0) + score

            for rank, (idx, score) in enumerate(sparse_results):
                score = 1.0 / (self.rrf_k + rank + 1)
                all_results[idx] = all_results.get(idx, 0) + score

        # 按累积分数排序
        fused_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        trace.rrf_candidate_count = len(fused_results)

        # 邻域扩展
        fused_indices = [idx for idx, _ in fused_results[:top_k]]
        if self.neighbor_expansion:
            fused_indices = self._expand_neighbors(fused_indices)
            trace.neighbor_expanded = True

        trace.final_count = len(fused_indices)

        # 获取文档（使用原始查询的向量检索结果）
        dense_docs = self.vectorstore.similarity_search(
            original_query,
            k=max(fused_indices) + 1 if fused_indices else self.dense_k,
        )
        retrieved_docs = [dense_docs[i] for i in fused_indices if i < len(dense_docs)]

        # 重排序（使用原始查询）
        trace.rerank_input_count = len(retrieved_docs)
        ranked_chunks = _rerank_docs(
            query=original_query,
            docs=retrieved_docs,
            reranker=self.reranker,
            top_k=top_k,
        )

        # 构建返回结果
        if include_sources:
            sources = []
            for i, chunk in enumerate(ranked_chunks):
                original_idx = None
                selected_doc = None
                for idx, doc in enumerate(retrieved_docs):
                    if doc.page_content == chunk:
                        original_idx = fused_indices[idx] if idx < len(fused_indices) else idx
                        selected_doc = doc
                        break
                metadata = (
                    dict(selected_doc.metadata)
                    if selected_doc is not None and isinstance(selected_doc.metadata, dict)
                    else {}
                )
                sources.append({
                    "chunk_id": f"chunk_{original_idx}" if original_idx is not None else f"chunk_{i}",
                    "index": original_idx if original_idx is not None else i,
                    "score": 1.0 / (i + 1),
                    "doc_uid": metadata.get("doc_uid", ""),
                    "doc_name": metadata.get("doc_name", ""),
                    "project_uid": metadata.get("project_uid", ""),
                    "page_no": metadata.get("page_no"),
                })
            return RetrievalResult(
                chunks=ranked_chunks,
                sources=sources,
                metadata={
                    "trace": {
                        **trace.__dict__,
                        "vector_backend": self.vector_backend,
                        "vector_key": self.vectorstore_collection_key[:16],
                    }
                },
            )

        return ranked_chunks

    def set_llm_client(self, llm_client: Any) -> None:
        """设置 LLM 客户端（用于 query 预处理）"""
        self.llm_client = llm_client


def build_hybrid_retriever(
    document_text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    embedding_cache_dir: str = "./models/embeddings",
    dense_k: int = 30,
    sparse_k: int = 30,
    rrf_k: int = 40,
    rerank_enabled: bool = True,
    rerank_model: str = "ms-marco-MiniLM-L-12-v2",
    neighbor_expansion: bool = True,
    neighbor_count: int = 1,
    query_preprocess_enabled: bool = False,
    query_rewrite_enabled: bool = True,
    query_split_enabled: bool = True,
) -> Callable[[str], str]:
    """构建混合检索器（工厂函数）

    与 local_rag.py 的 build_local_vector_retriever 保持一致的接口

    Args:
        document_text: 文档文本
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        embedding_model: 嵌入模型名称
        embedding_cache_dir: 嵌入模型缓存目录
        dense_k: Dense 候选数量
        sparse_k: Sparse 候选数量
        rrf_k: RRF 融合常数
        rerank_enabled: 是否启用重排序
        rerank_model: 重排序模型名称
        neighbor_expansion: 是否启用邻域扩展
        neighbor_count: 邻居数量
        query_preprocess_enabled: 是否启用 Query 预处理
        query_rewrite_enabled: 是否启用 Query 同义改写
        query_split_enabled: 是否启用 Query 子问题拆分

    Returns:
        检索函数 (query: str) -> str
    """
    # 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
    )
    chunks = splitter.split_text(document_text)
    vectorstore_key = stable_vectorstore_key(
        {
            "mode": "hybrid_doc_dense",
            "text_sha1": _sha1_text(document_text),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
        }
    )

    # 创建混合检索器
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        embedding_cache_dir=embedding_cache_dir,
        vectorstore_collection_key=vectorstore_key,
        dense_k=dense_k,
        sparse_k=sparse_k,
        rrf_k=rrf_k,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        neighbor_expansion=neighbor_expansion,
        neighbor_count=neighbor_count,
        query_preprocess_enabled=query_preprocess_enabled,
        query_rewrite_enabled=query_rewrite_enabled,
        query_split_enabled=query_split_enabled,
    )

    def search_document(query: str) -> str:
        results = retriever.search(query)
        if isinstance(results, RetrievalResult):
            return "\n".join(results.chunks)
        return "\n".join(results)

    return search_document


def build_hybrid_evidence_retriever(
    document_text: str,
    doc_uid: str = "",
    doc_name: str = "",
    project_uid: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    embedding_cache_dir: str = "./models/embeddings",
    dense_k: int = 30,
    sparse_k: int = 30,
    rrf_k: int = 40,
    rerank_enabled: bool = True,
    rerank_model: str = "ms-marco-MiniLM-L-12-v2",
    neighbor_expansion: bool = True,
    neighbor_count: int = 1,
    query_preprocess_enabled: bool = False,
    query_rewrite_enabled: bool = True,
    query_split_enabled: bool = True,
    top_k: int = 8,
) -> Callable[[str], dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
    )
    chunks = splitter.split_text(document_text)
    vectorstore_key = stable_vectorstore_key(
        {
            "mode": "hybrid_doc_evidence",
            "project_uid": str(project_uid or ""),
            "doc_uid": str(doc_uid or ""),
            "doc_name": str(doc_name or ""),
            "text_sha1": _sha1_text(document_text),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
        }
    )

    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        embedding_cache_dir=embedding_cache_dir,
        vectorstore_collection_key=vectorstore_key,
        dense_k=dense_k,
        sparse_k=sparse_k,
        rrf_k=rrf_k,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        neighbor_expansion=neighbor_expansion,
        neighbor_count=neighbor_count,
        query_preprocess_enabled=query_preprocess_enabled,
        query_rewrite_enabled=query_rewrite_enabled,
        query_split_enabled=query_split_enabled,
    )

    def search_document_evidence(query: str) -> dict[str, Any]:
        result = retriever.search(
            query=query,
            top_k=top_k,
            include_sources=True,
        )
        if isinstance(result, RetrievalResult):
            return _retrieval_result_to_evidence_payload(
                result,
                doc_uid=doc_uid,
                doc_name=doc_name,
                project_uid=project_uid,
            )

        fallback_payload = EvidencePayload(
            evidences=[
                EvidenceItem(
                    project_uid=project_uid,
                    doc_uid=doc_uid,
                    doc_name=doc_name,
                    chunk_id=f"chunk_{i}",
                    text=text,
                    score=float(1.0 / (i + 1)),
                )
                for i, text in enumerate(result)
            ],
            trace={"mode": "hybrid", "fallback": "missing_sources"},
        )
        return fallback_payload.model_dump()

    return search_document_evidence


def build_local_evidence_retriever_with_settings(
    document_text: str,
    doc_uid: str = "",
    doc_name: str = "",
    project_uid: str = "",
) -> Callable[[str], dict[str, Any]]:
    settings = load_agent_settings()

    if settings.rag_hybrid_enabled:
        logger.info("启用 Hybrid 结构化证据检索")
        return build_hybrid_evidence_retriever(
            document_text=document_text,
            doc_uid=doc_uid,
            doc_name=doc_name,
            project_uid=project_uid,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            embedding_model=settings.local_embedding_model,
            embedding_cache_dir=settings.local_embedding_cache_dir,
            dense_k=settings.rag_dense_candidate_k,
            sparse_k=settings.rag_sparse_candidate_k,
            rrf_k=settings.rag_rrf_candidate_k,
            rerank_enabled=settings.rag_rerank_enabled,
            rerank_model=settings.rag_rerank_model,
            neighbor_expansion=settings.rag_neighbor_expansion,
            neighbor_count=settings.rag_neighbor_count,
            top_k=settings.rag_top_k,
        )

    from .local import build_local_evidence_retriever

    return build_local_evidence_retriever(
        document_text=document_text,
        doc_uid=doc_uid,
        doc_name=doc_name,
        project_uid=project_uid,
    )


def build_project_evidence_retriever_with_settings(
    documents: list[dict[str, str]],
    project_uid: str = "",
) -> Callable[[str], dict[str, Any]]:
    settings = load_agent_settings()
    raw_max_project_chars = int(settings.rag_project_max_chars)
    raw_max_project_chunks = int(settings.rag_project_max_chunks)
    max_project_chars = raw_max_project_chars if raw_max_project_chars > 0 else None
    max_project_chunks = raw_max_project_chunks if raw_max_project_chunks > 0 else None
    project_rerank_enabled = (
        bool(settings.rag_rerank_enabled) and bool(settings.rag_project_rerank_enabled)
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
        add_start_index=True,
    )

    chunks: list[str] = []
    metadatas: list[dict[str, Any]] = []
    chunk_embeddings: list[list[float]] = []
    chunk_counter = 0
    doc_count = 0
    remaining_chars = max_project_chars if isinstance(max_project_chars, int) else None
    truncated_by_chars = False
    truncated_by_chunks = False
    reused_doc_indexes = 0
    built_doc_indexes = 0
    settings_signature = _settings_signature_for_project_index(settings)
    embeddings = FastEmbedEmbeddings(
        model_name=settings.local_embedding_model,
        cache_dir=settings.local_embedding_cache_dir,
    )
    for item in documents:
        if (
            (isinstance(remaining_chars, int) and remaining_chars <= 0)
            or truncated_by_chunks
        ):
            break
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        doc_uid = item.get("doc_uid")
        doc_name = item.get("doc_name")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(doc_uid, str) or not doc_uid.strip():
            continue
        normalized_text = text.strip()
        if isinstance(remaining_chars, int) and len(normalized_text) > remaining_chars:
            normalized_text = normalized_text[: max(remaining_chars, 0)]
            truncated_by_chars = True
        if not normalized_text:
            continue
        if isinstance(remaining_chars, int):
            remaining_chars -= len(normalized_text)
        normalized_doc_name = doc_name if isinstance(doc_name, str) else ""
        doc_index_artifact, reused_from_cache = _load_or_build_project_doc_index_artifact(
            project_uid=project_uid,
            doc_uid=doc_uid,
            doc_name=normalized_doc_name,
            normalized_text=normalized_text,
            settings_signature=settings_signature,
            splitter=splitter,
            embeddings=embeddings,
        )
        if reused_from_cache:
            reused_doc_indexes += 1
        else:
            built_doc_indexes += 1
        doc_count += 1
        doc_chunks = doc_index_artifact.get("chunks", [])
        doc_metadatas = doc_index_artifact.get("metadatas", [])
        doc_embeddings = doc_index_artifact.get("embeddings", [])
        doc_chunk_count = min(len(doc_chunks), len(doc_metadatas), len(doc_embeddings))
        for index in range(doc_chunk_count):
            if isinstance(max_project_chunks, int) and chunk_counter >= max_project_chunks:
                truncated_by_chunks = True
                break
            chunk_text = doc_chunks[index]
            metadata = (
                dict(doc_metadatas[index]) if isinstance(doc_metadatas[index], dict) else {}
            )
            chunk_vector = doc_embeddings[index]
            if not isinstance(chunk_text, str):
                continue
            if not isinstance(chunk_vector, list) or not chunk_vector:
                continue
            metadata["chunk_index"] = chunk_counter
            metadata["chunk_id"] = f"{doc_uid}:chunk_{chunk_counter}"
            metadata["doc_uid"] = doc_uid
            metadata["doc_name"] = normalized_doc_name
            metadata["project_uid"] = project_uid
            chunks.append(chunk_text)
            metadatas.append(metadata)
            chunk_embeddings.append(chunk_vector)
            chunk_counter += 1

    if truncated_by_chars or truncated_by_chunks:
        logger.info(
            "Project retriever input truncated: project=%s docs=%s chunks=%s max_chars=%s max_chunks=%s by_chars=%s by_chunks=%s",
            project_uid,
            doc_count,
            len(chunks),
            max_project_chars if max_project_chars is not None else "unlimited",
            max_project_chunks if max_project_chunks is not None else "unlimited",
            truncated_by_chars,
            truncated_by_chunks,
        )
    if reused_doc_indexes > 0 or built_doc_indexes > 0:
        logger.info(
            "Project index cache stats: project=%s reused_docs=%s built_docs=%s",
            project_uid,
            reused_doc_indexes,
            built_doc_indexes,
        )

    if not chunks:
        def _empty(_query: str) -> dict[str, Any]:
            payload = EvidencePayload(
                evidences=[],
                trace={
                    "mode": "project_dense",
                    "project_uid": project_uid,
                    "candidate_count": 0,
                    "doc_count": 0,
                },
            )
            return payload.model_dump()

        return _empty

    dense_k = max(1, min(int(settings.rag_dense_candidate_k), len(chunks)))
    reranker = (
        _build_local_reranker(settings.rag_rerank_model)
        if project_rerank_enabled
        else None
    )

    def search_project_evidence(query: str) -> dict[str, Any]:
        query_vector = _normalize_embedding_vector(embeddings.embed_query(query))
        dense_candidates = _cosine_topk(
            query_vector=query_vector,
            candidate_vectors=chunk_embeddings,
            k=dense_k,
        )
        docs = [
            SimpleNamespace(page_content=chunks[index], metadata=metadatas[index])
            for index, _score in dense_candidates
        ]
        ranked_chunks = _rerank_docs(
            query=query,
            docs=docs,
            reranker=reranker,
            top_k=max(1, min(int(settings.rag_top_k), len(docs) if docs else 1)),
        )

        used_indices: set[int] = set()
        evidences: list[EvidenceItem] = []
        for rank, chunk in enumerate(ranked_chunks):
            matched_idx: int | None = None
            for i, doc in enumerate(docs):
                if i in used_indices:
                    continue
                if doc.page_content == chunk:
                    matched_idx = i
                    used_indices.add(i)
                    break
            metadata: dict[str, Any] = {}
            if matched_idx is not None and isinstance(docs[matched_idx].metadata, dict):
                metadata = docs[matched_idx].metadata

            source_doc_uid = metadata.get("doc_uid")
            source_doc_name = metadata.get("doc_name")
            source_chunk_id = metadata.get("chunk_id")
            offset_start = metadata.get("start_index")
            if not isinstance(offset_start, int):
                offset_start = None
            evidences.append(
                EvidenceItem(
                    project_uid=project_uid,
                    doc_uid=source_doc_uid if isinstance(source_doc_uid, str) else "",
                    doc_name=source_doc_name if isinstance(source_doc_name, str) else "",
                    chunk_id=(
                        source_chunk_id
                        if isinstance(source_chunk_id, str) and source_chunk_id
                        else f"chunk_{rank}"
                    ),
                    text=chunk,
                    score=float(1.0 / (rank + 1)),
                    offset_start=offset_start,
                    offset_end=(offset_start + len(chunk)) if isinstance(offset_start, int) else None,
                )
            )

        payload = EvidencePayload(
            evidences=evidences,
            trace={
                "mode": "project_dense",
                "project_uid": project_uid,
                "candidate_count": len(docs),
                "doc_count": doc_count,
                "top_k": settings.rag_top_k,
                "dense_k": dense_k,
                "rerank_enabled": project_rerank_enabled,
                "truncated_by_chars": truncated_by_chars,
                "truncated_by_chunks": truncated_by_chunks,
                "index_cache_reused_docs": reused_doc_indexes,
                "index_cache_built_docs": built_doc_indexes,
            },
        )
        return payload.model_dump()

    return search_project_evidence


def build_local_vector_retriever_with_settings(document_text: str) -> Callable[[str], str]:
    """使用 settings 配置构建向量检索器（兼容 local_rag.py 接口）"""
    settings = load_agent_settings()

    if settings.rag_hybrid_enabled:
        # 使用 Hybrid 检索
        logger.info("启用 Hybrid 检索 (Dense + BM25 + RRF)")
        return build_hybrid_retriever(
            document_text=document_text,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            embedding_model=settings.local_embedding_model,
            embedding_cache_dir=settings.local_embedding_cache_dir,
            dense_k=settings.rag_dense_candidate_k,
            sparse_k=settings.rag_sparse_candidate_k,
            rrf_k=settings.rag_rrf_candidate_k,
            rerank_enabled=settings.rag_rerank_enabled,
            rerank_model=settings.rag_rerank_model,
            neighbor_expansion=settings.rag_neighbor_expansion,
            neighbor_count=settings.rag_neighbor_count,
        )
    else:
        # 使用原有的纯 Dense 检索
        from .local import build_local_vector_retriever
        return build_local_vector_retriever(document_text)


def preprocess_query(
    query: str,
    enabled: bool = False,
    llm_client: Any = None,
) -> list[str]:
    """Query 预处理：同义改写/多子问题拆分

    Args:
        query: 原始查询
        enabled: 是否启用预处理
        llm_client: LLM 客户端（可选）

    Returns:
        处理后的查询列表
    """
    if not enabled or not query:
        return [query]

    if llm_client is None:
        # 无 LLM 时返回原查询
        return [query]

    try:
        # 使用 LLM 进行查询改写
        prompt = f"""将以下复杂问题拆分为多个简单子问题，以便于文档检索。
返回多个子问题，用换行分隔。不要添加任何解释或前缀。

原问题：{query}

子问题："""

        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        result = response.choices[0].message.content
        # 解析结果
        sub_queries = [
            q.strip()
            for q in result.split("\n")
            if q.strip()
        ]

        if sub_queries:
            return sub_queries

    except Exception as e:
        logger.warning(f"Query 预处理失败: {e}")

    return [query]


def query_rewrite(
    query: str,
    llm_client: Any,
    rewrite_enabled: bool = True,
) -> str:
    """Query 同义改写

    将用户问题改写为更适合文档检索的表述，
    保留核心语义的同时优化检索匹配度。

    Args:
        query: 原始查询
        llm_client: LLM 客户端（必需）
        rewrite_enabled: 是否启用改写

    Returns:
        改写后的问题字符串
    """
    if not rewrite_enabled or not query:
        return query

    if llm_client is None:
        logger.warning("无 LLM 客户端，无法执行 query rewrite")
        return query

    try:
        prompt = f"""将以下问题改写为更易于文档检索的形式。
改写时保持原问题的核心语义，但使用更通用、更可能被文档匹配的表述。
只返回改写后的问题，不要添加任何解释。

原问题：{query}

改写后："""

        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        result = response.choices[0].message.content.strip()
        if result:
            logger.info(f"Query 改写: {query} -> {result}")
            return result

    except Exception as e:
        logger.warning(f"Query 改写失败: {e}")

    return query


def query_split(
    query: str,
    llm_client: Any,
    split_enabled: bool = True,
) -> list[str]:
    """Query 子问题拆分

    将复杂问题拆分为多个简单的子问题，
    便于分别检索后合并答案。

    Args:
        query: 原始查询
        llm_client: LLM 客户端（必需）
        split_enabled: 是否启用拆分

    Returns:
        子问题列表（如果不需要拆分则返回原问题单元素列表）
    """
    if not split_enabled or not query:
        return [query]

    if llm_client is None:
        logger.warning("无 LLM 客户端，无法执行 query split")
        return [query]

    try:
        prompt = f"""分析以下问题，判断是否需要拆分为多个子问题才能完整回答。
如果只需要一个简单问题就能回答，返回原问题。
如果需要多角度或多步骤才能完整回答，将其拆分为多个子问题。

返回格式：
- 如果不需要拆分：直接返回原问题
- 如果需要拆分：用换行分隔多个子问题

原问题：{query}

"""

        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        result = response.choices[0].message.content.strip()
        if not result:
            return [query]

        # 解析结果
        sub_queries = [
            q.strip()
            for q in result.split("\n")
            if q.strip() and q.strip() != query
        ]

        if sub_queries:
            logger.info(f"Query 拆分: {query} -> {sub_queries}")
            return sub_queries

    except Exception as e:
        logger.warning(f"Query 拆分失败: {e}")

    return [query]
