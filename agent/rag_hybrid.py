"""Hybrid 检索模块

实现 BM25 稀疏检索 + Dense 密集检索 + RRF 融合
支持配置化开关和失败回退
支持邻域扩展和证据来源输出
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .settings import load_agent_settings

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        chunks: list[str],
        embedding_model: str,
        embedding_cache_dir: str,
        dense_k: int = 30,
        sparse_k: int = 30,
        rrf_k: int = 40,
        rerank_enabled: bool = True,
        rerank_model: str = "ms-marco-MiniLM-L-12-v2",
        neighbor_expansion: bool = True,
        neighbor_count: int = 1,
    ):
        self.chunks = chunks
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k
        self.rerank_enabled = rerank_enabled
        self.rerank_model = rerank_model
        self.neighbor_expansion = neighbor_expansion
        self.neighbor_count = neighbor_count

        # 初始化 Dense 检索（向量检索）
        embeddings = FastEmbedEmbeddings(
            model_name=embedding_model,
            cache_dir=embedding_cache_dir,
        )
        self.vectorstore = InMemoryVectorStore.from_texts(chunks, embedding=embeddings)

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
                for idx, doc in enumerate(retrieved_docs):
                    if doc.page_content == chunk:
                        original_idx = fused_indices[idx] if idx < len(fused_indices) else idx
                        break
                sources.append({
                    "chunk_id": f"chunk_{original_idx}" if original_idx is not None else f"chunk_{i}",
                    "index": original_idx if original_idx is not None else i,
                    "score": 1.0 / (i + 1) if i < len(fused_results) else 0,
                })
            return RetrievalResult(
                chunks=ranked_chunks,
                sources=sources,
                metadata={"trace": trace.__dict__},
            )

        return ranked_chunks


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

    # 创建混合检索器
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        embedding_cache_dir=embedding_cache_dir,
        dense_k=dense_k,
        sparse_k=sparse_k,
        rrf_k=rrf_k,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        neighbor_expansion=neighbor_expansion,
        neighbor_count=neighbor_count,
    )

    def search_document(query: str) -> str:
        results = retriever.search(query)
        if isinstance(results, RetrievalResult):
            return "\n".join(results.chunks)
        return "\n".join(results)

    return search_document


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
        from .local_rag import build_local_vector_retriever
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
