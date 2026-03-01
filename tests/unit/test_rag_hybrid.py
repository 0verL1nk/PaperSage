"""RAG Hybrid 模块单元测试

测试 Hybrid 检索、RRF 融合、邻域扩展功能
"""
import pytest
from types import SimpleNamespace

from agent.rag_hybrid import (
    _reciprocal_rank_fusion,
    _normalize_scores,
    _rerank_docs,
    _build_bm25_retriever,
    HybridRetriever,
    build_hybrid_retriever,
    RetrievalResult,
    RetrievalTrace,
)


class TestReciprocalRankFusion:
    """测试 RRF 融合"""

    def test_rrf_fusion_with_empty_results(self):
        """测试空结果"""
        result = _reciprocal_rank_fusion([], [], k=60)
        assert result == []

    def test_rrf_fusion_with_dense_only(self):
        """测试只有 Dense 结果"""
        dense = [(0, 1.0), (1, 0.8), (2, 0.6)]
        result = _reciprocal_rank_fusion(dense, [], k=60)
        assert len(result) == 3
        # 应该按原始顺序排列
        assert result[0][0] == 0

    def test_rrf_fusion_with_both_results(self):
        """测试两种结果融合"""
        dense = [(0, 1.0), (1, 0.8), (2, 0.6)]
        sparse = [(1, 1.0), (3, 0.9), (0, 0.7)]
        result = _reciprocal_rank_fusion(dense, sparse, k=60)
        # 应该有 4 个唯一索引
        assert len(result) == 4
        # 检查结果按 score 降序
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


class TestNormalizeScores:
    """测试分数归一化"""

    def test_normalize_empty(self):
        """测试空列表"""
        assert _normalize_scores([]) == []

    def test_normalize_single(self):
        """测试单个分数"""
        assert _normalize_scores([1.0]) == [1.0]

    def test_normalize_multiple(self):
        """测试多个分数"""
        result = _normalize_scores([0.0, 0.5, 1.0])
        assert result == [0.0, 0.5, 1.0]

    def test_normalize_same_values(self):
        """测试相同分数"""
        result = _normalize_scores([1.0, 1.0, 1.0])
        assert result == [1.0, 1.0, 1.0]


class TestRerankDocs:
    """测试重排序"""

    def test_rerank_empty_docs(self):
        """测试空文档"""
        result = _rerank_docs("query", [], None, 10)
        assert result == []

    def test_rerank_no_reranker(self):
        """测试无 reranker"""
        docs = [
            SimpleNamespace(page_content="A"),
            SimpleNamespace(page_content="B"),
            SimpleNamespace(page_content="C"),
        ]
        result = _rerank_docs("query", docs, None, 2)
        assert result == ["A", "B"]

    def test_rerank_with_reranker(self):
        """测试有 reranker"""
        docs = [
            SimpleNamespace(page_content="A"),
            SimpleNamespace(page_content="B"),
            SimpleNamespace(page_content="C"),
        ]

        class FakeReranker:
            def rerank(self, request):
                # 返回排序后的结果
                return [{"text": "C"}, {"text": "A"}, {"text": "B"}]

        result = _rerank_docs("query", docs, FakeReranker(), 2)
        assert result == ["C", "A"]

    def test_rerank_fallback_on_error(self):
        """测试 rerank 失败回退"""
        docs = [
            SimpleNamespace(page_content="A"),
            SimpleNamespace(page_content="B"),
        ]

        class BadReranker:
            def rerank(self, request):
                raise RuntimeError("rerank failed")

        result = _rerank_docs("query", docs, BadReranker(), 1)
        assert result == ["A"]


class TestHybridRetriever:
    """测试混合检索器"""

    def test_neighbor_expansion(self):
        """测试邻域扩展"""
        chunks = ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4"]

        # 使用本地 embedding 模型进行测试（使用最小的）
        retriever = HybridRetriever(
            chunks=chunks,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_cache_dir="./models/embeddings",
            dense_k=3,
            sparse_k=3,
            rrf_k=5,
            rerank_enabled=False,
            neighbor_expansion=True,
            neighbor_count=1,
        )

        # 测试邻域扩展
        indices = [2]  # 只选择中间的 chunk
        expanded = retriever._expand_neighbors(indices)
        # 应该包含 2 以及前后邻居 1 和 3
        assert 2 in expanded
        assert 1 in expanded
        assert 3 in expanded
        # 不应包含太远的
        assert 0 not in expanded
        assert 4 not in expanded

    def test_neighbor_expansion_disabled(self):
        """测试关闭邻域扩展"""
        chunks = ["chunk0", "chunk1", "chunk2"]
        retriever = HybridRetriever(
            chunks=chunks,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_cache_dir="./models/embeddings",
            neighbor_expansion=False,
        )

        indices = [1]
        expanded = retriever._expand_neighbors(indices)
        assert expanded == [1]


class TestBuildHybridRetriever:
    """测试工厂函数"""

    def test_build_with_defaults(self):
        """测试默认参数"""
        retriever = build_hybrid_retriever(
            "这是测试文本。",
            rerank_enabled=False,
        )
        assert callable(retriever)

    def test_build_with_custom_params(self):
        """测试自定义参数"""
        retriever = build_hybrid_retriever(
            "这是测试文本。",
            chunk_size=200,
            chunk_overlap=30,
            dense_k=5,
            sparse_k=5,
            rrf_k=10,
            rerank_enabled=False,
            neighbor_expansion=True,
            neighbor_count=2,
        )
        assert callable(retriever)


class TestRetrievalResult:
    """测试检索结果类"""

    def test_retrieval_result_creation(self):
        """测试结果创建"""
        result = RetrievalResult(
            chunks=["chunk1", "chunk2"],
            sources=[{"chunk_id": "chunk_0"}, {"chunk_id": "chunk_1"}],
            metadata={"trace": {"dense_count": 10}},
        )
        assert len(result.chunks) == 2
        assert len(result.sources) == 2


class TestRetrievalTrace:
    """测试检索轨迹"""

    def test_retrieval_trace_creation(self):
        """测试轨迹创建"""
        trace = RetrievalTrace(
            dense_candidate_count=30,
            sparse_candidate_count=30,
            rrf_candidate_count=40,
            rerank_input_count=20,
            final_count=8,
            neighbor_expanded=True,
            fallback_reason=None,
        )
        assert trace.dense_candidate_count == 30
        assert trace.neighbor_expanded is True
