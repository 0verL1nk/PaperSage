"""RAG Hybrid 模块单元测试

测试 Hybrid 检索、RRF 融合、邻域扩展功能
"""
from types import SimpleNamespace

import pytest

from agent.rag.hybrid import (
    HybridRetriever,
    RetrievalResult,
    RetrievalTrace,
    _normalize_scores,
    _reciprocal_rank_fusion,
    _rerank_docs,
    _retrieval_result_to_evidence_payload,
    build_hybrid_retriever,
    build_project_evidence_retriever_with_settings,
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

    def test_retrieval_result_to_evidence_payload(self):
        result = RetrievalResult(
            chunks=["chunk-content"],
            sources=[{"chunk_id": "chunk_0", "index": 0, "score": 0.9}],
            metadata={"trace": {"dense_candidate_count": 10}},
        )
        payload = _retrieval_result_to_evidence_payload(result, doc_uid="doc-1")
        assert payload["evidences"][0]["doc_uid"] == "doc-1"
        assert payload["evidences"][0]["chunk_id"] == "chunk_0"
        assert payload["trace"]["dense_candidate_count"] == 10


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


class TestQueryRewrite:
    """测试 Query 同义改写"""

    def test_query_rewrite_disabled(self):
        """测试关闭改写时返回原查询"""
        from agent.rag.hybrid import query_rewrite

        result = query_rewrite("什么是机器学习", None, rewrite_enabled=False)
        assert result == "什么是机器学习"

    def test_query_rewrite_no_llm(self):
        """测试无 LLM 时返回原查询"""
        from agent.rag.hybrid import query_rewrite

        result = query_rewrite("什么是机器学习", None, rewrite_enabled=True)
        assert result == "什么是机器学习"

    def test_query_rewrite_empty_query(self):
        """测试空查询"""
        from agent.rag.hybrid import query_rewrite

        result = query_rewrite("", None, rewrite_enabled=True)
        assert result == ""

    def test_query_rewrite_with_mock(self):
        """测试有 LLM 时的改写"""
        from agent.rag.hybrid import query_rewrite

        # 使用简单的 mock 对象
        class MockMsg:
            content = "机器学习的定义是什么"

        class MockChoice:
            message = MockMsg()

        class MockResponse:
            choices = [MockChoice()]

        class MockCompletions:
            @staticmethod
            def create(**kwargs):
                return MockResponse()

        class MockChat:
            completions = MockCompletions()

        class MockLLM:
            chat = MockChat()

        result = query_rewrite("什么是机器学习", MockLLM(), rewrite_enabled=True)
        assert result == "机器学习的定义是什么"


class TestQuerySplit:
    """测试 Query 子问题拆分"""

    def test_query_split_disabled(self):
        """测试关闭拆分时返回单元素列表"""
        from agent.rag.hybrid import query_split

        result = query_split("什么是机器学习", None, split_enabled=False)
        assert result == ["什么是机器学习"]

    def test_query_split_no_llm(self):
        """测试无 LLM 时返回单元素列表"""
        from agent.rag.hybrid import query_split

        result = query_split("什么是机器学习", None, split_enabled=True)
        assert result == ["什么是机器学习"]

    def test_query_split_empty_query(self):
        """测试空查询"""
        from agent.rag.hybrid import query_split

        result = query_split("", None, split_enabled=True)
        assert result == [""]

    def test_query_split_with_mock(self):
        """测试有 LLM 时的拆分"""
        from agent.rag.hybrid import query_split

        # 使用简单的 mock 对象
        class MockMsg:
            content = "机器学习的定义\n机器学习的应用场景\n机器学习的算法类型"

        class MockChoice:
            message = MockMsg()

        class MockResponse:
            choices = [MockChoice()]

        class MockCompletions:
            @staticmethod
            def create(**kwargs):
                return MockResponse()

        class MockChat:
            completions = MockCompletions()

        class MockLLM:
            chat = MockChat()

        result = query_split("什么是机器学习", MockLLM(), split_enabled=True)
        assert len(result) == 3
        assert "机器学习的定义" in result


class TestHybridRetrieverQueryPreprocess:
    """测试 HybridRetriever Query 预处理集成"""

    def test_set_llm_client(self):
        """测试设置 LLM 客户端"""
        chunks = ["chunk0", "chunk1", "chunk2"]
        retriever = HybridRetriever(
            chunks=chunks,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_cache_dir="./models/embeddings",
            query_preprocess_enabled=True,
        )

        class MockLLM:
            pass

        retriever.set_llm_client(MockLLM())
        assert retriever.llm_client is not None

    def test_search_without_preprocess(self):
        """测试不启用预处理时的检索"""
        chunks = ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4"]
        retriever = HybridRetriever(
            chunks=chunks,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_cache_dir="./models/embeddings",
            dense_k=3,
            rerank_enabled=False,
            query_preprocess_enabled=False,
        )

        result = retriever.search("chunk1", top_k=2)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_search_with_preprocess_no_llm(self):
        """测试启用预处理但无 LLM 时的检索"""
        chunks = ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4"]
        retriever = HybridRetriever(
            chunks=chunks,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_cache_dir="./models/embeddings",
            dense_k=3,
            rerank_enabled=False,
            query_preprocess_enabled=True,
        )

        result = retriever.search("chunk1", top_k=2)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_query_preprocess_params_in_constructor(self):
        """测试构造函数参数"""
        chunks = ["chunk0", "chunk1"]
        retriever = HybridRetriever(
            chunks=chunks,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_cache_dir="./models/embeddings",
            query_preprocess_enabled=True,
            query_rewrite_enabled=False,
            query_split_enabled=False,
        )

        assert retriever.query_preprocess_enabled is True
        assert retriever.query_rewrite_enabled is False
        assert retriever.query_split_enabled is False


def test_project_retriever_reuses_persisted_doc_indexes(monkeypatch, tmp_path):
    from agent.rag import hybrid as hybrid_module

    class _FakeEmbeddings:
        embed_documents_calls = 0

        def __init__(self, **_kwargs):
            pass

        def embed_documents(self, texts):
            _FakeEmbeddings.embed_documents_calls += 1
            return [[float(len(str(text)))] for text in texts]

        def embed_query(self, query):
            return [float(len(str(query)))]

    fake_settings = SimpleNamespace(
        local_embedding_model="fake-embedding",
        local_embedding_cache_dir=str(tmp_path / "embeddings"),
        rag_chunk_size=8,
        rag_chunk_overlap=0,
        rag_dense_candidate_k=5,
        rag_top_k=2,
        rag_rerank_enabled=False,
        rag_project_rerank_enabled=False,
        rag_rerank_model="fake-rerank",
        rag_project_max_chars=10000,
        rag_project_max_chunks=1000,
    )

    monkeypatch.setenv("AGENT_PROJECT_INDEX_CACHE_DIR", str(tmp_path / "project_indexes"))
    monkeypatch.setattr(hybrid_module, "FastEmbedEmbeddings", _FakeEmbeddings)
    monkeypatch.setattr(hybrid_module, "load_agent_settings", lambda: fake_settings)

    documents = [
        {"doc_uid": "d1", "doc_name": "doc1", "text": "alpha beta gamma"},
        {"doc_uid": "d2", "doc_name": "doc2", "text": "delta epsilon zeta"},
    ]

    _FakeEmbeddings.embed_documents_calls = 0
    retriever_first = build_project_evidence_retriever_with_settings(
        documents=documents,
        project_uid="p-cache",
    )
    assert callable(retriever_first)
    first_calls = _FakeEmbeddings.embed_documents_calls
    assert first_calls > 0

    payload_first = retriever_first("alpha")
    assert isinstance(payload_first, dict)
    assert "evidences" in payload_first

    retriever_second = build_project_evidence_retriever_with_settings(
        documents=documents,
        project_uid="p-cache",
    )
    assert callable(retriever_second)
    assert _FakeEmbeddings.embed_documents_calls == first_calls

    payload_second = retriever_second("alpha")
    trace = payload_second.get("trace", {})
    assert int(trace.get("index_cache_reused_docs", 0)) >= 1


def test_project_retriever_returns_empty_for_low_relevance(monkeypatch, tmp_path):
    from agent.rag import hybrid as hybrid_module

    class _FakeEmbeddings:
        def __init__(self, **_kwargs):
            pass

        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                text_lower = str(text).lower()
                if "alpha" in text_lower:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

        def embed_query(self, _query):
            return [1.0, 0.0]

    fake_settings = SimpleNamespace(
        local_embedding_model="fake-embedding",
        local_embedding_cache_dir=str(tmp_path / "embeddings"),
        rag_chunk_size=1024,
        rag_chunk_overlap=0,
        rag_dense_candidate_k=5,
        rag_top_k=2,
        rag_rerank_enabled=False,
        rag_project_rerank_enabled=False,
        rag_rerank_model="fake-rerank",
        rag_project_max_chars=10000,
        rag_project_max_chunks=1000,
        rag_min_relevance_score=0.95,
    )

    monkeypatch.setenv("AGENT_PROJECT_INDEX_CACHE_DIR", str(tmp_path / "project_indexes"))
    monkeypatch.setattr(hybrid_module, "FastEmbedEmbeddings", _FakeEmbeddings)
    monkeypatch.setattr(hybrid_module, "load_agent_settings", lambda: fake_settings)

    documents = [
        {"doc_uid": "d1", "doc_name": "doc1", "text": "beta only content"},
    ]
    retriever = build_project_evidence_retriever_with_settings(
        documents=documents,
        project_uid="p-empty",
    )

    payload = retriever("alpha query")
    assert payload["evidences"] == []
    trace = payload.get("trace", {})
    assert trace.get("no_relevant_candidates") is True
    assert trace.get("candidate_count") == 0
    assert trace.get("candidate_count_raw", 0) >= 1


def test_project_retriever_uses_similarity_score(monkeypatch, tmp_path):
    from agent.rag import hybrid as hybrid_module

    class _FakeEmbeddings:
        def __init__(self, **_kwargs):
            pass

        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                text_lower = str(text).lower()
                if "alpha" in text_lower:
                    vectors.append([1.0, 0.0])
                elif "mix" in text_lower:
                    vectors.append([0.6, 0.8])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

        def embed_query(self, _query):
            return [1.0, 0.0]

    fake_settings = SimpleNamespace(
        local_embedding_model="fake-embedding",
        local_embedding_cache_dir=str(tmp_path / "embeddings"),
        rag_chunk_size=1024,
        rag_chunk_overlap=0,
        rag_dense_candidate_k=5,
        rag_top_k=2,
        rag_rerank_enabled=False,
        rag_project_rerank_enabled=False,
        rag_rerank_model="fake-rerank",
        rag_project_max_chars=10000,
        rag_project_max_chunks=1000,
        rag_min_relevance_score=0.0,
    )

    monkeypatch.setenv("AGENT_PROJECT_INDEX_CACHE_DIR", str(tmp_path / "project_indexes"))
    monkeypatch.setattr(hybrid_module, "FastEmbedEmbeddings", _FakeEmbeddings)
    monkeypatch.setattr(hybrid_module, "load_agent_settings", lambda: fake_settings)

    documents = [
        {"doc_uid": "d1", "doc_name": "doc1", "text": "alpha exact match"},
        {"doc_uid": "d2", "doc_name": "doc2", "text": "mix partial match"},
    ]
    retriever = build_project_evidence_retriever_with_settings(
        documents=documents,
        project_uid="p-score",
    )

    payload = retriever("alpha query")
    scores = [item.get("score") for item in payload.get("evidences", [])]
    assert len(scores) == 2
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(0.6, abs=1e-6)
