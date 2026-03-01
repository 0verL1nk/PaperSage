from agent import llm_provider as provider
from agent.settings import AgentSettings


def _settings(*, enable_thinking: bool, reasoning_effort: str) -> AgentSettings:
    return AgentSettings(
        openai_compatible_base_url="https://example.com/v1",
        local_embedding_model="m1",
        local_embedding_fallback_model="m2",
        local_embedding_cache_dir="./cache",
        rag_chunk_size=500,
        rag_chunk_overlap=80,
        rag_dense_candidate_k=30,
        rag_sparse_candidate_k=30,
        rag_rrf_candidate_k=40,
        rag_rerank_candidate_k=50,
        rag_top_k=8,
        rag_rerank_enabled=True,
        rag_rerank_model="r1",
        rag_hybrid_enabled=False,
        rag_neighbor_expansion=True,
        rag_neighbor_count=1,
        rag_query_preprocess_enabled=False,
        agent_temperature=0.2,
        agent_enable_thinking=enable_thinking,
        agent_reasoning_effort=reasoning_effort,
    )


def test_build_model_sets_dashscope_enable_thinking(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        provider,
        "load_agent_settings",
        lambda: _settings(enable_thinking=True, reasoning_effort="medium"),
    )
    monkeypatch.setattr(provider, "ChatOpenAI", fake_chat_openai)

    provider.build_openai_compatible_chat_model(
        api_key="k",
        model_name="m",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert captured["extra_body"] == {"enable_thinking": True}
    assert captured["reasoning_effort"] is None


def test_build_model_sets_openai_reasoning_effort(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        provider,
        "load_agent_settings",
        lambda: _settings(enable_thinking=True, reasoning_effort="high"),
    )
    monkeypatch.setattr(provider, "ChatOpenAI", fake_chat_openai)

    provider.build_openai_compatible_chat_model(
        api_key="k",
        model_name="m",
        base_url="https://api.openai.com/v1",
    )

    assert captured["reasoning_effort"] == "high"
    assert captured["extra_body"] is None


def test_build_model_disables_thinking_when_overridden(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        provider,
        "load_agent_settings",
        lambda: _settings(enable_thinking=True, reasoning_effort="high"),
    )
    monkeypatch.setattr(provider, "ChatOpenAI", fake_chat_openai)

    provider.build_openai_compatible_chat_model(
        api_key="k",
        model_name="m",
        base_url="https://api.openai.com/v1",
        enable_thinking=False,
        reasoning_effort="",
    )

    assert captured["reasoning_effort"] is None
    assert captured["extra_body"] is None
