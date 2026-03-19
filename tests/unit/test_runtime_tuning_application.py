from agent.application.runtime_tuning import apply_runtime_tuning_env


def test_apply_runtime_tuning_env_sets_and_clears_values():
    environ = {
        "AGENT_POLICY_ASYNC_ENABLED": "false",
        "AGENT_POLICY_ASYNC_REFRESH_SECONDS": "9.0",
        "RAG_INDEX_BATCH_SIZE": "128",
    }
    applied = apply_runtime_tuning_env(
        settings={
            "agent_policy_async_enabled": True,
            "agent_policy_async_refresh_seconds": 2.0,
            "agent_policy_async_min_confidence": None,
            "agent_policy_async_max_staleness_seconds": None,
            "rag_index_batch_size": None,
            "agent_document_text_cache_max_chars": 1000,
            "local_rag_project_max_chars": None,
            "local_rag_project_max_chunks": 50,
        },
        environ=environ,
    )

    assert "AGENT_POLICY_ASYNC_ENABLED" not in environ
    assert "AGENT_POLICY_ASYNC_REFRESH_SECONDS" not in environ
    assert "RAG_INDEX_BATCH_SIZE" not in environ
    assert environ["AGENT_DOCUMENT_TEXT_CACHE_MAX_CHARS"] == "1000"
    assert environ["LOCAL_RAG_PROJECT_MAX_CHUNKS"] == "50"
    assert "AGENT_POLICY_ASYNC_ENABLED" not in applied
