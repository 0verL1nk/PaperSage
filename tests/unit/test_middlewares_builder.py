from agent.middlewares.builder import build_middleware_list


def test_build_middleware_list_passes_default_model_to_subagent_middleware(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.middlewares.builder.load_subagent_configs",
        lambda: [{"name": "researcher", "description": "d", "system_prompt": "p"}],
    )

    def _fake_subagent_middleware(**kwargs):
        captured.update(kwargs)
        if "default_model" not in kwargs:
            raise ValueError("default_model is required")
        return object()

    monkeypatch.setattr(
        "agent.middlewares.builder.SubAgentMiddleware",
        _fake_subagent_middleware,
    )

    build_middleware_list(model="llm", enable_auto_summarization=False, enable_tool_selector=False)

    assert captured["default_model"] == "llm"
    assert captured["subagents"] == [{"name": "researcher", "description": "d", "system_prompt": "p"}]
