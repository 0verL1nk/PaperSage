from agent.middlewares.builder import build_middleware_list


def test_build_middleware_list_builds_typed_subagents_for_subagent_middleware(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.middlewares.builder.load_subagent_configs",
        lambda: [{"name": "researcher", "description": "d", "system_prompt": "p"}],
    )

    def _fake_subagent_middleware(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "agent.middlewares.builder.SubAgentMiddleware",
        _fake_subagent_middleware,
    )

    build_middleware_list(model="llm", enable_auto_summarization=False, enable_tool_selector=False)

    assert "backend" in captured
    assert captured["subagents"] == [
        {
            "name": "researcher",
            "description": "d",
            "system_prompt": "p",
            "model": "llm",
            "tools": [],
        }
    ]
