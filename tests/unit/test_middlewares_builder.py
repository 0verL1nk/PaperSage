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


def test_build_middleware_list_includes_mindmap_format_middleware() -> None:
    from unittest.mock import patch

    with patch("agent.middlewares.builder.load_subagent_configs", return_value=[]):
        middlewares = build_middleware_list(
            model="llm",
            enable_auto_summarization=False,
            enable_tool_selector=False,
        )

    assert any(
        middleware.__class__.__name__ == "MindmapFormatMiddleware" for middleware in middlewares
    )
