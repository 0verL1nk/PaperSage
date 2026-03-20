from agent.middlewares.builder import build_middleware_list
from agent.middlewares.orchestration import OrchestrationMiddleware
from agent.middlewares.plan import plan_middleware
from agent.middlewares.team import TeamMiddleware
from agent.middlewares.todolist import todolist_middleware
from agent.profiles import paper_leader_profile, paper_worker_profile


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
def test_build_middleware_list_filters_team_middlewares_by_profile(monkeypatch):
    monkeypatch.setattr("agent.middlewares.builder.load_subagent_configs", lambda: [])

    leader_middlewares = build_middleware_list(
        model=object(),
        profile=paper_leader_profile,
        deps=object(),
        enable_auto_summarization=False,
        enable_tool_selector=False,
    )
    worker_middlewares = build_middleware_list(
        model=object(),
        profile=paper_worker_profile,
        deps=object(),
        enable_auto_summarization=False,
        enable_tool_selector=False,
    )

    assert any(isinstance(item, TeamMiddleware) for item in leader_middlewares)
    assert todolist_middleware in leader_middlewares
    assert plan_middleware in leader_middlewares
    assert any(isinstance(item, OrchestrationMiddleware) for item in leader_middlewares)

    assert not any(isinstance(item, TeamMiddleware) for item in worker_middlewares)
    assert todolist_middleware not in worker_middlewares
    assert plan_middleware not in worker_middlewares
    assert any(isinstance(item, OrchestrationMiddleware) for item in worker_middlewares)
