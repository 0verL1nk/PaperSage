from agent import runtime_agent


def test_build_runtime_tools_blocks_spawn_tools(monkeypatch):
    class _FakeBrave:
        def run(self, query):
            return f"web:{query}"

    monkeypatch.setattr(runtime_agent, "build_agent_tools", lambda **kwargs: kwargs["allowed_tools"])
    monkeypatch.setattr(
        runtime_agent,
        "discover_available_tools",
        lambda **_kwargs: [
            type("ToolMeta", (), {"name": "search_document"})(),
            type("ToolMeta", (), {"name": "start_plan"})(),
            type("ToolMeta", (), {"name": "start_team"})(),
            type("ToolMeta", (), {"name": "use_skill"})(),
        ],
    )

    allowed = runtime_agent.build_runtime_tools(
        search_document_fn=lambda query: query,
        blocked_tools=runtime_agent.SPAWN_TOOL_NAMES,
    )

    assert allowed == {"search_document", "use_skill"}
