from agent import runtime_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def test_build_runtime_tools_blocks_spawn_tools(monkeypatch):
    class _FakeBrave:
        def run(self, query):
            return f"web:{query}"

    monkeypatch.setattr(
        runtime_agent, "build_agent_tools", lambda **kwargs: kwargs["allowed_tools"]
    )
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


class TestCreateRuntimeAgent:
    def test_default_uses_sqlite_checkpointer(self, monkeypatch):
        """Test that create_runtime_agent uses sqlite checkpointer by default."""
        created_agent = None

        # Mock create_agent to capture the checkpointer argument
        def mock_create_agent(**kwargs):
            nonlocal created_agent
            created_agent = kwargs
            return "agent_instance"

        monkeypatch.setattr(runtime_agent, "create_agent", mock_create_agent)
        monkeypatch.setattr(runtime_agent, "build_progressive_tool_middleware", lambda tools: None)

        # Call without checkpointer - should default to sqlite
        runtime_agent.create_runtime_agent(
            model="mock_model",
            system_prompt="test prompt",
            tools=[],
        )

        # Verify sqlite checkpointer was created by default
        assert created_agent is not None
        checkpointer = created_agent.get("checkpointer")
        assert checkpointer is not None
        assert isinstance(checkpointer, SqliteSaver)

    def test_explicit_checkpointer_overrides_default(self, monkeypatch):
        """Test that explicit checkpointer is used when provided."""
        created_agent = None

        def mock_create_agent(**kwargs):
            nonlocal created_agent
            created_agent = kwargs
            return "agent_instance"

        monkeypatch.setattr(runtime_agent, "create_agent", mock_create_agent)
        monkeypatch.setattr(runtime_agent, "build_progressive_tool_middleware", lambda tools: None)

        # Provide explicit InMemorySaver
        explicit_checkpointer = InMemorySaver()
        runtime_agent.create_runtime_agent(
            model="mock_model",
            system_prompt="test prompt",
            tools=[],
            checkpointer=explicit_checkpointer,
        )

        # Verify explicit checkpointer was used
        assert created_agent is not None
        checkpointer = created_agent.get("checkpointer")
        assert checkpointer is explicit_checkpointer
        assert isinstance(checkpointer, InMemorySaver)
