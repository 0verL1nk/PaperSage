from agent.profiles import paper_worker_profile
from agent.session_factory import AgentDependencies
from agent.tools import team as team_tools


class _FakeRuntime:
    def __init__(self) -> None:
        self.captured: dict[str, object] = {}

    def spawn_agent(self, **kwargs):
        self.captured.update(kwargs)
        return "agent-123"



def test_spawn_agent_tool_uses_role_alias_and_session_context(monkeypatch):
    runtime = _FakeRuntime()
    deps = AgentDependencies(search_document_fn=lambda query: query)

    monkeypatch.setattr(team_tools, "get_team_runtime", lambda: runtime)
    team_tools.set_current_session("thread-1")
    team_tools.set_session_runtime_context(
        "thread-1",
        default_model="fake-llm",
        dependencies=deps,
    )

    result = team_tools.spawn_agent.invoke(
        {"name": "researcher", "role": "teammate", "system_prompt": "WORKER_OVERRIDE"}
    )

    assert result == "Agent spawned: agent-123 (role=teammate, profile=paper_worker)"
    assert runtime.captured["name"] == "researcher"
    assert runtime.captured["model"] == "fake-llm"
    assert runtime.captured["profile"] is paper_worker_profile
    assert runtime.captured["deps"] is deps
    assert runtime.captured["system_prompt"] == "WORKER_OVERRIDE"
