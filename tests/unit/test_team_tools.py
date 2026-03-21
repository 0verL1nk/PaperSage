from agent.profiles import paper_worker_profile
from agent.session_factory import AgentDependencies
from agent.tools import team as team_tools


class _FakeRuntime:
    def __init__(self) -> None:
        self.captured: dict[str, object] = {}
        self.agents_payload: list[dict[str, object]] = []

    def spawn_agent(self, **kwargs):
        self.captured.update(kwargs)
        return "agent-123"

    def list_agents(self):
        return self.agents_payload

    def send_message(self, agent_id: str, message: str) -> None:
        self.captured["send_agent_id"] = agent_id
        self.captured["send_message"] = message

    def close_agent(self, agent_id: str) -> None:
        self.captured["closed_agent_id"] = agent_id


class _BusyRuntime(_FakeRuntime):
    def send_message(self, agent_id: str, message: str) -> None:
        raise ValueError(f"Agent {agent_id} is busy")

    def close_agent(self, agent_id: str) -> None:
        raise ValueError(f"Cannot close agent {agent_id}: still busy")



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


def test_spawn_agent_tool_resolves_runtime_context_from_config_thread_id(monkeypatch):
    runtime = _FakeRuntime()
    deps = AgentDependencies(search_document_fn=lambda query: query)

    monkeypatch.setattr(team_tools, "get_team_runtime", lambda: runtime)
    team_tools.set_session_runtime_context(
        "thread-from-config",
        default_model="fake-llm",
        dependencies=deps,
    )

    result = team_tools.spawn_agent.invoke(
        {"name": "researcher", "role": "teammate", "system_prompt": "WORKER_OVERRIDE"},
        config={"configurable": {"thread_id": "thread-from-config"}},
    )

    assert result == "Agent spawned: agent-123 (role=teammate, profile=paper_worker)"
    assert runtime.captured["name"] == "researcher"
    assert runtime.captured["model"] == "fake-llm"
    assert runtime.captured["profile"] is paper_worker_profile
    assert runtime.captured["deps"] is deps


def test_list_agents_uses_config_thread_runtime(monkeypatch):
    thread_one_runtime = _FakeRuntime()
    thread_two_runtime = _FakeRuntime()
    thread_two_runtime.agents_payload = [{"agent_id": "agent-2", "name": "reviewer"}]

    team_tools.set_current_session("thread-1")
    monkeypatch.setitem(team_tools._team_runtimes, "thread-1", thread_one_runtime)
    monkeypatch.setitem(team_tools._team_runtimes, "thread-2", thread_two_runtime)

    result = team_tools.list_agents.invoke(
        {},
        config={"configurable": {"thread_id": "thread-2"}},
    )

    assert '"agent_id": "agent-2"' in result
    assert '"name": "reviewer"' in result


def test_send_message_returns_busy_guidance(monkeypatch):
    runtime = _BusyRuntime()
    monkeypatch.setattr(team_tools, "get_team_runtime", lambda: runtime)
    team_tools.set_current_session("thread-1")

    result = team_tools.send_message.invoke(
        {"agent_id": "agent-1", "message": "继续执行"},
    )

    assert result == (
        "Error: Agent agent-1 is busy. Do not send another message yet; "
        "call get_agent_result later."
    )


def test_close_agent_returns_busy_guidance(monkeypatch):
    runtime = _BusyRuntime()
    monkeypatch.setattr(team_tools, "get_team_runtime", lambda: runtime)
    team_tools.set_current_session("thread-1")

    result = team_tools.close_agent.invoke({"agent_id": "agent-1"})

    assert result == (
        "Error: Cannot close agent agent-1: still busy. "
        "Wait for get_agent_result to return completed before closing."
    )
