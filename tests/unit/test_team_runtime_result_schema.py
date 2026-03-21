import json
from pathlib import Path
from types import SimpleNamespace

from agent.team.runtime import AgentInstance, AgentState, TeamRuntime


class _FakeAgent:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def invoke(self, payload, config=None):
        self.calls.append({"payload": payload, "config": config})
        return {"messages": [SimpleNamespace(content=self.content)]}



def test_team_runtime_get_agent_result_returns_structured_payload(tmp_path: Path) -> None:
    runtime = TeamRuntime("team-1", result_dir=tmp_path)
    agent = _FakeAgent("结论 <evidence>chunk-1|p1|o0-10</evidence>")
    instance = AgentInstance(
        agent_id="agent-1",
        name="researcher",
        model="fake-llm",
        system_prompt="WORKER",
        tools=[],
        agent=agent,
        state=AgentState.IDLE,
        result_file=tmp_path / "agent-1.result.txt",
        profile_name="paper_worker",
        thread_id="team:team-1:agent-1",
    )
    runtime.agents["agent-1"] = instance

    runtime._execute_agent(instance, "执行检索")
    result = json.loads(runtime.get_agent_result("agent-1"))

    assert result["kind"] == "agent_result_v1"
    assert result["agent_id"] == "agent-1"
    assert result["profile_name"] == "paper_worker"
    assert result["status"] == "completed"
    assert result["summary"].startswith("结论")
    assert result["output"] == "结论 <evidence>chunk-1|p1|o0-10</evidence>"
    assert result["evidence"] == ["chunk-1|p1|o0-10"]
    assert result["risks"] == []
    assert result["artifacts"] == []
    assert agent.calls[0]["config"] == {"configurable": {"thread_id": "team:team-1:agent-1"}}

    runtime.cleanup()



def test_team_runtime_get_agent_result_returns_structured_busy_payload(tmp_path: Path) -> None:
    runtime = TeamRuntime("team-1", result_dir=tmp_path)
    runtime.agents["agent-1"] = AgentInstance(
        agent_id="agent-1",
        name="researcher",
        model="fake-llm",
        system_prompt="WORKER",
        tools=[],
        agent=object(),
        state=AgentState.BUSY,
        result_file=tmp_path / "agent-1.result.txt",
        profile_name="paper_worker",
        thread_id="team:team-1:agent-1",
    )

    result = json.loads(runtime.get_agent_result("agent-1"))

    assert result["status"] == "busy"
    assert result["output"] == ""
    assert result["summary"] == (
        "Agent agent-1 is still busy. "
        "Do not send another message or close it yet; call get_agent_result later."
    )
    assert result["error"] == result["summary"]
    assert result["retry_after_ms"] == 5000
    assert result["next_action"]["type"] == "wait_and_retry"
    assert result["next_action"]["tool"] == "get_agent_result"
    assert "send_message" in result["next_action"]["avoid"]
    assert "close_agent" in result["next_action"]["avoid"]

    runtime.cleanup()
