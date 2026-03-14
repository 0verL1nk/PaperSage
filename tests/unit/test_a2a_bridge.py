from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

from agent.a2a import WORKFLOW_PLAN_ACT, A2AMessage
from agent.orchestration.a2a_bridge import (
    DEFAULT_MAX_REPLAN_ROUNDS,
    A2ABridge,
)
from agent.orchestration.langgraph_team_dag import execute_team_task_via_a2a_bridge


@dataclass(frozen=True)
class _FakeSession:
    coordinator: Any
    session_id: str


class _FakeCoordinator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        question: str,
        *,
        workflow_mode: str,
        on_event: Any = None,
        max_replan_rounds: int,
    ) -> tuple[str, list[A2AMessage]]:
        self.calls.append(
            {
                "question": question,
                "workflow_mode": workflow_mode,
                "on_event": on_event,
                "max_replan_rounds": max_replan_rounds,
            }
        )
        return (
            "final answer",
            [
                A2AMessage(
                    sender="researcher",
                    receiver="user",
                    performative="final",
                    content="final answer",
                )
            ],
        )


class _FakeBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_with_session(
        self,
        session: Any,
        orchestration_input: dict[str, Any],
        *,
        on_event: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "session": session,
                "orchestration_input": orchestration_input,
                "on_event": on_event,
            }
        )
        return {"status": "completed", "final_answer": "ok"}


def test_orchestration_to_a2a_input_defaults() -> None:
    bridge = A2ABridge()

    invocation = bridge.orchestration_to_a2a_input({"prompt": "explain this paper"})

    assert invocation.question == "explain this paper"
    assert invocation.workflow_mode == WORKFLOW_PLAN_ACT
    assert invocation.max_replan_rounds == DEFAULT_MAX_REPLAN_ROUNDS


def test_run_with_session_maps_inputs_and_outputs() -> None:
    coordinator = _FakeCoordinator()
    session = _FakeSession(coordinator=coordinator, session_id="sess-1")
    bridge = A2ABridge()

    output = bridge.run_with_session(
        cast(Any, session),
        {
            "user_prompt": "compare methods",
            "workflow_mode": "plan_act",
            "max_replan_rounds": 4,
        },
    )

    assert coordinator.calls == [
        {
            "question": "compare methods",
            "workflow_mode": "plan_act",
            "on_event": None,
            "max_replan_rounds": 4,
        }
    ]
    assert output["status"] == "completed"
    assert output["final_answer"] == "final answer"
    assert output["trace"]
    assert output["session_id"] == "sess-1"
    assert output["workflow_mode"] == "plan_act"


def test_run_with_session_canonicalizes_legacy_plan_act_replan_mode() -> None:
    coordinator = _FakeCoordinator()
    session = _FakeSession(coordinator=coordinator, session_id="sess-legacy")
    bridge = A2ABridge()

    output = bridge.run_with_session(
        cast(Any, session),
        {
            "user_prompt": "compare methods",
            "workflow_mode": "plan_act_replan",
            "max_replan_rounds": 2,
        },
    )

    assert coordinator.calls[0]["workflow_mode"] == "plan_act"
    assert output["workflow_mode"] == "plan_act"


def test_execute_team_task_via_a2a_bridge_delegates_call() -> None:
    fake_bridge = _FakeBridge()
    session = object()
    event_callback: Callable[[A2AMessage], None] = lambda _msg: None

    result = execute_team_task_via_a2a_bridge(
        {"question": "hello"},
        session=cast(Any, session),
        bridge=cast(Any, fake_bridge),
        on_event=event_callback,
    )

    assert result == {"status": "completed", "final_answer": "ok"}
    assert fake_bridge.calls == [
        {
            "session": session,
            "orchestration_input": {"question": "hello"},
            "on_event": event_callback,
        }
    ]
