from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
import pytest

from agent import multi_agent_a2a as module


class _FakeAgent:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        content = self._responses.pop(0) if self._responses else ""
        return {"messages": [SimpleNamespace(type="ai", content=content)]}


def test_run_react_mode_returns_direct_answer():
    react = _FakeAgent(["react-answer"])
    planner = _FakeAgent([])
    researcher = _FakeAgent([])
    reviewer = _FakeAgent([])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s1",
    )

    answer, trace = coordinator.run("q", workflow_mode=module.WORKFLOW_REACT)

    assert answer == "react-answer"
    assert [item.performative for item in trace] == ["request", "final"]
    assert react.calls[0][1]["configurable"]["thread_id"] == "s1:react"


def test_run_plan_act_mode_skips_review():
    react = _FakeAgent([])
    planner = _FakeAgent(["plan-1"])
    researcher = _FakeAgent(["draft-1"])
    reviewer = _FakeAgent([])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s2",
    )

    answer, trace = coordinator.run("q", workflow_mode=module.WORKFLOW_PLAN_ACT)

    assert answer == "draft-1"
    assert [item.performative for item in trace] == ["request", "plan", "final"]
    assert len(reviewer.calls) == 0


def test_run_plan_act_replan_with_revision():
    react = _FakeAgent([])
    planner = _FakeAgent(["plan-1", "plan-2"])
    researcher = _FakeAgent(["draft-1", "draft-2"])
    reviewer = _FakeAgent(["Decision: REVISE\nFeedback: add evidence"])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s3",
    )

    answer, trace = coordinator.run("q", workflow_mode=module.WORKFLOW_PLAN_ACT_REPLAN)

    assert answer == "draft-2"
    performatives = [item.performative for item in trace]
    assert performatives[:5] == ["request", "plan", "draft", "review", "replan"]
    assert performatives[-1] == "final"
    assert performatives.count("draft") >= 2
    assert performatives.count("review") >= 2


def test_run_plan_act_replan_passes_without_replan():
    react = _FakeAgent([])
    planner = _FakeAgent(["plan-1"])
    researcher = _FakeAgent(["draft-1"])
    reviewer = _FakeAgent(["Decision: PASS\nFeedback: good"])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s4",
    )

    answer, trace = coordinator.run("q", workflow_mode=module.WORKFLOW_PLAN_ACT_REPLAN)

    assert answer == "draft-1"
    assert [item.performative for item in trace] == [
        "request",
        "plan",
        "draft",
        "review",
        "final",
    ]


def test_run_accepts_structured_plan_and_review_json():
    react = _FakeAgent([])
    planner = _FakeAgent(['{"steps":["collect evidence","answer concisely"],"goal":"qa"}'])
    researcher = _FakeAgent(["draft-1"])
    reviewer = _FakeAgent(['{"decision":"PASS","feedback":"grounded and complete"}'])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s4-json",
    )

    answer, trace = coordinator.run("q", workflow_mode=module.WORKFLOW_PLAN_ACT_REPLAN)

    assert answer == "draft-1"
    assert trace[1].performative == "plan"
    assert trace[1].content.startswith("1. collect evidence")
    assert trace[3].performative == "review"
    assert "Decision: PASS" in trace[3].content


def test_run_defaults_to_revise_when_reviewer_output_is_unstructured():
    react = _FakeAgent([])
    planner = _FakeAgent(["plan-1", "plan-2"])
    researcher = _FakeAgent(["draft-1", "draft-2"])
    reviewer = _FakeAgent(["looks good"])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s4-unsafe",
    )

    answer, trace = coordinator.run("q", workflow_mode=module.WORKFLOW_PLAN_ACT_REPLAN)

    assert answer == "draft-2"
    performatives = [item.performative for item in trace]
    assert "replan" in performatives


def test_run_uses_default_plan_when_planner_returns_invalid_json():
    react = _FakeAgent([])
    planner = _FakeAgent(['{"steps":["only one"]}'])
    researcher = _FakeAgent(["draft-1"])
    reviewer = _FakeAgent([])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s4-plan-fallback",
    )

    answer, trace = coordinator.run("q", workflow_mode=module.WORKFLOW_PLAN_ACT)

    assert answer == "draft-1"
    assert trace[1].performative == "plan"
    assert "Retrieve evidence from document" in trace[1].content


def test_run_emits_dispatch_events_via_callback():
    react = _FakeAgent([])
    planner = _FakeAgent(["plan-1"])
    researcher = _FakeAgent(["draft-1"])
    reviewer = _FakeAgent(["Decision: PASS\nFeedback: good"])
    coordinator = module.A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="s5",
    )
    callbacks = []

    coordinator.run(
        "q",
        workflow_mode=module.WORKFLOW_PLAN_ACT_REPLAN,
        on_event=lambda item: callbacks.append(item.performative),
    )

    assert "dispatch" in callbacks
    assert callbacks.count("dispatch") >= 3


def test_create_multi_agent_a2a_session_builds_four_agents(monkeypatch):
    captured = []

    class _StubAgent:
        def invoke(self, *_args, **_kwargs):
            return {"messages": [SimpleNamespace(type="ai", content="ok")]}

    def fake_create_agent(*, model, tools, system_prompt, checkpointer):
        captured.append(
            {
                "model": model,
                "tools": tools,
                "system_prompt": system_prompt,
                "checkpointer": checkpointer,
            }
        )
        return _StubAgent()

    monkeypatch.setattr(module, "create_agent", fake_create_agent)
    monkeypatch.setattr(
        module,
        "build_agent_tools",
        lambda _search_document_fn, search_document_evidence_fn=None, allowed_tools=None: [
            f"tool:{name}" for name in sorted(allowed_tools or [])
        ],
    )

    session = module.create_multi_agent_a2a_session(
        llm="fake-llm", search_document_fn=lambda q: q
    )

    assert session.session_id.startswith("a2a-")
    assert len(captured) == 4
    assert captured[0]["tools"] == sorted(f"tool:{name}" for name in module.REACT_ALLOWED_TOOLS)  # react
    assert captured[2]["tools"] == sorted(
        f"tool:{name}" for name in module.RESEARCHER_ALLOWED_TOOLS
    )  # researcher
    assert isinstance(captured[0]["checkpointer"], InMemorySaver)


def test_state_transition_rejects_invalid_edge():
    with pytest.raises(ValueError):
        module.A2AMultiAgentCoordinator._transition_state(
            module.STATE_SUBMITTED,
            module.STATE_REVIEWING,
        )
