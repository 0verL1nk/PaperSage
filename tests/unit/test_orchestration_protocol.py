import json
import os

from agent.domain.orchestration import (
    ExecutionPlan,
    PlanStep,
    PolicyDecision,
    TeamExecution,
    TeamRole,
    build_trace_event,
    create_trace_context,
)
from agent.orchestration.orchestrator import (
    _persist_team_todo_records,
    execute_orchestrated_turn,
)
from agent.orchestration.team_runtime import (
    TeamTaskAttemptResult,
    _build_team_todo_records_mechanical,
    run_team_tasks,
)


class _FakeLeaderAgent:
    def invoke(self, _payload, config=None):
        assert isinstance(config, dict)
        return {"messages": [{"role": "assistant", "content": "ok"}]}


class _SequencedLeaderAgent:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def invoke(self, payload, config=None):
        self.calls.append({"payload": payload, "config": config})
        response = self._responses.pop(0) if self._responses else "ok"
        if isinstance(response, dict):
            return response
        return {"messages": [{"role": "assistant", "content": response}]}


def _team_done_output(label: str) -> str:
    return f"[结论]\n{label}\n\n[证据]\nsearch evidence [chunk_1]\n\n[待验证点]\nnone"


def _start_plan_result(goal: str = "拆步骤回答") -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "start_plan", "args": {"goal": goal, "reason": "multi-step"}}
                ],
            },
            {
                "role": "tool",
                "name": "start_plan",
                "content": f'{{"type":"mode_activate","mode":"plan","goal":"{goal}"}}',
            },
        ]
    }


def _start_team_result(goal: str = "交叉验证") -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "start_team", "args": {"goal": goal, "reason": "need team"}}
                ],
            },
            {
                "role": "tool",
                "name": "start_team",
                "content": f'{{"type":"mode_activate","mode":"team","goal":"{goal}"}}',
            },
        ]
    }


def test_build_trace_event_contains_required_envelope_fields():
    context = create_trace_context(
        run_id="run-test",
        task_id="task-test",
        channel="internal.test",
    )
    event = build_trace_event(
        context=context,
        sender="user",
        receiver="leader",
        performative="request",
        content="hello",
    )

    assert event["run_id"] == "run-test"
    assert event["task_id"] == "task-test"
    assert event["channel"] == "internal.test"
    assert event["sequence"] == 1
    assert event["span_id"] == "run-test:span:1"
    assert event["parent_span_id"] == ""
    assert event["timestamp"]
    assert isinstance(event.get("a2aMessage"), dict)
    assert event["a2aMessage"]["taskId"] == "task-test"
    assert event["a2aMessage"]["contextId"] == "run-test"
    assert event["a2aMessage"]["metadata"]["performative"] == "request"


def test_build_trace_event_rejects_invalid_route():
    context = create_trace_context(run_id="run-invalid", task_id="task-invalid")
    try:
        build_trace_event(
            context=context,
            sender="leader",
            receiver="leader",
            performative="dispatch",
            content="invalid",
        )
    except ValueError as exc:
        assert "Invalid trace route" in str(exc)
    else:
        raise AssertionError("Expected invalid dispatch route to raise ValueError")


def test_build_trace_event_accepts_policy_switch_route():
    context = create_trace_context(run_id="run-switch", task_id="task-switch")
    event = build_trace_event(
        context=context,
        sender="policy_engine",
        receiver="leader",
        performative="policy_switch",
        content="async_switch@pre_plan",
    )

    assert event["performative"] == "policy_switch"
    assert event["receiver"] == "leader"


def test_run_team_tasks_normalizes_role_name_and_emits_round_metadata(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [TeamRole(name="Leader", goal="collect evidence")],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **_kwargs: _team_done_output("done"),
    )
    events = []

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=1,
        max_rounds=1,
        on_event=lambda event: events.append(event),
    )

    assert execution.enabled is True
    assert execution.roles == ["leader_role"]
    assert len(events) == 5
    assert events[0]["performative"] == "plan_todo"
    assert events[1]["performative"] == "dispatch"
    assert events[1]["meta"]["round"] == 1
    assert events[1]["meta"]["role"] == "leader_role"
    assert events[2]["performative"] == "member_request"
    assert events[3]["performative"] == "member_output"
    assert events[-1]["performative"] == "leader_decision"
    assert events[-1]["meta"]["round"] == 1


def test_run_team_tasks_uses_todo_dependencies_for_ready_queue(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [
            TeamRole(name="researcher", goal="collect evidence"),
            TeamRole(name="reviewer", goal="cross check"),
        ],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 2,
    )
    # patch _leader_plan_todo 返回与机械生成完全一致的结构，以便断言 todo_id 顺序
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._leader_plan_todo",
        lambda *, roles, actual_rounds, plan_id, **_kw: _build_team_todo_records_mechanical(
            roles=roles, rounds=actual_rounds, plan_id=plan_id
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **kwargs: _team_done_output(f"{kwargs['role'].name}-done"),
    )
    events: list[dict] = []

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=2,
        max_rounds=2,
        on_event=lambda event: events.append(event),
    )

    assert execution.enabled is True
    assert execution.roles == ["researcher", "reviewer"]
    assert execution.rounds == 2
    assert len(execution.todo_records) == 4
    assert execution.todo_stats["done"] == 4
    assert execution.todo_stats["blocked"] == 0

    dispatch_events = [e for e in events if e["performative"] == "dispatch"]
    dispatch_todo_ids = [str(e.get("meta", {}).get("todo_id") or "") for e in dispatch_events]
    assert dispatch_todo_ids == [
        "team_r1_researcher",
        "team_r1_reviewer",
        "team_r2_researcher",
        "team_r2_reviewer",
    ]

    todo_map = {str(item["id"]): item for item in execution.todo_records}
    assert todo_map["team_r1_researcher"]["dependencies"] == []
    assert todo_map["team_r1_reviewer"]["dependencies"] == ["team_r1_researcher"]
    assert todo_map["team_r2_researcher"]["dependencies"] == ["team_r1_researcher"]
    assert todo_map["team_r2_reviewer"]["dependencies"] == [
        "team_r1_reviewer",
        "team_r2_researcher",
    ]
    decision_events = [e for e in events if e["performative"] == "leader_decision"]
    assert len(decision_events) == 4
    assert all(e["meta"]["verification_status"] == "passed" for e in decision_events)
    assert all(e["meta"]["decision"] == "accept" for e in decision_events)


def test_run_team_tasks_retries_failed_member_task_once(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [TeamRole(name="researcher", goal="collect evidence")],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 1,
    )
    responses = iter(
        [
            TeamTaskAttemptResult(
                answer="[结论]\nfirst try\n\n[证据]\nnone\n\n[待验证点]\nmissing",
                tool_names=[],
                tool_trace_events=[],
                skill_activation_events=[],
                tool_activation_events=[],
            ),
            TeamTaskAttemptResult(
                answer=_team_done_output("second try"),
                tool_names=["search_document"],
                tool_trace_events=[
                    {
                        "sender": "leader",
                        "receiver": "search_document",
                        "performative": "tool_call",
                        "content": "{'query': 'q'}",
                    },
                    {
                        "sender": "search_document",
                        "receiver": "leader",
                        "performative": "tool_result",
                        "content": "evidence [chunk_1]",
                    },
                ],
                skill_activation_events=[],
                tool_activation_events=[],
            ),
        ]
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **_kwargs: next(responses),
    )

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=1,
        max_rounds=1,
    )

    assert execution.enabled is True
    assert execution.todo_stats["done"] == 1
    assert execution.todo_stats["blocked"] == 0
    performatives = [str(item.get("performative") or "") for item in execution.trace_events]
    assert "task_retry" not in performatives
    assert performatives.count("member_request") == 2
    decision_events = [
        item for item in execution.trace_events if item.get("performative") == "leader_decision"
    ]
    assert len(decision_events) == 2
    assert decision_events[0]["meta"]["verification_status"] == "failed"
    assert decision_events[0]["meta"]["decision"] == "retry"
    assert decision_events[-1]["meta"]["verification_status"] == "passed"
    assert decision_events[-1]["meta"]["decision"] == "accept"
    assert "tool_call" in performatives
    assert "tool_result" in performatives


def test_run_team_tasks_marks_block_as_leader_decision(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [TeamRole(name="researcher", goal="collect evidence")],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **_kwargs: TeamTaskAttemptResult(
            answer="[结论]\nfirst try\n\n[证据]\nnone\n\n[待验证点]\nmissing",
            tool_names=[],
            tool_trace_events=[],
            skill_activation_events=[],
            tool_activation_events=[],
        ),
    )

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=1,
        max_rounds=1,
    )

    assert execution.enabled is True
    assert execution.todo_stats["blocked"] == 1
    decision_events = [
        item for item in execution.trace_events if item.get("performative") == "leader_decision"
    ]
    assert len(decision_events) == 2
    assert decision_events[-1]["meta"]["decision"] == "block"
    assert decision_events[-1]["meta"]["verification_status"] == "failed"


def test_run_team_tasks_blocks_when_dependency_cycle_detected(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [TeamRole(name="researcher", goal="collect evidence")],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 1,
    )
    # 直接 patch _leader_plan_todo 注入一个有循环依赖的 todo 列表
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._leader_plan_todo",
        lambda **_kwargs: [
            {
                "id": "task_a",
                "title": "a",
                "status": "todo",
                "assignee": "researcher",
                "dependencies": ["task_b"],
                "history": [],
                "round": 1,
                "details": "do a",
            },
            {
                "id": "task_b",
                "title": "b",
                "status": "todo",
                "assignee": "researcher",
                "dependencies": ["task_a"],
                "history": [],
                "round": 1,
                "details": "do b",
            },
        ],
    )

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=1,
        max_rounds=1,
    )

    assert execution.enabled is True
    assert execution.fallback_reason == "todo_dependency_cycle"
    assert execution.todo_stats["blocked"] == 2
    # plan_todo: leader 每次规划都产生一个（共 max_todo_plan_retries+1 次，最后一次超限后返回）
    # plan_todo_reject: 程序检测到环后反馈给 leader（共 max_todo_plan_retries 次）
    plan_todo_events = [e for e in execution.trace_events if e.get("performative") == "plan_todo"]
    reject_events = [
        e for e in execution.trace_events if e.get("performative") == "plan_todo_reject"
    ]
    assert len(plan_todo_events) >= 1  # 至少有首次 leader 规划事件
    assert len(reject_events) >= 1  # 至少有一次系统反馈环事件
    # reject 事件的 sender/receiver 符合路由规则
    for ev in reject_events:
        assert ev.get("sender") == "system"
        assert ev.get("receiver") == "leader"
    non_plan_events = [
        e
        for e in execution.trace_events
        if e.get("performative") not in {"plan_todo", "plan_todo_reject"}
    ]
    assert non_plan_events == []

def test_persist_team_todo_records_replaces_same_plan_id(tmp_path):
    """同 plan_id 的记录应被全量替换，不同 plan_id 的记录应保留。"""

    todo_file = tmp_path / ".agent" / "todo.json"
    os.environ["AGENT_FILE_TOOLS_ROOT"] = str(tmp_path)
    os.environ["AGENT_TODO_FILE"] = ".agent/todo.json"

    # 先写入两个不同 plan_id 的记录
    todo_file.parent.mkdir(parents=True, exist_ok=True)
    todo_file.write_text(
        json.dumps(
            [
                {"id": "old_t1", "plan_id": "team:plan-A", "status": "done", "title": "old A"},
                {"id": "old_t2", "plan_id": "team:plan-B", "status": "todo", "title": "old B"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 用 plan-A 的新记录来 persist（应替换 plan-A 的旧记录，保留 plan-B）
    te = TeamExecution(
        enabled=True,
        roles=["researcher"],
        member_count=1,
        rounds=1,
        summary="",
        todo_records=[
            {"id": "new_t1", "plan_id": "team:plan-A", "status": "done", "title": "new A"},
            {"id": "new_t2", "plan_id": "team:plan-A", "status": "done", "title": "new A2"},
        ],
        todo_stats={"done": 2},
        trace_events=[],
    )
    _persist_team_todo_records(te)

    result = json.loads(todo_file.read_text(encoding="utf-8"))
    ids = [r["id"] for r in result]
    # plan-B 的 old_t2 应保留，plan-A 的 old_t1 应被替换为 new_t1/new_t2
    assert "old_t2" in ids, "不同 plan_id 的记录应被保留"
    assert "old_t1" not in ids, "同 plan_id 的旧记录应被替换"
    assert "new_t1" in ids
    assert "new_t2" in ids
    assert len(result) == 3  # old_t2 + new_t1 + new_t2
