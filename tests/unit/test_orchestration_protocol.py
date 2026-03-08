import json

from agent.domain.orchestration import (
    PolicyDecision,
    TeamExecution,
    TeamRole,
    build_trace_event,
    create_trace_context,
)
from agent.orchestration.orchestrator import execute_orchestrated_turn
from agent.orchestration.team_runtime import run_team_tasks


class _FakeLeaderAgent:
    def invoke(self, _payload, config=None):
        assert isinstance(config, dict)
        return {"messages": [{"role": "assistant", "content": "ok"}]}


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


def test_execute_orchestrated_turn_emits_protocolized_trace():
    result = execute_orchestrated_turn(
        prompt="请回答",
        hinted_prompt="请回答",
        leader_agent=_FakeLeaderAgent(),
        leader_runtime_config={},
        llm=None,
        force_plan=False,
        force_team=False,
    )

    assert result.trace_payload
    sequences = [int(item.get("sequence", 0)) for item in result.trace_payload]
    assert sequences == sorted(sequences)
    assert sequences[0] == 1
    for item in result.trace_payload:
        assert item.get("run_id")
        assert item.get("task_id")
        assert item.get("span_id")
        assert "timestamp" in item
        assert item.get("channel") == "internal.orchestrator"
        assert isinstance(item.get("a2aMessage"), dict)
        assert item["a2aMessage"]["messageId"] == item["span_id"]


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
        lambda **_kwargs: "done",
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
    assert len(events) == 3
    assert events[0]["performative"] == "plan_todo"
    assert events[1]["performative"] == "dispatch"
    assert events[1]["meta"]["round"] == 1
    assert events[1]["meta"]["role"] == "leader_role"
    assert events[2]["performative"] == "review"
    assert events[2]["meta"]["round"] == 1


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
    from agent.orchestration.team_runtime import _build_team_todo_records_mechanical, TeamRole as TR
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._leader_plan_todo",
        lambda *, roles, actual_rounds, plan_id, **_kw: _build_team_todo_records_mechanical(
            roles=roles, rounds=actual_rounds, plan_id=plan_id
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **kwargs: f"{kwargs['role'].name}-done",
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
    reject_events = [e for e in execution.trace_events if e.get("performative") == "plan_todo_reject"]
    assert len(plan_todo_events) >= 1   # 至少有首次 leader 规划事件
    assert len(reject_events) >= 1      # 至少有一次系统反馈环事件
    # reject 事件的 sender/receiver 符合路由规则
    for ev in reject_events:
        assert ev.get("sender") == "system"
        assert ev.get("receiver") == "leader"
    non_plan_events = [
        e for e in execution.trace_events
        if e.get("performative") not in {"plan_todo", "plan_todo_reject"}
    ]
    assert non_plan_events == []


def test_execute_orchestrated_turn_syncs_team_todo_store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_TOOLS_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_TODO_FILE", ".agent/todo.json")
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.intercept_policy",
        lambda *_args, **_kwargs: PolicyDecision(
            plan_enabled=False,
            team_enabled=True,
            reason="forced",
            confidence=1.0,
            source="test",
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.run_team_tasks",
        lambda **_kwargs: TeamExecution(
            enabled=True,
            roles=["researcher"],
            member_count=1,
            rounds=1,
            summary="ok",
            todo_records=[
                {
                    "id": "team_r1_researcher",
                    "title": "[Round 1] researcher",
                    "status": "done",
                    "assignee": "researcher",
                    "dependencies": [],
                    "history": [],
                }
            ],
            todo_stats={"done": 1},
            trace_events=[],
        ),
    )

    execute_orchestrated_turn(
        prompt="请回答",
        hinted_prompt="请回答",
        leader_agent=_FakeLeaderAgent(),
        leader_runtime_config={},
        llm=object(),
    )

    todo_path = tmp_path / ".agent" / "todo.json"
    assert todo_path.exists()
    payload = json.loads(todo_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["id"] == "team_r1_researcher"


def test_persist_team_todo_records_replaces_same_plan_id(tmp_path):
    """同 plan_id 的记录应被全量替换，不同 plan_id 的记录应保留。"""
    from agent.orchestration.orchestrator import _persist_team_todo_records
    from agent.domain.orchestration import TeamExecution
    import json, os

    todo_file = tmp_path / ".agent" / "todo.json"
    os.environ["AGENT_FILE_TOOLS_ROOT"] = str(tmp_path)
    os.environ["AGENT_TODO_FILE"] = ".agent/todo.json"

    # 先写入两个不同 plan_id 的记录
    todo_file.parent.mkdir(parents=True, exist_ok=True)
    todo_file.write_text(json.dumps([
        {"id": "old_t1", "plan_id": "team:plan-A", "status": "done", "title": "old A"},
        {"id": "old_t2", "plan_id": "team:plan-B", "status": "todo", "title": "old B"},
    ], ensure_ascii=False), encoding="utf-8")

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
