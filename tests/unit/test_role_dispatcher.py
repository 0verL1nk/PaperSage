from agent.domain.orchestration import TeamRole
from agent.orchestration.langgraph_team_dag import build_ready_role_dispatches
from agent.orchestration.role_dispatcher import RoleDispatchPlan, build_role_dispatch_plan


def test_build_role_dispatch_plan_is_deterministic_and_typed() -> None:
    captured: dict[str, object] = {}

    def _fake_generate(prompt: str, llm: object, *, max_members: int) -> list[TeamRole]:
        captured["prompt"] = prompt
        captured["llm"] = llm
        captured["max_members"] = max_members
        return [
            TeamRole(name="writer", goal="summarize"),
            TeamRole(name="researcher", goal="collect evidence"),
        ]

    def _fake_normalize(raw_roles: list[TeamRole], *, max_members: int) -> list[TeamRole]:
        captured["normalized_from"] = list(raw_roles)
        captured["normalized_max_members"] = max_members
        return list(raw_roles)

    marker = object()
    plan = build_role_dispatch_plan(
        prompt="compare autonomous driving approaches",
        llm=marker,
        max_members=3,
        role_generator=_fake_generate,
        role_normalizer=_fake_normalize,
    )

    assert isinstance(plan, RoleDispatchPlan)
    assert [item.name for item in plan.roles] == ["writer", "researcher"]
    assert plan.role_order == {"writer": 0, "researcher": 1}
    assert plan.role_map["writer"].goal == "summarize"
    assert captured["prompt"] == "compare autonomous driving approaches"
    assert captured["llm"] is marker
    assert captured["max_members"] == 3
    assert captured["normalized_max_members"] == 3


def test_build_role_dispatch_plan_clamps_max_members_to_positive_value() -> None:
    captured: dict[str, int] = {}

    def _fake_generate(prompt: str, llm: object | None, *, max_members: int) -> list[TeamRole]:
        captured["generator"] = max_members
        assert prompt == "q"
        assert llm is None
        return [TeamRole(name="researcher", goal="collect")]

    def _fake_normalize(raw_roles: list[TeamRole], *, max_members: int) -> list[TeamRole]:
        captured["normalizer"] = max_members
        return raw_roles

    plan = build_role_dispatch_plan(
        prompt="q",
        llm=None,
        max_members=0,
        role_generator=_fake_generate,
        role_normalizer=_fake_normalize,
    )

    assert [item.name for item in plan.roles] == ["researcher"]
    assert captured == {"generator": 1, "normalizer": 1}


def test_build_ready_role_dispatches_integrates_role_plan_with_dag_ordering() -> None:
    plan = RoleDispatchPlan.from_roles(
        [
            TeamRole(name="researcher", goal="collect evidence"),
            TeamRole(name="writer", goal="prepare final answer"),
        ]
    )
    todo_records = [
        {
            "id": "t_writer",
            "status": "todo",
            "dependencies": [],
            "round": 1,
            "assignee": "writer",
        },
        {
            "id": "t_researcher",
            "status": "todo",
            "dependencies": [],
            "round": 1,
            "assignee": "researcher",
        },
    ]

    ready_dispatches = build_ready_role_dispatches(todo_records, role_plan=plan)

    assert [item.arg for item in ready_dispatches] == [
        {
            "todo_id": "t_researcher",
            "assignee": "researcher",
            "role_goal": "collect evidence",
        },
        {
            "todo_id": "t_writer",
            "assignee": "writer",
            "role_goal": "prepare final answer",
        },
    ]


def test_build_ready_role_dispatches_keeps_dispatch_when_role_not_found() -> None:
    plan = RoleDispatchPlan.from_roles([TeamRole(name="researcher", goal="collect evidence")])
    todo_records = [
        {
            "id": "t_missing_role",
            "status": "todo",
            "dependencies": [],
            "round": 1,
            "assignee": "reviewer",
        }
    ]

    ready_dispatches = build_ready_role_dispatches(todo_records, role_plan=plan)

    assert [item.arg for item in ready_dispatches] == [
        {
            "todo_id": "t_missing_role",
            "assignee": "reviewer",
            "role_goal": "",
        }
    ]
