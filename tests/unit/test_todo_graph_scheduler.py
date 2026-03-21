from agent.domain.todo_graph import TodoGraph
from agent.middlewares.todolist import Todo
from agent.orchestration.todo_scheduler import LeaderTodoScheduler


def test_todo_graph_returns_ready_and_blocked_todos() -> None:
    todos = [
        {"id": "1", "content": "Task 1", "status": "completed", "depends_on": []},
        {"id": "2", "content": "Task 2", "status": "pending", "depends_on": ["1"]},
        {"id": "3", "content": "Task 3", "status": "pending", "depends_on": ["4"]},
        {"id": "4", "content": "Task 4", "status": "failed", "depends_on": []},
        {"id": "5", "content": "Task 5", "status": "canceled", "depends_on": []},
        {"id": "6", "content": "Task 6", "status": "pending", "depends_on": ["5"]},
    ]

    graph = TodoGraph(todos)

    ready_ids = [todo["id"] for todo in graph.get_ready_todos()]
    blocked_ids = [todo["id"] for todo in graph.get_blocked_todos()]

    assert ready_ids == ["2"]
    assert sorted(blocked_ids) == ["3", "6"]


def test_scheduler_selects_ready_todos_deterministically() -> None:
    scheduler = LeaderTodoScheduler()
    todos = [
        Todo(id="1", content="Task 1", status="completed", depends_on=[]),
        Todo(
            id="2",
            content="Task 2",
            status="pending",
            depends_on=["1"],
            assignee="teammate",
            execution_backend="local",
        ),
        Todo(
            id="3",
            content="Task 3",
            status="pending",
            depends_on=[],
            assignee="teammate",
            execution_backend="a2a",
        ),
    ]

    selected = scheduler.select_ready_todos(todos)

    assert [todo.id for todo in selected] == ["2", "3"]
    assert [todo.execution_backend for todo in selected] == ["local", "a2a"]


def test_scheduler_refreshes_pending_todos_to_ready_or_blocked() -> None:
    scheduler = LeaderTodoScheduler()
    todos = [
        Todo(id="1", content="Task 1", status="completed", depends_on=[]),
        Todo(id="2", content="Task 2", status="pending", depends_on=["1"]),
        Todo(id="3", content="Task 3", status="pending", depends_on=["4"]),
        Todo(id="4", content="Task 4", status="failed", depends_on=[]),
    ]

    refreshed = scheduler.refresh_todo_states(todos)

    refreshed_map = {todo.id: todo.status for todo in refreshed}
    assert refreshed_map["2"] == "ready"
    assert refreshed_map["3"] == "blocked"
