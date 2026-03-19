from agent.middlewares.todolist import Todo, write_todos


def test_todo_model_supports_scheduler_fields() -> None:
    todo = Todo(
        id="todo-1",
        content="检索证据",
        status="ready",
        depends_on=[],
        assignee="teammate",
        execution_backend="a2a",
        retry_count=1,
        result={"summary": "done"},
        artifact_ref="artifact-1",
        error="",
    )

    payload = todo.model_dump()

    assert payload["assignee"] == "teammate"
    assert payload["execution_backend"] == "a2a"
    assert payload["retry_count"] == 1
    assert payload["result"] == {"summary": "done"}


def test_write_todos_preserves_scheduler_fields_in_state_update() -> None:
    todo = Todo(
        id="todo-1",
        content="检索证据",
        status="ready",
        depends_on=[],
        assignee="teammate",
        execution_backend="local",
        artifact_ref="artifact-1",
    )

    command = write_todos.func([todo], "tool-call-1")
    update = command.update or {}
    updated_todos = update["todos"]

    assert updated_todos[0].assignee == "teammate"
    assert updated_todos[0].execution_backend == "local"
    assert updated_todos[0].artifact_ref == "artifact-1"
