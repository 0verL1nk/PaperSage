from agent.a2a_standard import (
    A2AInMemoryServer,
    TASK_STATE_CANCELED,
    TASK_STATE_WORKING,
    build_agent_card,
    build_coordinator_executor,
)


def test_build_agent_card_contains_core_fields():
    card = build_agent_card(
        name="Paper Agent",
        description="A2A test agent",
        url="http://localhost/a2a",
    )
    assert card["protocolVersion"] == "0.3.0"
    assert card["preferredTransport"] == "JSONRPC"
    assert isinstance(card["capabilities"], dict)
    assert isinstance(card["skills"], list) and card["skills"]


def test_message_send_and_tasks_get_roundtrip():
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="A2A test agent",
            url="http://localhost/a2a",
        ),
        execute_message_fn=lambda text: f"answer:{text}",
    )
    send_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "kind": "message",
                    "messageId": "m-1",
                }
            },
        }
    )
    task = send_resp["result"]
    task_id = task["id"]
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"][0]["text"] == "answer:hello"

    get_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tasks/get",
            "params": {"id": task_id, "historyLength": 1},
        }
    )
    assert get_resp["result"]["id"] == task_id
    assert len(get_resp["result"]["history"]) == 1


def test_tasks_cancel_moves_non_terminal_task_to_canceled():
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="A2A test agent",
            url="http://localhost/a2a",
        ),
        execute_message_fn=lambda text: text,
    )
    send_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "x"}],
                    "kind": "message",
                    "messageId": "m-1",
                }
            },
        }
    )
    task_id = send_resp["result"]["id"]

    # Force state to working for cancel path coverage.
    server.tasks[task_id].status["state"] = TASK_STATE_WORKING
    cancel_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        }
    )
    assert cancel_resp["result"]["status"]["state"] == TASK_STATE_CANCELED


def test_unknown_method_returns_jsonrpc_error():
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="A2A test agent",
            url="http://localhost/a2a",
        ),
        execute_message_fn=lambda text: text,
    )
    resp = server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": "x", "method": "unknown/method", "params": {}}
    )
    assert resp["error"]["code"] == -32601


def test_message_send_rejects_restarting_terminal_task():
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="A2A test agent",
            url="http://localhost/a2a",
        ),
        execute_message_fn=lambda text: text,
    )
    first = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "first"}],
                    "kind": "message",
                    "messageId": "m-1",
                }
            },
        }
    )
    task_id = first["result"]["id"]
    second = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "second"}],
                    "taskId": task_id,
                    "kind": "message",
                    "messageId": "m-2",
                }
            },
        }
    )
    assert second["error"]["code"] == -32010


def test_build_coordinator_executor_uses_run_with_mode():
    class _FakeCoordinator:
        def __init__(self):
            self.calls = []

        def run(self, question, workflow_mode):
            self.calls.append((question, workflow_mode))
            return "ok", []

    coordinator = _FakeCoordinator()
    executor = build_coordinator_executor(coordinator, workflow_mode="plan_act")
    result = executor("q")
    assert result == "ok"
    assert coordinator.calls == [("q", "plan_act")]
