from agent.a2a.standard import (
    A2A_VERSION_HEADER,
    A2AInMemoryServer,
    METHOD_CANCEL_TASK,
    METHOD_GET_TASK,
    METHOD_SEND_MESSAGE,
    METHOD_SEND_STREAMING_MESSAGE,
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
    assert card["a2aVersion"] == "1.0"
    assert "1.0" in card["supportedA2AVersions"]
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


def test_v1_send_get_cancel_roundtrip():
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
            "method": METHOD_SEND_MESSAGE,
            "headers": {A2A_VERSION_HEADER: "1.0"},
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
    assert send_resp["meta"]["a2aVersion"] == "1.0"
    assert task["status"]["state"] == "completed"

    get_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "2",
            "method": METHOD_GET_TASK,
            "headers": {A2A_VERSION_HEADER: "1.0"},
            "params": {"id": task_id},
        }
    )
    assert get_resp["result"]["id"] == task_id

    server.tasks[task_id].status["state"] = TASK_STATE_WORKING
    cancel_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "3",
            "method": METHOD_CANCEL_TASK,
            "headers": {A2A_VERSION_HEADER: "1.0"},
            "params": {"id": task_id},
        }
    )
    assert cancel_resp["result"]["status"]["state"] == TASK_STATE_CANCELED


def test_send_streaming_message_returns_event_batches():
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="A2A test agent",
            url="http://localhost/a2a",
            supports_streaming=True,
        ),
        execute_message_fn=lambda text: f"answer:{text}",
    )
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "stream-1",
            "method": METHOD_SEND_STREAMING_MESSAGE,
            "headers": {A2A_VERSION_HEADER: "1.0"},
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "stream me"}],
                    "kind": "message",
                    "messageId": "m-stream",
                }
            },
        }
    )
    result = resp["result"]
    assert "task" in result
    assert result["task"]["status"]["state"] == "completed"
    events = result.get("events")
    assert isinstance(events, list) and len(events) >= 2
    assert events[-1]["event"] == "TaskCompleted"


def test_rejects_unsupported_a2a_version():
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="A2A test agent",
            url="http://localhost/a2a",
        ),
        execute_message_fn=lambda text: text,
    )
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "x",
            "method": METHOD_SEND_MESSAGE,
            "headers": {A2A_VERSION_HEADER: "9.9"},
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
    assert resp["error"]["code"] == -32602
    assert "supportedVersions" in resp["error"]["data"]


def test_handle_jsonrpc_stream_emits_incremental_frames():
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="A2A test agent",
            url="http://localhost/a2a",
            supports_streaming=True,
        ),
        execute_message_fn=lambda text: text,
        execute_message_stream_fn=lambda _text: ["A", "B"],
    )
    frames = list(
        server.handle_jsonrpc_stream(
            {
                "jsonrpc": "2.0",
                "id": "stream-incremental",
                "method": METHOD_SEND_STREAMING_MESSAGE,
                "headers": {A2A_VERSION_HEADER: "1.0"},
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "stream me"}],
                        "kind": "message",
                        "messageId": "m-stream-incremental",
                    }
                },
            }
        )
    )

    assert len(frames) >= 4
    assert all(frame.get("jsonrpc") == "2.0" for frame in frames)
    assert all(frame.get("meta", {}).get("stream") is True for frame in frames)

    sequence_values = [int(frame.get("meta", {}).get("sequence", 0)) for frame in frames]
    assert sequence_values == sorted(sequence_values)
    assert sequence_values[0] == 1

    events = [frame["result"]["event"] for frame in frames]
    assert events[0]["event"] == "TaskStatusUpdate"
    assert events[1]["event"] == "TaskArtifactUpdate"
    assert events[-1]["event"] == "TaskCompleted"
    assert events[-1]["final"] is True
