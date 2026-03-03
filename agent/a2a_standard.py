from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import uuid4

from .multi_agent_a2a import WORKFLOW_PLAN_ACT_REPLAN, A2AMultiAgentCoordinator


JSONRPC_VERSION = "2.0"
METHOD_MESSAGE_SEND = "message/send"
METHOD_TASKS_GET = "tasks/get"
METHOD_TASKS_CANCEL = "tasks/cancel"
METHOD_AGENT_GET_CARD = "agent/getAuthenticatedExtendedCard"

TASK_STATE_SUBMITTED = "submitted"
TASK_STATE_WORKING = "working"
TASK_STATE_INPUT_REQUIRED = "input-required"
TASK_STATE_COMPLETED = "completed"
TASK_STATE_CANCELED = "canceled"
TASK_STATE_FAILED = "failed"
TASK_STATE_REJECTED = "rejected"
TASK_STATE_AUTH_REQUIRED = "auth-required"
TASK_STATE_UNKNOWN = "unknown"

TERMINAL_STATES = {
    TASK_STATE_COMPLETED,
    TASK_STATE_CANCELED,
    TASK_STATE_FAILED,
    TASK_STATE_REJECTED,
}

# JSON-RPC standard error codes
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603

# A2A-compatible server error range
ERR_TASK_NOT_FOUND = -32004
ERR_TASK_TERMINAL = -32010


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    version: str = "1.0.0",
    protocol_version: str = "0.3.0",
    supports_streaming: bool = False,
    supports_push_notifications: bool = False,
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "protocolVersion": protocol_version,
        "name": name,
        "description": description,
        "url": url,
        "preferredTransport": "JSONRPC",
        "additionalInterfaces": [{"url": url, "transport": "JSONRPC"}],
        "version": version,
        "capabilities": {
            "streaming": supports_streaming,
            "pushNotifications": supports_push_notifications,
            "stateTransitionHistory": True,
            "extensions": [],
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": skills
        if skills is not None
        else [
            {
                "id": "paper-qa",
                "name": "Paper QA",
                "description": "Evidence-grounded paper question answering.",
                "tags": ["paper", "qa", "rag"],
                "examples": ["Summarize this paper", "Compare this method with baseline"],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain", "application/json"],
            }
        ],
        "supportsAuthenticatedExtendedCard": False,
    }


def _text_part(text: str) -> dict[str, Any]:
    return {"kind": "text", "text": text}


def _message(role: str, text: str, *, task_id: str | None = None, context_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": role,
        "parts": [_text_part(text)],
        "messageId": str(uuid4()),
        "kind": "message",
    }
    if task_id:
        payload["taskId"] = task_id
    if context_id:
        payload["contextId"] = context_id
    return payload


def _task_status(state: str, status_message: str | None = None, *, task_id: str | None = None, context_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"state": state, "timestamp": _now_iso()}
    if isinstance(status_message, str) and status_message.strip():
        payload["message"] = _message(
            "agent",
            status_message.strip(),
            task_id=task_id,
            context_id=context_id,
        )
    return payload


@dataclass
class A2ATask:
    id: str
    context_id: str
    status: dict[str, Any]
    history: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self, *, history_length: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "contextId": self.context_id,
            "status": self.status,
            "artifacts": self.artifacts,
            "kind": "task",
            "metadata": self.metadata,
        }
        if history_length is None or history_length < 0:
            payload["history"] = self.history
        else:
            payload["history"] = self.history[-history_length:]
        return payload


class A2AInMemoryServer:
    def __init__(
        self,
        *,
        agent_card: dict[str, Any],
        execute_message_fn: Callable[[str], str],
    ) -> None:
        self.agent_card = agent_card
        self.execute_message_fn = execute_message_fn
        self.tasks: dict[str, A2ATask] = {}

    def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            return self._jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Request")

        request_id = request.get("id")
        if request.get("jsonrpc") != JSONRPC_VERSION:
            return self._jsonrpc_error(request_id, ERR_INVALID_REQUEST, "jsonrpc must be '2.0'")

        method = request.get("method")
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        if not isinstance(method, str) or not method.strip():
            return self._jsonrpc_error(request_id, ERR_INVALID_REQUEST, "method must be a string")

        try:
            if method == METHOD_MESSAGE_SEND:
                result = self._handle_message_send(params)
            elif method == METHOD_TASKS_GET:
                result = self._handle_tasks_get(params)
            elif method == METHOD_TASKS_CANCEL:
                result = self._handle_tasks_cancel(params)
            elif method == METHOD_AGENT_GET_CARD:
                result = self.agent_card
            else:
                return self._jsonrpc_error(request_id, ERR_METHOD_NOT_FOUND, f"Unknown method: {method}")
        except ValueError as exc:
            return self._jsonrpc_error(request_id, ERR_INVALID_PARAMS, str(exc))
        except KeyError as exc:
            return self._jsonrpc_error(request_id, ERR_TASK_NOT_FOUND, str(exc))
        except RuntimeError as exc:
            return self._jsonrpc_error(request_id, ERR_TASK_TERMINAL, str(exc))
        except Exception as exc:
            return self._jsonrpc_error(request_id, ERR_INTERNAL, f"Internal error: {exc}")
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    @staticmethod
    def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        payload = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        if data is not None:
            payload["error"]["data"] = data
        return payload

    @staticmethod
    def _extract_user_text(message: dict[str, Any]) -> str:
        parts = message.get("parts")
        if not isinstance(parts, list):
            raise ValueError("message.parts must be a list")
        texts: list[str] = []
        for item in parts:
            if not isinstance(item, dict):
                continue
            if item.get("kind") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        if not texts:
            raise ValueError("message.parts must include at least one text part")
        return "\n".join(texts)

    def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]:
        message = params.get("message")
        if not isinstance(message, dict):
            raise ValueError("params.message is required")
        role = message.get("role")
        if role != "user":
            raise ValueError("params.message.role must be 'user'")

        user_text = self._extract_user_text(message)
        provided_task_id = message.get("taskId")
        if provided_task_id is not None and not isinstance(provided_task_id, str):
            raise ValueError("message.taskId must be a string when provided")

        if isinstance(provided_task_id, str) and provided_task_id:
            existing = self.tasks.get(provided_task_id)
            if existing and existing.status.get("state") in TERMINAL_STATES:
                raise RuntimeError(f"task '{provided_task_id}' is terminal and cannot be restarted")

        task_id = provided_task_id if isinstance(provided_task_id, str) and provided_task_id else str(uuid4())
        context_id = message.get("contextId")
        if not isinstance(context_id, str) or not context_id:
            context_id = str(uuid4())

        user_msg = _message("user", user_text, task_id=task_id, context_id=context_id)
        task = A2ATask(
            id=task_id,
            context_id=context_id,
            status=_task_status(TASK_STATE_SUBMITTED, "Task submitted", task_id=task_id, context_id=context_id),
            history=[user_msg],
            artifacts=[],
            metadata=params.get("metadata") if isinstance(params.get("metadata"), dict) else {},
        )
        task.status = _task_status(TASK_STATE_WORKING, "Task is running", task_id=task_id, context_id=context_id)

        answer = self.execute_message_fn(user_text)
        if not isinstance(answer, str) or not answer.strip():
            answer = "No valid answer generated."
        answer = answer.strip()

        agent_msg = _message("agent", answer, task_id=task_id, context_id=context_id)
        task.history.append(agent_msg)
        task.artifacts = [
            {
                "artifactId": str(uuid4()),
                "name": "final_answer",
                "description": "Final answer generated by agent.",
                "parts": [_text_part(answer)],
                "extensions": [],
            }
        ]
        task.status = _task_status(TASK_STATE_COMPLETED, "Task completed", task_id=task_id, context_id=context_id)
        self.tasks[task_id] = task
        return task.to_dict()

    def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("params.id is required for tasks/get")
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        history_length = params.get("historyLength")
        if history_length is not None and not isinstance(history_length, int):
            raise ValueError("historyLength must be an integer")
        return task.to_dict(history_length=history_length)

    def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("params.id is required for tasks/cancel")
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        if task.status.get("state") in TERMINAL_STATES:
            return task.to_dict()
        task.status = _task_status(TASK_STATE_CANCELED, "Task canceled by client", task_id=task_id, context_id=task.context_id)
        return task.to_dict()


def build_coordinator_executor(
    coordinator: A2AMultiAgentCoordinator,
    *,
    workflow_mode: str = WORKFLOW_PLAN_ACT_REPLAN,
) -> Callable[[str], str]:
    def _executor(user_text: str) -> str:
        answer, _trace = coordinator.run(user_text, workflow_mode=workflow_mode)
        return answer

    return _executor

