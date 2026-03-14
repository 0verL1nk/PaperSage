from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator
from uuid import uuid4

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    CancelTaskRequest,
    DeleteTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigRequest,
    GetTaskRequest,
    JSONRPCError,
    JSONRPCErrorResponse,
    ListTaskPushNotificationConfigRequest,
    Message,
    Part,
    PushNotificationNotSupportedError,
    Role,
    SendMessageRequest,
    SetTaskPushNotificationConfigRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskResubscriptionRequest,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

from .coordinator import (
    WORKFLOW_TEAM,
    A2AMultiAgentCoordinator,
    canonicalize_workflow_mode,
)

JSONRPC_VERSION = "2.0"
METHOD_MESSAGE_SEND = "message/send"
METHOD_MESSAGE_STREAM = "message/stream"
METHOD_TASKS_GET = "tasks/get"
METHOD_TASKS_CANCEL = "tasks/cancel"
METHOD_TASKS_RESUBSCRIBE = "tasks/resubscribe"
METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_SET = "tasks/pushNotificationConfig/set"
METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_GET = "tasks/pushNotificationConfig/get"
METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_LIST = "tasks/pushNotificationConfig/list"
METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_DELETE = "tasks/pushNotificationConfig/delete"
METHOD_AGENT_GET_CARD = "agent/getAuthenticatedExtendedCard"
METHOD_SEND_MESSAGE = "SendMessage"
METHOD_GET_TASK = "GetTask"
METHOD_CANCEL_TASK = "CancelTask"
METHOD_SEND_STREAMING_MESSAGE = "SendStreamingMessage"
METHOD_RESUBSCRIBE_TASK = "ResubscribeTask"
METHOD_SET_TASK_PUSH_NOTIFICATION_CONFIG = "SetTaskPushNotificationConfig"
METHOD_GET_TASK_PUSH_NOTIFICATION_CONFIG = "GetTaskPushNotificationConfig"
METHOD_LIST_TASK_PUSH_NOTIFICATION_CONFIG = "ListTaskPushNotificationConfig"
METHOD_DELETE_TASK_PUSH_NOTIFICATION_CONFIG = "DeleteTaskPushNotificationConfig"
METHOD_GET_AUTHENTICATED_EXTENDED_CARD = "GetAuthenticatedExtendedCard"

A2A_VERSION_HEADER = "A2A-Version"
A2A_VERSION_V03 = "0.3.0"
A2A_VERSION_V1 = "1.0"
DEFAULT_SUPPORTED_A2A_VERSIONS = (A2A_VERSION_V03, A2A_VERSION_V1)

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
VALID_INPUT_MODES = {"hybrid", "strict"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sdk_dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


def _coerce_task_state(state: str) -> TaskState:
    try:
        return TaskState(str(state))
    except Exception:
        return TaskState.unknown


def _resolve_input_mode() -> str:
    raw = str(os.getenv("AGENT_A2A_INPUT_MODE", "hybrid") or "").strip().lower()
    if raw in VALID_INPUT_MODES:
        return raw
    return "hybrid"


def build_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    version: str = "1.0.0",
    protocol_version: str = A2A_VERSION_V03,
    a2a_version: str = A2A_VERSION_V1,
    supported_versions: list[str] | None = None,
    supports_streaming: bool = False,
    supports_push_notifications: bool = False,
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    negotiated_versions = _normalize_supported_versions(
        supported_versions or [protocol_version, a2a_version]
    )
    normalized_skills = (
        skills
        if skills is not None
        else [
            {
                "id": "paper-qa",
                "name": "Paper QA",
                "description": "Evidence-grounded paper question answering.",
                "tags": ["paper", "qa", "rag"],
                "examples": ["Summarize this paper", "Compare this method with baseline"],
                "input_modes": ["text/plain", "application/json"],
                "output_modes": ["text/plain", "application/json"],
            }
        ]
    )
    card_model = AgentCard(
        protocol_version=protocol_version,
        name=name,
        description=description,
        url=url,
        preferred_transport="JSONRPC",
        additional_interfaces=[AgentInterface(url=url, transport="JSONRPC")],
        version=version,
        capabilities=AgentCapabilities(
            streaming=supports_streaming,
            push_notifications=supports_push_notifications,
            state_transition_history=True,
            extensions=[],
        ),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=[AgentSkill(**item) for item in normalized_skills if isinstance(item, dict)],
        supports_authenticated_extended_card=False,
    )
    payload = _sdk_dump(card_model)
    payload["a2aVersion"] = a2a_version
    payload["supportedA2AVersions"] = negotiated_versions
    return payload


def _normalize_supported_versions(raw_versions: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_versions:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized or list(DEFAULT_SUPPORTED_A2A_VERSIONS)


def _text_part(text: str) -> dict[str, Any]:
    return _sdk_dump(Part(root=TextPart(text=text)))


def _message(
    role: str, text: str, *, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    normalized_role = Role.user if str(role).strip().lower() == "user" else Role.agent
    model = Message(
        role=normalized_role,
        parts=[Part(root=TextPart(text=text))],
        message_id=str(uuid4()),
        task_id=task_id,
        context_id=context_id,
    )
    return _sdk_dump(model)


def _task_status(
    state: str,
    status_message: str | None = None,
    *,
    task_id: str | None = None,
    context_id: str | None = None,
) -> dict[str, Any]:
    model = TaskStatus(
        state=_coerce_task_state(state),
        timestamp=_now_iso(),
        message=(
            Message(
                role=Role.agent,
                parts=[Part(root=TextPart(text=status_message.strip()))],
                message_id=str(uuid4()),
                task_id=task_id,
                context_id=context_id,
            )
            if isinstance(status_message, str) and status_message.strip()
            else None
        ),
    )
    return _sdk_dump(model)


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
            "context_id": self.context_id,
            "status": self.status,
            "artifacts": self.artifacts,
            "kind": "task",
            "metadata": self.metadata,
        }
        if history_length is None or history_length < 0:
            payload["history"] = self.history
        else:
            payload["history"] = self.history[-history_length:]
        try:
            return _sdk_dump(Task.model_validate(payload))
        except Exception:
            return payload


class A2AInMemoryServer:
    def __init__(
        self,
        *,
        agent_card: dict[str, Any],
        execute_message_fn: Callable[[str], str],
        execute_message_stream_fn: Callable[[str], Iterable[str]] | None = None,
        supported_versions: list[str] | None = None,
    ) -> None:
        self.agent_card = agent_card
        self.execute_message_fn = execute_message_fn
        self.execute_message_stream_fn = execute_message_stream_fn
        self.input_mode = _resolve_input_mode()
        self.tasks: dict[str, A2ATask] = {}
        capabilities = agent_card.get("capabilities")
        self.supports_streaming = False
        self.supports_push_notifications = False
        if isinstance(capabilities, dict):
            self.supports_streaming = bool(capabilities.get("streaming"))
            self.supports_push_notifications = bool(
                capabilities.get("pushNotifications")
                if capabilities.get("pushNotifications") is not None
                else capabilities.get("push_notifications")
            )
        self.push_notification_configs: dict[str, list[dict[str, Any]]] = {}
        if isinstance(supported_versions, list) and supported_versions:
            resolved_versions = _normalize_supported_versions(supported_versions)
        else:
            card_versions = agent_card.get("supportedA2AVersions")
            if isinstance(card_versions, list) and card_versions:
                resolved_versions = _normalize_supported_versions(card_versions)
            else:
                protocol_version = str(
                    agent_card.get("protocol_version")
                    or agent_card.get("protocolVersion")
                    or A2A_VERSION_V03
                )
                a2a_version = str(
                    agent_card.get("a2a_version") or agent_card.get("a2aVersion") or A2A_VERSION_V1
                )
                resolved_versions = _normalize_supported_versions([protocol_version, a2a_version])
        self.supported_versions = resolved_versions
        self.default_version = (
            A2A_VERSION_V1
            if A2A_VERSION_V1 in self.supported_versions
            else self.supported_versions[0]
        )

    def _reject_legacy_message_fields_if_strict(self, message: dict[str, Any]) -> None:
        if self.input_mode != "strict":
            return
        legacy_keys = [key for key in ("taskId", "contextId", "messageId") if key in message]
        if not legacy_keys:
            return
        raise ValueError(
            "params.message uses legacy camelCase fields in strict mode; "
            f"found={','.join(sorted(legacy_keys))}. "
            "Use task_id/context_id/message_id."
        )

    def _read_compat_param(
        self,
        params: dict[str, Any],
        *,
        snake: str,
        camel: str,
        label: str,
    ) -> Any:
        value = params.get(snake)
        if value is not None:
            return value
        legacy_value = params.get(camel)
        if legacy_value is None:
            return None
        if self.input_mode == "strict":
            raise ValueError(f"{label}.{camel} is not allowed in strict mode; use {snake}")
        return legacy_value

    @staticmethod
    def _iter_nonempty_text_chunks(chunks: Iterable[str]) -> Iterator[str]:
        for item in chunks:
            text = str(item or "")
            if not text:
                continue
            yield text

    def _ensure_streaming_supported(self) -> None:
        if self.supports_streaming:
            return
        error = UnsupportedOperationError(message="Streaming is not supported by this agent")
        raise ValueError(error.message)

    def _ensure_push_notifications_supported(self) -> None:
        if self.supports_push_notifications:
            return
        error = PushNotificationNotSupportedError()
        raise ValueError(error.message)

    def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            return self._jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Request")

        request_id = request.get("id")
        if request.get("jsonrpc") != JSONRPC_VERSION:
            return self._jsonrpc_error(request_id, ERR_INVALID_REQUEST, "jsonrpc must be '2.0'")

        method = request.get("method")
        raw_params = request.get("params")
        params: dict[str, Any] = (
            {str(key): value for key, value in raw_params.items()}
            if isinstance(raw_params, dict)
            else {}
        )
        if not isinstance(method, str) or not method.strip():
            return self._jsonrpc_error(request_id, ERR_INVALID_REQUEST, "method must be a string")
        negotiated_version = self._resolve_a2a_version(
            request=request, method=method, params=params
        )
        if negotiated_version not in self.supported_versions:
            return self._jsonrpc_error(
                request_id,
                ERR_INVALID_PARAMS,
                f"Unsupported A2A version: {negotiated_version}",
                data={"supportedVersions": self.supported_versions},
            )
        canonical_method = self._canonical_method_name(method)
        if canonical_method is None:
            return self._jsonrpc_error(
                request_id, ERR_METHOD_NOT_FOUND, f"Unknown method: {method}"
            )

        try:
            result: Any
            if canonical_method == "send_message":
                result = self._handle_message_send(params)
            elif canonical_method == "get_task":
                result = self._handle_tasks_get(params)
            elif canonical_method == "cancel_task":
                result = self._handle_tasks_cancel(params)
            elif canonical_method == "get_card":
                result = self.agent_card
            elif canonical_method == "send_streaming_message":
                result = self._handle_send_streaming_message(params)
            elif canonical_method == "resubscribe_task":
                result = self._handle_tasks_resubscribe(params)
            elif canonical_method == "set_push_notification_config":
                result = self._handle_set_push_notification_config(params)
            elif canonical_method == "get_push_notification_config":
                result = self._handle_get_push_notification_config(params)
            elif canonical_method == "list_push_notification_config":
                result = self._handle_list_push_notification_config(params)
            elif canonical_method == "delete_push_notification_config":
                result = self._handle_delete_push_notification_config(params)
            else:
                return self._jsonrpc_error(
                    request_id,
                    ERR_METHOD_NOT_FOUND,
                    f"Unknown method: {method}",
                )
        except ValueError as exc:
            return self._jsonrpc_error(request_id, ERR_INVALID_PARAMS, str(exc))
        except KeyError as exc:
            return self._jsonrpc_error(request_id, ERR_TASK_NOT_FOUND, str(exc))
        except RuntimeError as exc:
            return self._jsonrpc_error(request_id, ERR_TASK_TERMINAL, str(exc))
        except Exception as exc:
            return self._jsonrpc_error(request_id, ERR_INTERNAL, f"Internal error: {exc}")
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "result": result,
            "meta": {
                "a2aVersion": negotiated_version,
                "method": canonical_method,
            },
        }

    def handle_jsonrpc_stream(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if not isinstance(request, dict):
            yield self._jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Request")
            return
        request_id = request.get("id")
        if request.get("jsonrpc") != JSONRPC_VERSION:
            yield self._jsonrpc_error(request_id, ERR_INVALID_REQUEST, "jsonrpc must be '2.0'")
            return
        method = request.get("method")
        raw_params = request.get("params")
        params: dict[str, Any] = (
            {str(key): value for key, value in raw_params.items()}
            if isinstance(raw_params, dict)
            else {}
        )
        if not isinstance(method, str) or not method.strip():
            yield self._jsonrpc_error(request_id, ERR_INVALID_REQUEST, "method must be a string")
            return
        negotiated_version = self._resolve_a2a_version(
            request=request, method=method, params=params
        )
        if negotiated_version not in self.supported_versions:
            yield self._jsonrpc_error(
                request_id,
                ERR_INVALID_PARAMS,
                f"Unsupported A2A version: {negotiated_version}",
                data={"supportedVersions": self.supported_versions},
            )
            return
        canonical_method = self._canonical_method_name(method)
        if canonical_method not in {"send_streaming_message", "resubscribe_task"}:
            # Non-stream methods degrade to single response frame for compatibility.
            response = self.handle_jsonrpc(request)
            response.setdefault("meta", {})
            response["meta"]["stream"] = False
            yield response
            return

        sequence = 0
        try:
            event_stream: Iterator[dict[str, Any]]
            if canonical_method == "send_streaming_message":
                event_stream = self._iter_send_streaming_events(params)
            else:
                event_stream = self._iter_tasks_resubscribe_events(params)
            for event in event_stream:
                sequence += 1
                yield {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": request_id,
                    "result": {"event": event},
                    "meta": {
                        "a2aVersion": negotiated_version,
                        "method": canonical_method,
                        "stream": True,
                        "sequence": sequence,
                    },
                }
        except ValueError as exc:
            yield self._jsonrpc_error(request_id, ERR_INVALID_PARAMS, str(exc))
        except KeyError as exc:
            yield self._jsonrpc_error(request_id, ERR_TASK_NOT_FOUND, str(exc))
        except RuntimeError as exc:
            yield self._jsonrpc_error(request_id, ERR_TASK_TERMINAL, str(exc))
        except Exception as exc:
            yield self._jsonrpc_error(request_id, ERR_INTERNAL, f"Internal error: {exc}")

    @staticmethod
    def _extract_header(
        request: dict[str, Any],
        header_name: str,
    ) -> str | None:
        raw_headers = request.get("headers")
        if not isinstance(raw_headers, dict):
            return None
        lowered = {str(k).lower(): v for k, v in raw_headers.items()}
        value = lowered.get(header_name.lower())
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    def _resolve_a2a_version(
        self,
        *,
        request: dict[str, Any],
        method: str,
        params: dict[str, Any],
    ) -> str:
        header_version = self._extract_header(request, A2A_VERSION_HEADER)
        if isinstance(header_version, str) and header_version.strip():
            return header_version.strip()
        param_version = self._read_compat_param(
            params,
            snake="a2a_version",
            camel="a2aVersion",
            label="params",
        )
        if isinstance(param_version, str) and param_version.strip():
            return param_version.strip()
        if "/" in method:
            return A2A_VERSION_V03
        return self.default_version

    @staticmethod
    def _canonical_method_name(method: str) -> str | None:
        normalized = str(method or "").strip()
        mapping = {
            METHOD_MESSAGE_SEND: "send_message",
            METHOD_SEND_MESSAGE: "send_message",
            METHOD_MESSAGE_STREAM: "send_streaming_message",
            METHOD_TASKS_GET: "get_task",
            METHOD_GET_TASK: "get_task",
            METHOD_TASKS_CANCEL: "cancel_task",
            METHOD_CANCEL_TASK: "cancel_task",
            METHOD_TASKS_RESUBSCRIBE: "resubscribe_task",
            METHOD_RESUBSCRIBE_TASK: "resubscribe_task",
            METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_SET: "set_push_notification_config",
            METHOD_SET_TASK_PUSH_NOTIFICATION_CONFIG: "set_push_notification_config",
            METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_GET: "get_push_notification_config",
            METHOD_GET_TASK_PUSH_NOTIFICATION_CONFIG: "get_push_notification_config",
            METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_LIST: "list_push_notification_config",
            METHOD_LIST_TASK_PUSH_NOTIFICATION_CONFIG: "list_push_notification_config",
            METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_DELETE: "delete_push_notification_config",
            METHOD_DELETE_TASK_PUSH_NOTIFICATION_CONFIG: "delete_push_notification_config",
            METHOD_AGENT_GET_CARD: "get_card",
            METHOD_GET_AUTHENTICATED_EXTENDED_CARD: "get_card",
            METHOD_SEND_STREAMING_MESSAGE: "send_streaming_message",
        }
        return mapping.get(normalized)

    @staticmethod
    def _jsonrpc_error(
        request_id: Any, code: int, message: str, data: Any = None
    ) -> dict[str, Any]:
        error = JSONRPCError(code=code, message=message, data=data)
        payload = JSONRPCErrorResponse(id=request_id, error=error)
        return _sdk_dump(payload)

    @staticmethod
    def _extract_user_text(message: Message) -> str:
        texts: list[str] = []
        for part in message.parts:
            root = getattr(part, "root", None)
            if not isinstance(root, TextPart):
                continue
            text = str(root.text or "").strip()
            if text:
                texts.append(text)
        if not texts:
            raise ValueError("message.parts must include at least one text part")
        return "\n".join(texts)

    def _build_or_reuse_task(
        self,
        params: dict[str, Any],
    ) -> tuple[A2ATask, str]:
        raw_message = params.get("message")
        if not isinstance(raw_message, dict):
            raise ValueError("params.message is required")
        self._reject_legacy_message_fields_if_strict(raw_message)
        normalized_message = dict(raw_message)
        if (
            normalized_message.get("message_id") is None
            and normalized_message.get("messageId") is None
        ):
            normalized_message["message_id"] = str(uuid4())
        normalized_params = dict(params)
        normalized_params["message"] = normalized_message
        try:
            request_payload = SendMessageRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_MESSAGE_SEND,
                    "params": normalized_params,
                }
            )
        except Exception as exc:
            raise ValueError(f"params invalid for message/send: {exc}") from exc
        message = request_payload.params.message
        if message.role != Role.user:
            raise ValueError("params.message.role must be 'user'")

        user_text = self._extract_user_text(message)
        provided_task_id = message.task_id
        if provided_task_id is not None and not isinstance(provided_task_id, str):
            raise ValueError("message.taskId must be a string when provided")

        if isinstance(provided_task_id, str) and provided_task_id:
            existing = self.tasks.get(provided_task_id)
            if existing and existing.status.get("state") in TERMINAL_STATES:
                raise RuntimeError(f"task '{provided_task_id}' is terminal and cannot be restarted")

        task_id = (
            provided_task_id
            if isinstance(provided_task_id, str) and provided_task_id
            else str(uuid4())
        )
        context_id = message.context_id
        if not isinstance(context_id, str) or not context_id:
            context_id = str(uuid4())

        user_msg = _message("user", user_text, task_id=task_id, context_id=context_id)
        metadata_raw = request_payload.params.metadata
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        task = A2ATask(
            id=task_id,
            context_id=context_id,
            status=_task_status(
                TASK_STATE_SUBMITTED, "Task submitted", task_id=task_id, context_id=context_id
            ),
            history=[user_msg],
            artifacts=[],
            metadata=metadata,
        )
        return task, user_text

    def _finalize_task_with_answer(self, task: A2ATask, answer: str) -> dict[str, Any]:
        final_answer = str(answer or "").strip()
        if not final_answer:
            final_answer = "No valid answer generated."
        agent_msg = _message("agent", final_answer, task_id=task.id, context_id=task.context_id)
        task.history.append(agent_msg)
        task.artifacts = [
            _sdk_dump(
                Artifact(
                    artifact_id=str(uuid4()),
                    name="final_answer",
                    description="Final answer generated by agent.",
                    parts=[Part(root=TextPart(text=final_answer))],
                    extensions=[],
                )
            )
        ]
        task.status = _task_status(
            TASK_STATE_COMPLETED,
            "Task completed",
            task_id=task.id,
            context_id=task.context_id,
        )
        self.tasks[task.id] = task
        return task.to_dict()

    def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]:
        task, user_text = self._build_or_reuse_task(params)
        task.status = _task_status(
            TASK_STATE_WORKING, "Task is running", task_id=task.id, context_id=task.context_id
        )
        answer = self.execute_message_fn(user_text)
        return self._finalize_task_with_answer(task, str(answer or ""))

    def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        history_length = self._read_compat_param(
            params,
            snake="history_length",
            camel="historyLength",
            label="params",
        )
        if history_length is not None and not isinstance(history_length, int):
            raise ValueError("historyLength must be an integer")
        try:
            validated = GetTaskRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_TASKS_GET,
                    "params": params,
                }
            )
        except Exception as exc:
            raise ValueError(f"params invalid for tasks/get: {exc}") from exc
        task_id = validated.params.id
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        return task.to_dict(history_length=validated.params.history_length)

    def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            validated = CancelTaskRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_TASKS_CANCEL,
                    "params": params,
                }
            )
        except Exception as exc:
            raise ValueError(f"params invalid for tasks/cancel: {exc}") from exc
        task_id = validated.params.id
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        if task.status.get("state") in TERMINAL_STATES:
            return task.to_dict()
        task.status = _task_status(
            TASK_STATE_CANCELED,
            "Task canceled by client",
            task_id=task_id,
            context_id=task.context_id,
        )
        return task.to_dict()

    def _handle_tasks_resubscribe(self, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_streaming_supported()
        events = list(self._iter_tasks_resubscribe_events(params))
        task_payload: dict[str, Any] = {}
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            maybe_task = event.get("task")
            if isinstance(maybe_task, dict):
                task_payload = maybe_task
                break
        if not task_payload:
            task_id = params.get("id")
            if isinstance(task_id, str) and task_id in self.tasks:
                task_payload = self.tasks[task_id].to_dict()
        return {
            "task": task_payload,
            "events": events,
        }

    def _handle_set_push_notification_config(self, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_push_notifications_supported()
        try:
            validated = SetTaskPushNotificationConfigRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_SET,
                    "params": params,
                }
            )
        except Exception as exc:
            raise ValueError(f"params invalid for tasks/pushNotificationConfig/set: {exc}") from exc
        task_id = validated.params.task_id
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        config_payload = _sdk_dump(validated.params.push_notification_config)
        config_id = config_payload.get("id")
        if not isinstance(config_id, str) or not config_id:
            config_payload["id"] = str(uuid4())
            config_id = config_payload["id"]
        task_configs = self.push_notification_configs.setdefault(task_id, [])
        replaced = False
        for index, existing in enumerate(task_configs):
            if isinstance(existing, dict) and existing.get("id") == config_id:
                task_configs[index] = config_payload
                replaced = True
                break
        if not replaced:
            task_configs.append(config_payload)
        return {"taskId": task_id, "pushNotificationConfig": config_payload}

    def _handle_get_push_notification_config(self, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_push_notifications_supported()
        try:
            validated = GetTaskPushNotificationConfigRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_GET,
                    "params": params,
                }
            )
        except Exception as exc:
            raise ValueError(f"params invalid for tasks/pushNotificationConfig/get: {exc}") from exc
        task_id = validated.params.id
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        configs = self.push_notification_configs.get(task_id) or []
        if not configs:
            raise ValueError("Push notification config not found")
        config_id = getattr(validated.params, "push_notification_config_id", None)
        selected: dict[str, Any] | None = None
        if isinstance(config_id, str) and config_id:
            for item in configs:
                if isinstance(item, dict) and item.get("id") == config_id:
                    selected = item
                    break
            if selected is None:
                raise ValueError(
                    f"Push notification config '{config_id}' not found for task '{task_id}'"
                )
        else:
            first = configs[0]
            if isinstance(first, dict):
                selected = first
        if selected is None:
            raise ValueError("Push notification config not found")
        return {"taskId": task_id, "pushNotificationConfig": selected}

    def _handle_list_push_notification_config(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._ensure_push_notifications_supported()
        try:
            validated = ListTaskPushNotificationConfigRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_LIST,
                    "params": params,
                }
            )
        except Exception as exc:
            raise ValueError(
                f"params invalid for tasks/pushNotificationConfig/list: {exc}"
            ) from exc
        task_id = validated.params.id
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        configs = self.push_notification_configs.get(task_id) or []
        payload: list[dict[str, Any]] = []
        for item in configs:
            if not isinstance(item, dict):
                continue
            payload.append({"taskId": task_id, "pushNotificationConfig": item})
        return payload

    def _handle_delete_push_notification_config(
        self, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        self._ensure_push_notifications_supported()
        try:
            validated = DeleteTaskPushNotificationConfigRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_TASKS_PUSH_NOTIFICATION_CONFIG_DELETE,
                    "params": params,
                }
            )
        except Exception as exc:
            raise ValueError(
                f"params invalid for tasks/pushNotificationConfig/delete: {exc}"
            ) from exc
        task_id = validated.params.id
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        configs = self.push_notification_configs.get(task_id) or []
        if not configs:
            return None
        config_id = validated.params.push_notification_config_id
        self.push_notification_configs[task_id] = [
            item for item in configs if not isinstance(item, dict) or item.get("id") != config_id
        ]
        return None

    def _iter_answer_chunks(self, user_text: str) -> Iterator[str]:
        if callable(self.execute_message_stream_fn):
            try:
                yield from self._iter_nonempty_text_chunks(
                    self.execute_message_stream_fn(user_text)
                )
                return
            except Exception:
                pass
        fallback = self.execute_message_fn(user_text)
        fallback_text = str(fallback or "").strip()
        if fallback_text:
            yield fallback_text
            return
        yield "No valid answer generated."

    def _iter_send_streaming_events(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self._ensure_streaming_supported()
        task, user_text = self._build_or_reuse_task(params)
        task.status = _task_status(
            TASK_STATE_WORKING,
            "Task is running",
            task_id=task.id,
            context_id=task.context_id,
        )
        self.tasks[task.id] = task
        working_status_model = TaskStatus.model_validate(task.status)
        yield {
            "event": "TaskStatusUpdate",
            **_sdk_dump(
                TaskStatusUpdateEvent(
                    context_id=task.context_id,
                    task_id=task.id,
                    status=working_status_model,
                    final=False,
                )
            ),
        }

        answer_parts: list[str] = []
        for index, chunk in enumerate(self._iter_answer_chunks(user_text), start=1):
            answer_parts.append(chunk)
            chunk_artifact = Artifact(
                artifact_id=f"{task.id}:chunk:{index}",
                name="answer_chunk",
                description="Streaming answer chunk.",
                parts=[Part(root=TextPart(text=chunk))],
            )
            yield {
                "event": "TaskArtifactUpdate",
                **_sdk_dump(
                    TaskArtifactUpdateEvent(
                        context_id=task.context_id,
                        task_id=task.id,
                        artifact=chunk_artifact,
                        append=True,
                        last_chunk=False,
                    )
                ),
            }

        final_task = self._finalize_task_with_answer(task, "".join(answer_parts))
        artifacts = final_task.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                yield {
                    "event": "TaskArtifactUpdate",
                    **_sdk_dump(
                        TaskArtifactUpdateEvent(
                            context_id=task.context_id,
                            task_id=task.id,
                            artifact=Artifact.model_validate(artifact),
                            append=True,
                            last_chunk=True,
                        )
                    ),
                }
        completed_status = TaskStatus.model_validate(final_task.get("status") or {})
        yield {
            "event": "TaskCompleted",
            **_sdk_dump(
                TaskStatusUpdateEvent(
                    context_id=task.context_id,
                    task_id=task.id,
                    status=completed_status,
                    final=True,
                )
            ),
            "task": final_task,
        }

    def _iter_tasks_resubscribe_events(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            validated = TaskResubscriptionRequest.model_validate(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": "local",
                    "method": METHOD_TASKS_RESUBSCRIBE,
                    "params": params,
                }
            )
        except Exception as exc:
            raise ValueError(f"params invalid for tasks/resubscribe: {exc}") from exc
        task_id = validated.params.id
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")
        status = TaskStatus.model_validate(task.status)
        is_terminal = str(status.state.value) in TERMINAL_STATES
        yield {
            "event": "TaskStatusUpdate",
            **_sdk_dump(
                TaskStatusUpdateEvent(
                    context_id=task.context_id,
                    task_id=task.id,
                    status=status,
                    final=is_terminal,
                )
            ),
        }
        artifacts = list(task.artifacts or [])
        artifact_count = len(artifacts)
        for index, artifact in enumerate(artifacts, start=1):
            if not isinstance(artifact, dict):
                continue
            yield {
                "event": "TaskArtifactUpdate",
                **_sdk_dump(
                    TaskArtifactUpdateEvent(
                        context_id=task.context_id,
                        task_id=task.id,
                        artifact=Artifact.model_validate(artifact),
                        append=True,
                        last_chunk=index == artifact_count,
                    )
                ),
            }
        if is_terminal:
            yield {
                "event": "TaskCompleted",
                **_sdk_dump(
                    TaskStatusUpdateEvent(
                        context_id=task.context_id,
                        task_id=task.id,
                        status=status,
                        final=True,
                    )
                ),
                "task": task.to_dict(),
            }

    def _handle_send_streaming_message(self, params: dict[str, Any]) -> dict[str, Any]:
        events = list(self._iter_send_streaming_events(params))
        final_task_payload: dict[str, Any] = {}
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            maybe_task = event.get("task")
            if isinstance(maybe_task, dict):
                final_task_payload = maybe_task
                break
        return {
            "task": final_task_payload,
            "events": events,
        }


def build_coordinator_executor(
    coordinator: A2AMultiAgentCoordinator,
    *,
    workflow_mode: str = WORKFLOW_TEAM,
) -> Callable[[str], str]:
    def _executor(user_text: str) -> str:
        answer, _trace = coordinator.run(
            user_text,
            workflow_mode=canonicalize_workflow_mode(workflow_mode),
        )
        return answer

    return _executor
