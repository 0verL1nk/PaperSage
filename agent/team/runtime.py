"""Team 运行时：管理多个 agent 实例的生命周期和执行"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.domain.orchestration import TeamTodoRecord
    from agent.orchestration.executors import TaskExecutionResult

logger = logging.getLogger(__name__)
_EVIDENCE_TAG_PATTERN = re.compile(r"<evidence>([^<]+)</evidence>", flags=re.IGNORECASE)


def _extract_evidence_references(output: str) -> list[str]:
    if not isinstance(output, str) or not output.strip():
        return []
    references: list[str] = []
    seen: set[str] = set()
    for raw_ref in _EVIDENCE_TAG_PATTERN.findall(output):
        ref = str(raw_ref or "").strip()
        if not ref or ref in seen:
            continue
        seen.add(ref)
        references.append(ref)
    return references


def _summarize_output(output: str, *, limit: int = 160) -> str:
    text = " ".join(str(output or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _build_agent_result_payload(
    *,
    instance: AgentInstance,
    status: str,
    output: str = "",
    error: str = "",
    task_id: str | None = None,
) -> dict[str, Any]:
    summary_source = error or output
    payload: dict[str, Any] = {
        "kind": "agent_result_v1",
        "agent_id": instance.agent_id,
        "task_id": task_id,
        "name": instance.name,
        "profile_name": instance.profile_name,
        "thread_id": instance.thread_id,
        "status": status,
        "summary": _summarize_output(summary_source),
        "output": output,
        "evidence": _extract_evidence_references(output),
        "risks": [error] if error else [],
        "artifacts": [],
    }
    if error:
        payload["error"] = error
    return payload


def _serialize_agent_result_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _build_busy_agent_result_payload(instance: AgentInstance) -> str:
    busy_message = (
        f"Agent {instance.agent_id} is still busy. "
        "Do not send another message or close it yet; call get_agent_result later."
    )
    payload = _build_agent_result_payload(
        instance=instance,
        status="busy",
        error=busy_message,
    )
    payload["summary"] = busy_message
    payload["retry_after_ms"] = 5000
    payload["next_action"] = {
        "type": "wait_and_retry",
        "tool": "get_agent_result",
        "avoid": ["send_message", "close_agent"],
        "message": (
            "Agent is still processing the previous message. "
            "Do not send another message or close it yet; call get_agent_result later."
        ),
    }
    return _serialize_agent_result_payload(payload)


def _coerce_stored_agent_result(instance: AgentInstance, raw_content: str) -> str:
    text = str(raw_content or "").strip()
    if not text:
        return _serialize_agent_result_payload(
            _build_agent_result_payload(instance=instance, status="idle")
        )
    try:
        parsed = json.loads(text)
    except Exception:
        return _serialize_agent_result_payload(
            _build_agent_result_payload(
                instance=instance,
                status="completed",
                output=text,
            )
        )
    if isinstance(parsed, dict):
        return _serialize_agent_result_payload(parsed)
    return _serialize_agent_result_payload(
        _build_agent_result_payload(
            instance=instance,
            status="completed",
            output=text,
        )
    )


class AgentState(Enum):
    """Agent 状态"""

    IDLE = "idle"
    BUSY = "busy"
    CLOSED = "closed"


@dataclass
class AgentInstance:
    """Agent 实例"""

    agent_id: str
    name: str
    model: Any
    system_prompt: str
    tools: list[Any]
    agent: Any
    state: AgentState
    result_file: Path
    profile_name: str = ""
    thread_id: str = ""
    future: Future[str] | None = None


class TeamRuntime:
    """Team 运行时"""

    def __init__(
        self,
        team_id: str,
        result_dir: Path | None = None,
        execution_handler: Callable[[TeamTodoRecord, str], Any] | None = None,
    ):
        self.team_id = team_id
        self.result_dir = result_dir or Path(f".agent/team/{team_id}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.agents: dict[str, AgentInstance] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.execution_handler = execution_handler

    def spawn_agent(
        self,
        name: str,
        model: Any,
        system_prompt: str,
        tools: list[Any] | None = None,
        *,
        profile: Any | None = None,
        deps: Any | None = None,
    ) -> str:
        """创建新的 agent 实例"""
        agent_id = str(uuid.uuid4())
        result_file = self.result_dir / f"{agent_id}.result.txt"
        thread_id = f"team:{self.team_id}:{agent_id}"
        tools = list(tools or [])
        profile_name = ""

        if profile is None:
            from ..runtime_agent import create_runtime_agent

            agent = create_runtime_agent(
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )
        else:
            if deps is None:
                raise ValueError("Profile-based team agent requires runtime dependencies")

            from ..session_factory import AgentRuntimeOptions, create_agent_session

            session = create_agent_session(
                profile=profile,
                deps=deps,
                options=AgentRuntimeOptions(
                    llm=model,
                    system_prompt=(system_prompt or None),
                    thread_id=thread_id,
                ),
            )
            agent = session.agent
            profile_name = session.profile_name
            thread_id = session.thread_id

        instance = AgentInstance(
            agent_id=agent_id,
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            agent=agent,
            state=AgentState.IDLE,
            result_file=result_file,
            profile_name=profile_name,
            thread_id=thread_id,
        )

        self.agents[agent_id] = instance
        logger.info("Spawned agent %s (%s) profile=%s thread_id=%s", agent_id, name, profile_name or "runtime", thread_id)
        return agent_id

    def send_message(self, agent_id: str, message: str) -> None:
        """发送消息给 agent 并异步执行"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        instance = self.agents[agent_id]

        if instance.state == AgentState.CLOSED:
            raise ValueError(f"Agent {agent_id} is closed")

        if instance.state == AgentState.BUSY:
            raise ValueError(f"Agent {agent_id} is busy")

        instance.state = AgentState.BUSY
        instance.future = self.executor.submit(self._execute_agent, instance, message)
        logger.info(f"Sent message to agent {agent_id}")

    def _execute_agent(self, instance: AgentInstance, message: str) -> str:
        """执行 agent 并保存结果"""
        try:
            result = instance.agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": instance.thread_id}},
            )

            answer = ""
            if isinstance(result, dict):
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        answer = str(last_msg.content)
                    elif isinstance(last_msg, dict):
                        answer = str(last_msg.get("content", ""))

            payload = _build_agent_result_payload(
                instance=instance,
                status="completed",
                output=answer,
            )
            instance.result_file.write_text(
                _serialize_agent_result_payload(payload), encoding="utf-8"
            )
            instance.state = AgentState.IDLE
            logger.info(f"Agent {instance.agent_id} completed execution")
            return _serialize_agent_result_payload(payload)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            payload = _build_agent_result_payload(
                instance=instance,
                status="failed",
                error=error_msg,
            )
            instance.result_file.write_text(
                _serialize_agent_result_payload(payload), encoding="utf-8"
            )
            instance.state = AgentState.IDLE
            logger.error(f"Agent {instance.agent_id} execution failed: {e}")
            return _serialize_agent_result_payload(payload)

    def get_agent_result(self, agent_id: str) -> str:
        """获取 agent 的结构化执行结果。"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        instance = self.agents[agent_id]

        if instance.state == AgentState.BUSY:
            return _build_busy_agent_result_payload(instance)

        if not instance.result_file.exists():
            return _serialize_agent_result_payload(
                _build_agent_result_payload(instance=instance, status="idle")
            )

        return _coerce_stored_agent_result(
            instance,
            instance.result_file.read_text(encoding="utf-8"),
        )

    def list_agents(self) -> list[dict[str, str]]:
        """列出所有 agent 及其状态"""
        return [
            {
                "agent_id": instance.agent_id,
                "name": instance.name,
                "state": instance.state.value,
                "profile_name": instance.profile_name,
                "thread_id": instance.thread_id,
            }
            for instance in self.agents.values()
        ]

    def close_agent(self, agent_id: str) -> None:
        """关闭 agent 实例"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        instance = self.agents[agent_id]

        if instance.state == AgentState.BUSY:
            raise ValueError(f"Cannot close agent {agent_id}: still busy")

        instance.state = AgentState.CLOSED
        del self.agents[agent_id]
        logger.info(f"Closed agent {agent_id}")

    def cleanup(self) -> None:
        """清理资源"""
        self.executor.shutdown(wait=True)
        logger.info(f"Team {self.team_id} cleaned up")

    def execute_todo(self, todo: TeamTodoRecord, message: str) -> TaskExecutionResult:
        """Execute a todo through the local team runtime contract."""
        from agent.orchestration.executors import normalize_task_execution_result

        if self.execution_handler is None:
            return normalize_task_execution_result(
                {
                    "status": "failed",
                    "error": "No local execution handler configured",
                },
                todo_id=todo.id,
                backend="local",
            )
        try:
            result = self.execution_handler(todo, message)
        except Exception as exc:
            return normalize_task_execution_result(
                {
                    "status": "failed",
                    "error": str(exc),
                },
                todo_id=todo.id,
                backend="local",
            )
        return normalize_task_execution_result(result, todo_id=todo.id, backend="local")
