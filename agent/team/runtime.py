"""Team 运行时：管理多个 agent 实例的生命周期和执行"""

from __future__ import annotations

import logging
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
    agent: Any  # 实际的 agent 对象
    state: AgentState
    result_file: Path
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
        tools: list[Any],
    ) -> str:
        """创建新的 agent 实例"""
        from ..runtime_agent import create_runtime_agent

        agent_id = str(uuid.uuid4())
        result_file = self.result_dir / f"{agent_id}.result.txt"

        agent = create_runtime_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
        )

        instance = AgentInstance(
            agent_id=agent_id,
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            agent=agent,
            state=AgentState.IDLE,
            result_file=result_file,
        )

        self.agents[agent_id] = instance
        logger.info(f"Spawned agent {agent_id} ({name})")
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
                config={"configurable": {"thread_id": f"team:{self.team_id}:{instance.agent_id}"}},
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

            instance.result_file.write_text(answer, encoding="utf-8")
            instance.state = AgentState.IDLE
            logger.info(f"Agent {instance.agent_id} completed execution")
            return answer

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            instance.result_file.write_text(error_msg, encoding="utf-8")
            instance.state = AgentState.IDLE
            logger.error(f"Agent {instance.agent_id} execution failed: {e}")
            return error_msg

    def get_agent_result(self, agent_id: str) -> str:
        """获取 agent 的执行结果"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        instance = self.agents[agent_id]

        if instance.state == AgentState.BUSY:
            return f"Agent {agent_id} is still busy"

        if not instance.result_file.exists():
            return ""

        return instance.result_file.read_text(encoding="utf-8")

    def list_agents(self) -> list[dict[str, str]]:
        """列出所有 agent 及其状态"""
        return [
            {
                "agent_id": instance.agent_id,
                "name": instance.name,
                "state": instance.state.value,
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
