"""Team 工具：提供 agent 团队管理能力"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool

from ..profiles import resolve_agent_profile
from ..team.runtime import TeamRuntime

_team_runtimes: dict[str, TeamRuntime] = {}
_current_session: ContextVar[str | None] = ContextVar("current_session", default=None)


@dataclass(frozen=True)
class TeamRuntimeContext:
    default_model: Any
    dependencies: Any | None = None


_session_runtime_contexts: dict[str, TeamRuntimeContext] = {}


def set_current_session(session_id: str) -> None:
    """设置当前 session ID"""
    _current_session.set(session_id)



def set_session_runtime_context(
    session_id: str,
    *,
    default_model: Any,
    dependencies: Any | None = None,
) -> None:
    """为当前 session 注册 team runtime 所需上下文。"""
    _session_runtime_contexts[session_id] = TeamRuntimeContext(
        default_model=default_model,
        dependencies=dependencies,
    )



def get_team_runtime() -> TeamRuntime:
    """获取或创建当前 session 的 team runtime"""
    session_id = _current_session.get()
    if session_id is None:
        session_id = "default"

    if session_id not in _team_runtimes:
        team_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
        _team_runtimes[session_id] = TeamRuntime(team_id)

    return _team_runtimes[session_id]



def _get_current_runtime_context() -> TeamRuntimeContext | None:
    session_id = _current_session.get()
    if session_id is None:
        return None
    return _session_runtime_contexts.get(session_id)


@tool
def spawn_agent(name: str, role: str = "teammate", system_prompt: str = "") -> str:
    """创建新的 agent 实例

    Args:
        name: agent 名称
        role: agent 角色，默认 teammate
        system_prompt: 可选的角色补充提示

    Returns:
        agent_id
    """
    runtime = get_team_runtime()
    runtime_context = _get_current_runtime_context()
    if runtime_context is None or runtime_context.dependencies is None:
        return "Error: Team runtime context unavailable"

    try:
        profile = resolve_agent_profile(role)
    except ValueError as exc:
        return f"Error: {exc}"

    agent_id = runtime.spawn_agent(
        name=name,
        model=runtime_context.default_model,
        system_prompt=system_prompt,
        tools=[],
        profile=profile,
        deps=runtime_context.dependencies,
    )
    return f"Agent spawned: {agent_id} (role={role}, profile={profile.name})"


@tool
def send_message(agent_id: str, message: str) -> str:
    """发送消息给 agent

    Args:
        agent_id: agent ID
        message: 消息内容

    Returns:
        状态信息
    """
    runtime = get_team_runtime()
    try:
        runtime.send_message(agent_id, message)
        return f"Message sent to agent {agent_id}"
    except ValueError as e:
        return f"Error: {str(e)}"


@tool
def list_agents() -> str:
    """列出所有 agent 及其状态

    Returns:
        agent 列表（JSON 格式）
    """
    runtime = get_team_runtime()
    agents = runtime.list_agents()
    import json

    return json.dumps(agents, indent=2)


@tool
def get_agent_result(agent_id: str) -> str:
    """获取 agent 的执行结果

    Args:
        agent_id: agent ID

    Returns:
        结构化执行结果（JSON 字符串）
    """
    runtime = get_team_runtime()
    try:
        return runtime.get_agent_result(agent_id)
    except ValueError as e:
        return f"Error: {str(e)}"


@tool
def close_agent(agent_id: str) -> str:
    """关闭 agent 实例

    Args:
        agent_id: agent ID

    Returns:
        状态信息
    """
    runtime = get_team_runtime()
    try:
        runtime.close_agent(agent_id)
        return f"Agent {agent_id} closed"
    except ValueError as e:
        return f"Error: {str(e)}"
