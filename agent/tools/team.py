"""Team 工具：提供 agent 团队管理能力"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
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



def _resolve_session_id(config: RunnableConfig | None = None) -> str:
    if config is not None:
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if thread_id:
            return str(thread_id)
    session_id = _current_session.get()
    if session_id:
        return session_id
    return "default"


def get_team_runtime() -> TeamRuntime:
    """获取或创建当前 session 的 team runtime"""
    session_id = _resolve_session_id()

    if session_id not in _team_runtimes:
        team_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
        _team_runtimes[session_id] = TeamRuntime(team_id)

    return _team_runtimes[session_id]


def _get_runtime_for_tool_call(config: RunnableConfig | None = None) -> TeamRuntime:
    session_id = _resolve_session_id(config)
    set_current_session(session_id)
    return get_team_runtime()



def _get_current_runtime_context(
    config: RunnableConfig | None = None,
) -> TeamRuntimeContext | None:
    session_id = _resolve_session_id(config)
    return _session_runtime_contexts.get(session_id)


def _format_team_runtime_error(message: str) -> str:
    normalized = str(message or "").strip()
    if not normalized:
        return "Error: Team runtime request failed"
    if normalized.startswith("Agent ") and normalized.endswith(" is busy"):
        return (
            f"Error: {normalized}. Do not send another message yet; "
            "call get_agent_result later."
        )
    if normalized.startswith("Cannot close agent ") and normalized.endswith(": still busy"):
        return (
            f"Error: {normalized}. Wait for get_agent_result to return completed "
            "before closing."
        )
    return f"Error: {normalized}"


@tool
def spawn_agent(
    name: str,
    role: str = "teammate",
    system_prompt: str = "",
    config: RunnableConfig = cast(RunnableConfig, None),
) -> str:
    """创建新的 agent 实例

    Args:
        name: agent 名称
        role: agent 角色，默认 teammate
        system_prompt: 可选的角色补充提示

    Returns:
        agent_id
    """
    runtime = _get_runtime_for_tool_call(config)
    runtime_context = _get_current_runtime_context(config)
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
def send_message(
    agent_id: str,
    message: str,
    config: RunnableConfig = cast(RunnableConfig, None),
) -> str:
    """发送消息给 agent

    Args:
        agent_id: agent ID
        message: 消息内容

    Returns:
        状态信息
    """
    runtime = _get_runtime_for_tool_call(config)
    try:
        runtime.send_message(agent_id, message)
        return f"Message sent to agent {agent_id}"
    except ValueError as e:
        return _format_team_runtime_error(str(e))


@tool
def list_agents(config: RunnableConfig = cast(RunnableConfig, None)) -> str:
    """列出所有 agent 及其状态

    Returns:
        agent 列表（JSON 格式）
    """
    runtime = _get_runtime_for_tool_call(config)
    agents = runtime.list_agents()
    import json

    return json.dumps(agents, indent=2)


@tool
def get_agent_result(
    agent_id: str,
    config: RunnableConfig = cast(RunnableConfig, None),
) -> str:
    """获取 agent 的执行结果

    Args:
        agent_id: agent ID

    Returns:
        结构化执行结果（JSON 字符串）
    """
    runtime = _get_runtime_for_tool_call(config)
    try:
        return runtime.get_agent_result(agent_id)
    except ValueError as e:
        return _format_team_runtime_error(str(e))


@tool
def close_agent(
    agent_id: str,
    config: RunnableConfig = cast(RunnableConfig, None),
) -> str:
    """关闭 agent 实例

    Args:
        agent_id: agent ID

    Returns:
        状态信息
    """
    runtime = _get_runtime_for_tool_call(config)
    try:
        runtime.close_agent(agent_id)
        return f"Agent {agent_id} closed"
    except ValueError as e:
        return _format_team_runtime_error(str(e))
