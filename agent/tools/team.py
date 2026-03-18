"""Team 工具：提供 agent 团队管理能力"""

import uuid

from langchain_core.tools import tool

from ..team.runtime import TeamRuntime

# 全局 team runtime 实例
_team_runtime: TeamRuntime | None = None

# 默认模型（由 TeamMiddleware 注入）
_default_model = None


def get_team_runtime() -> TeamRuntime:
    """获取或创建 team runtime"""
    global _team_runtime
    if _team_runtime is None:
        team_id = str(uuid.uuid4())
        _team_runtime = TeamRuntime(team_id)
    return _team_runtime


@tool
def spawn_agent(name: str, system_prompt: str) -> str:
    """创建新的 agent 实例

    Args:
        name: agent 名称
        system_prompt: agent 的系统提示

    Returns:
        agent_id
    """
    runtime = get_team_runtime()
    # 子agent 使用基本配置，不包括 team 工具（避免递归）
    agent_id = runtime.spawn_agent(
        name=name,
        model=_default_model,
        system_prompt=system_prompt,
        tools=[],  # 子agent 只需要基本功能
    )
    return f"Agent spawned: {agent_id}"


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
        执行结果
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
