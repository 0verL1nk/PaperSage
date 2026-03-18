"""Team 中间件：提供 agent 团队管理能力"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from ..tools import team as team_tools


class TeamMiddleware:
    """Team 中间件：按需提供 team 工具"""

    def __init__(self, default_model: Any):
        self.default_model = default_model
        # 注入默认配置到 team_tools 模块
        team_tools._default_model = default_model

    def __call__(self, state: dict, config: RunnableConfig) -> dict:
        """中间件处理逻辑"""
        # 从 config 中提取 thread_id 并设置为当前 session
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")
            if thread_id:
                team_tools.set_current_session(thread_id)

        return state

    @property
    def tools(self) -> list[Any]:
        """返回 team 工具"""
        return [
            team_tools.spawn_agent,
            team_tools.send_message,
            team_tools.list_agents,
            team_tools.get_agent_result,
            team_tools.close_agent,
        ]
