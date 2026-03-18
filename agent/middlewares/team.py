"""Team 中间件：提供 agent 团队管理能力"""

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

from ..tools import team as team_tools
from .types import AgentState


class TeamState(AgentState):
    """Team middleware state schema."""

    needs_team: NotRequired[bool]
    """Flag indicating if team mode should be activated."""


class TeamMiddleware(AgentMiddleware):
    """Team 中间件：按需提供 team 工具"""

    state_schema = TeamState

    def __init__(self, default_model: Any):
        super().__init__()
        self.default_model = default_model
        # 注入默认配置到 team_tools 模块
        team_tools._default_model = default_model
        self._needs_team = False

    def before_model(  # type: ignore[override]
        self, state: TeamState, runtime: Runtime, config: RunnableConfig | None = None
    ) -> dict[str, Any] | None:
        """在模型调用前设置当前 session 并检测 needs_team 标志"""
        # 从 config 中提取 thread_id 并设置为当前 session
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")
            if thread_id:
                team_tools.set_current_session(thread_id)

        # 检测 needs_team 标志
        self._needs_team = state.get("needs_team", False)
        return None

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Wrap model call to inject team guidance when needed.

        Args:
            request: Model request
            handler: Next handler in chain

        Returns:
            Model response
        """
        if self._needs_team:
            messages = getattr(request, "messages", [])
            if messages:
                guidance = (
                    "【重要提示】这是一个需要多角色协作的复杂任务,建议使用团队模式:\n\n"
                    "1. 使用 spawn_agent 工具创建专业 agent(如 researcher, reviewer, writer 等)\n"
                    "2. 使用 send_message 工具分配任务给各个 agent\n"
                    "3. 使用 get_agent_result 工具获取执行结果\n"
                    "4. 使用 list_agents 工具查看团队状态\n"
                    "5. 完成后使用 close_agent 工具关闭 agent\n\n"
                    "请先创建团队并分配任务,不要直接回答。"
                )
                result = list(messages)
                result.insert(-1, SystemMessage(content=guidance))
                request.messages = result
            # Reset flag after use
            self._needs_team = False

        return handler(request)

    @property
    def tools(self) -> Sequence[BaseTool]:  # type: ignore[override]
        """返回 team 工具"""
        return [
            team_tools.spawn_agent,
            team_tools.send_message,
            team_tools.list_agents,
            team_tools.get_agent_result,
            team_tools.close_agent,
        ]
