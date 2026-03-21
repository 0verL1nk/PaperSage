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

    def __init__(self, default_model: Any, dependencies: Any | None = None):
        super().__init__()
        self.default_model = default_model
        self.dependencies = dependencies
        self._needs_team = False

    def before_model(  # type: ignore[override]
        self, state: TeamState, runtime: Runtime, config: RunnableConfig | None = None
    ) -> dict[str, Any] | None:
        """在模型调用前设置当前 session 并检测 needs_team 标志"""
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")
            if thread_id:
                team_tools.set_current_session(thread_id)
                team_tools.set_session_runtime_context(
                    thread_id,
                    default_model=self.default_model,
                    dependencies=self.dependencies,
                )

        self._needs_team = state.get("needs_team", False)
        return None

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Wrap model call to inject team guidance when needed."""
        if self._needs_team:
            messages = getattr(request, "messages", [])
            if messages:
                guidance = (
                    "【重要提示】这是一个需要多角色协作的复杂任务,建议由你控制团队节奏:\n\n"
                    "1. 先判断是否真的需要团队分工,不要机械地创建 agent\n"
                    "2. 如需协作,由你来决定是否分派、分派给谁、是否并行\n"
                    "3. 可使用 spawn_agent / send_message / get_agent_result / list_agents / close_agent 作为协作工具\n"
                    "4. spawn_agent 默认创建 teammate,可通过 role 指定 reviewer 等角色,必要时再补充 system_prompt\n"
                    "5. 若 get_agent_result 返回 busy,说明 teammate 仍在处理上一条任务,不要重复 send_message 或 close_agent\n"
                    "6. 需要等待时可调用 sleep(seconds, reason) 稍作阻塞,之后再决定是否继续查询结果\n"
                    "7. teammate 结果只是中间产物,最终答复仍由你输出\n\n"
                    "请由你来决定是否分派任务,不要把当前对话 ownership 交给 teammate。"
                )
                result = list(messages)
                result.insert(-1, SystemMessage(content=guidance))
                request.messages = result
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
