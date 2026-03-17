import logging
from typing import Any

from ..domain.request_context import RequestContext
from .coordinator import WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN, WORKFLOW_REACT

logger = logging.getLogger(__name__)

WORKFLOW_LABELS = {
    WORKFLOW_REACT: "ReAct（Tool+Memory）",
    WORKFLOW_PLAN_ACT: "Plan-Act（A2A协调）",
    WORKFLOW_PLAN_ACT_REPLAN: "Plan-Act-RePlan（A2A协调）",
}


ROUTER_INSTRUCTION = """
你是工作流路由器。请在以下模式中选择最合适的一种：
- react: 简单问答、单跳检索、快速事实确认
- plan_act: 中等复杂任务，需要先规划再执行
- plan_act_replan: 高复杂度任务，需要规划-执行-复核-重规划

判定要求：
1) 只基于任务结构复杂度、目标数量、约束冲突程度做判断
2) 不要依赖词面匹配，不要把"专业术语多"误判为高复杂任务

仅返回严格 JSON，不要额外文本：
{"mode":"react|plan_act|plan_act_replan","reason":"简短原因","confidence":0.0}
"""


def _policy_to_workflow_mode(plan_enabled: bool, team_enabled: bool) -> str:
    """将 PolicyDecision 的 plan/team 标志映射回三档工作流模式。"""
    if team_enabled:
        return WORKFLOW_PLAN_ACT_REPLAN
    if plan_enabled:
        return WORKFLOW_PLAN_ACT
    return WORKFLOW_REACT


def auto_select_workflow_mode(
    prompt_or_ctx: str | RequestContext,
    coordinator: Any | None = None,
) -> tuple[str, str]:
    """根据 prompt 或 RequestContext 自动选择工作流模式。

    注意: 在agent-centric模式下,此函数已废弃。
    Agent通过工具(create_plan, activate_team_mode)自主决定工作流。
    此函数保留仅为API兼容性,默认返回react模式。
    """
    logger.warning(
        "auto_select_workflow_mode is deprecated in agent-centric mode. "
        "Agent decides workflow via tools (create_plan, activate_team_mode)."
    )
    return WORKFLOW_REACT, "agent-centric mode: agent decides via tools"
