from unittest.mock import MagicMock, patch

from agent.a2a.coordinator import WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN, WORKFLOW_REACT
from agent.a2a.router import _policy_to_workflow_mode, auto_select_workflow_mode
from agent.domain.orchestration import PolicyDecision
from agent.orchestration.policy_engine import decide_execution_policy

# ---------------------------------------------------------------------------
# _policy_to_workflow_mode 映射
# ---------------------------------------------------------------------------

def test_policy_to_mode_team_enabled():
    assert _policy_to_workflow_mode(plan_enabled=True, team_enabled=True) == WORKFLOW_PLAN_ACT_REPLAN


def test_policy_to_mode_plan_only():
    assert _policy_to_workflow_mode(plan_enabled=True, team_enabled=False) == WORKFLOW_PLAN_ACT


def test_policy_to_mode_react():
    assert _policy_to_workflow_mode(plan_enabled=False, team_enabled=False) == WORKFLOW_REACT


# ---------------------------------------------------------------------------
# 启发式路由（无 coordinator / llm）
# ---------------------------------------------------------------------------

def test_router_uses_replan_for_high_complexity_prompt():
    mode, _reason = auto_select_workflow_mode(
        "请先完成三步：1) 提炼问题与约束；2) 交叉验证证据来源；3) 形成结论并给出风险清单。"
        "同时回答两个问题：当前证据是否足够？若不足应如何补证？"
    )
    assert mode == WORKFLOW_PLAN_ACT_REPLAN


def test_router_uses_plan_act_for_medium_complexity_prompt():
    mode, _reason = auto_select_workflow_mode(
        "请完成以下任务：1. 拆解研究问题并提炼核心假设；2. 基于文档证据与外部文献交叉验证；"
        "3. 输出结构化结论并补充待验证风险点。请说明多目标冲突如何权衡？"
    )
    assert mode == WORKFLOW_PLAN_ACT


def test_router_uses_react_for_simple_fact_prompt():
    mode, _reason = auto_select_workflow_mode("这篇论文的核心结论是什么")
    assert mode == WORKFLOW_REACT


# ---------------------------------------------------------------------------
# LLM 路由（mock decide_execution_policy 直接返回指定 PolicyDecision）
# ---------------------------------------------------------------------------

def test_router_prefers_llm_plan_act_decision():
    decision = PolicyDecision(
        plan_enabled=True, team_enabled=False,
        reason="需要先规划", confidence=0.88, source="llm",
    )
    with patch("agent.a2a.router.intercept", return_value=decision):
        mode, reason = auto_select_workflow_mode("请帮我处理这个问题", coordinator=MagicMock())
    assert mode == WORKFLOW_PLAN_ACT
    assert "规划" in reason


def test_router_prefers_llm_team_decision():
    decision = PolicyDecision(
        plan_enabled=True, team_enabled=True,
        reason="多目标交叉验证", confidence=0.91, source="llm",
    )
    with patch("agent.a2a.router.intercept", return_value=decision):
        mode, _reason = auto_select_workflow_mode("请帮我处理这个问题", coordinator=MagicMock())
    assert mode == WORKFLOW_PLAN_ACT_REPLAN


def test_router_fallbacks_to_heuristic_when_llm_raises():
    coordinator = MagicMock()
    coordinator.llm = MagicMock()
    coordinator.llm.with_structured_output = MagicMock(side_effect=Exception("LLM unavailable"))
    # 高复杂度 prompt，启发式应路由到 plan_act 或以上
    mode, _reason = auto_select_workflow_mode(
        "请完成以下任务：1. 拆解研究问题并提炼核心假设；2. 基于文档证据与外部文献交叉验证；"
        "3. 输出结构化结论并补充待验证风险点。请说明多目标冲突如何权衡？",
        coordinator=coordinator,
    )
    assert mode in {WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN}


# ---------------------------------------------------------------------------
# decide_execution_policy 的 force 覆盖语义
# ---------------------------------------------------------------------------

def test_decide_policy_force_plan_overrides_llm():
    decision = decide_execution_policy("随便问一下", llm=None, force_plan=True)
    assert decision.plan_enabled is True
    assert decision.source == "manual"


def test_decide_policy_force_team_auto_enables_plan():
    decision = decide_execution_policy("随便问一下", llm=None, force_team=True)
    assert decision.plan_enabled is True
    assert decision.team_enabled is True
