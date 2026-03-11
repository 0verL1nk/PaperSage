"""
tests/unit/test_request_context_interceptor.py

RequestContext 拦截器专项测试。
覆盖：
- RequestContext 构造与字段
- policy_engine.intercept() 的路由行为（含 context_digest）
- force_plan / force_team 覆盖语义
- LLM 不可用时重试并抛错
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.domain.orchestration import PolicyDecision
from agent.domain.request_context import RequestContext
from agent.orchestration.policy_engine import POLICY_ROUTER_MAX_RETRIES, intercept

# ---------------------------------------------------------------------------
# RequestContext 基本构造
# ---------------------------------------------------------------------------

def test_request_context_defaults():
    ctx = RequestContext(prompt="测试问题")
    assert ctx.prompt == "测试问题"
    assert ctx.context_digest == ""


def test_request_context_with_digest():
    ctx = RequestContext(prompt="测试问题", context_digest="上一轮讨论了 Transformer 架构")
    assert ctx.context_digest == "上一轮讨论了 Transformer 架构"


# ---------------------------------------------------------------------------
# policy_engine.intercept() - 路由行为
# ---------------------------------------------------------------------------

def test_intercept_simple_prompt_without_llm_raises():
    """未配置路由模型且无手动覆盖时应抛错。"""
    ctx = RequestContext(prompt="这篇论文的标题是什么")
    with pytest.raises(RuntimeError, match="Policy router LLM is required"):
        intercept(ctx, llm=None)


def test_intercept_complex_prompt_without_llm_raises():
    """未配置路由模型时，即使 prompt 复杂也应抛错。"""
    ctx = RequestContext(
        prompt=(
            "请完成以下任务：1. 拆解研究问题并提炼核心假设；"
            "2. 基于文档证据与外部文献交叉验证；"
            "3. 输出结构化结论并补充待验证风险点。请说明多目标冲突如何权衡？"
        )
    )
    with pytest.raises(RuntimeError, match="Policy router LLM is required"):
        intercept(ctx, llm=None)


def test_intercept_force_plan_overrides_everything():
    """force_plan=True 应强制开启 plan，无论其他信号如何。"""
    ctx = RequestContext(prompt="随便问一下")
    decision = intercept(ctx, llm=None, force_plan=True)
    assert decision.plan_enabled is True
    assert decision.source == "manual"


def test_intercept_force_team_auto_enables_plan():
    """force_team=True 应同时开启 plan（team 依赖 plan）。"""
    ctx = RequestContext(prompt="随便问一下")
    decision = intercept(ctx, llm=None, force_team=True)
    assert decision.team_enabled is True
    assert decision.plan_enabled is True
    assert decision.source == "manual"


def test_intercept_uses_llm_when_available():
    """当 LLM 可用时，结果 source 应为 'llm'。"""
    llm_decision = PolicyDecision(
        plan_enabled=True,
        team_enabled=False,
        reason="LLM 判断需要规划",
        confidence=0.85,
        source="llm",
    )
    ctx = RequestContext(prompt="请帮我分析这篇论文的方法论")
    with patch(
        "agent.orchestration.policy_engine._route_with_llm",
        return_value=llm_decision,
    ):
        decision = intercept(ctx, llm=MagicMock())
    assert decision.source == "llm"
    assert decision.plan_enabled is True


def test_intercept_raises_after_llm_retries_exhausted():
    """LLM 路由失败时应重试并最终抛错。"""
    ctx = RequestContext(prompt="这篇论文的核心结论")
    with patch(
        "agent.orchestration.policy_engine._route_with_llm",
        return_value=None,
    ) as mock_route:
        with pytest.raises(RuntimeError, match="Policy router failed after"):
            intercept(ctx, llm=MagicMock(spec=["with_structured_output"]))
    assert mock_route.call_count == POLICY_ROUTER_MAX_RETRIES + 1


def test_intercept_context_digest_passed_to_llm():
    """context_digest 应被透传给 _route_with_llm。"""
    ctx = RequestContext(
        prompt="继续上次的分析",
        context_digest="上轮已分析了实验设计",
    )
    llm_decision = PolicyDecision(
        plan_enabled=True,
        team_enabled=False,
        reason="llm-route",
        confidence=0.9,
        source="llm",
    )
    with patch(
        "agent.orchestration.policy_engine._route_with_llm",
        return_value=llm_decision,
    ) as mock_llm:
        intercept(ctx, llm=MagicMock())
    _, call_kwargs = mock_llm.call_args
    assert call_kwargs.get("context_digest") == "上轮已分析了实验设计"
