"""
tests/unit/test_request_context_interceptor.py

RequestContext 拦截器专项测试。
覆盖：
- RequestContext 构造与字段
- policy_engine.intercept() 的路由行为（含 context_digest）
- force_plan / force_team 覆盖语义
- LLM 不可用时降级到启发式
"""
from unittest.mock import patch, MagicMock

from agent.domain.request_context import RequestContext
from agent.orchestration.policy_engine import intercept
from agent.domain.orchestration import PolicyDecision


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

def test_intercept_simple_prompt_no_context_stays_react():
    """简单问答 + 无 context_digest → 应保持 ReAct。"""
    ctx = RequestContext(prompt="这篇论文的标题是什么")
    decision = intercept(ctx, llm=None)
    assert decision.plan_enabled is False
    assert decision.team_enabled is False


def test_intercept_complex_prompt_routes_up():
    """复杂结构化任务 prompt → 启发式应路由到 plan 或以上。"""
    ctx = RequestContext(
        prompt=(
            "请完成以下任务：1. 拆解研究问题并提炼核心假设；"
            "2. 基于文档证据与外部文献交叉验证；"
            "3. 输出结构化结论并补充待验证风险点。请说明多目标冲突如何权衡？"
        )
    )
    decision = intercept(ctx, llm=None)
    assert decision.plan_enabled is True


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


def test_intercept_falls_back_to_heuristic_on_llm_error():
    """LLM 调用失败时应静默降级到启发式，source='heuristic'。"""
    ctx = RequestContext(prompt="这篇论文的核心结论")
    with patch(
        "agent.orchestration.policy_engine._route_with_llm",
        side_effect=Exception("LLM timeout"),
    ):
        decision = intercept(ctx, llm=MagicMock())
    assert decision.source == "heuristic"
    assert isinstance(decision.plan_enabled, bool)


def test_intercept_context_digest_passed_to_llm():
    """context_digest 应被透传给 _route_with_llm。"""
    ctx = RequestContext(
        prompt="继续上次的分析",
        context_digest="上轮已分析了实验设计",
    )
    with patch(
        "agent.orchestration.policy_engine._route_with_llm",
        return_value=None,
    ) as mock_llm:
        intercept(ctx, llm=MagicMock())
    _, call_kwargs = mock_llm.call_args
    assert call_kwargs.get("context_digest") == "上轮已分析了实验设计"
