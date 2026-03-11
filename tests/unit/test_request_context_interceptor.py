from unittest.mock import MagicMock, patch

import pytest

from agent.domain.orchestration import PolicyDecision
from agent.domain.request_context import RequestContext
from agent.orchestration.policy_engine import intercept


def test_request_context_defaults():
    ctx = RequestContext(prompt="测试问题")
    assert ctx.prompt == "测试问题"
    assert ctx.context_digest == ""


def test_request_context_with_digest():
    ctx = RequestContext(prompt="测试问题", context_digest="上一轮讨论了 Transformer 架构")
    assert ctx.context_digest == "上一轮讨论了 Transformer 架构"


def test_intercept_simple_prompt_without_llm_raises():
    ctx = RequestContext(prompt="这篇论文的标题是什么")
    with pytest.raises(RuntimeError, match="Policy router LLM is required"):
        intercept(ctx, llm=None)


def test_intercept_complex_prompt_without_llm_raises():
    ctx = RequestContext(
        prompt=(
            "请完成以下任务：1. 拆解研究问题并提炼核心假设；"
            "2. 基于文档证据与外部文献交叉验证；"
            "3. 输出结构化结论并补充待验证风险点。请说明多目标冲突如何权衡？"
        )
    )
    with pytest.raises(RuntimeError, match="Policy router LLM is required"):
        intercept(ctx, llm=None)


def test_intercept_uses_llm_when_available():
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


def test_intercept_raises_when_llm_returns_none():
    ctx = RequestContext(prompt="这篇论文的核心结论")
    with patch(
        "agent.orchestration.policy_engine._route_with_llm",
        return_value=None,
    ):
        with pytest.raises(RuntimeError, match="Policy router LLM is required"):
            intercept(ctx, llm=MagicMock(spec=["with_structured_output"]))


def test_intercept_context_digest_passed_to_llm():
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
