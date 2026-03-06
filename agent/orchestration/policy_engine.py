import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..domain.orchestration import PolicyDecision
from ..domain.request_context import RequestContext
from ..settings import load_agent_settings

logger = logging.getLogger(__name__)


POLICY_ROUTER_INSTRUCTION = """
你是执行策略路由器。目标：在时延、成本、质量之间做平衡，决定是否启用 `plan` 和 `team`。

决策约束：
1) 如果任务可在一次检索+一次回答中完成，应偏向低成本路径
2) 若存在明确多步骤依赖，应开启 plan
3) 若存在并行子问题、交叉验证需求或多视角产出，应开启 team
4) 禁止因“用户语气复杂”误判复杂任务，必须基于任务结构判断

判定原则：
1) 简单单跳问答：plan=false, team=false
2) 中等复杂任务（需要分步但不需要多人协作）：plan=true, team=false
3) 高复杂任务（多目标、多约束、需要交叉验证/结构化产出）：plan=true, team=true
"""


class PolicyRouterOutput(BaseModel):
    plan_enabled: bool = Field(description="是否启用 plan 阶段")
    team_enabled: bool = Field(description="是否启用 team 协作阶段")
    reason: str = Field(min_length=1, description="一句话说明决策依据")
    confidence: float = Field(ge=0.0, le=1.0, description="路由置信度，范围 [0,1]")


POLICY_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", POLICY_ROUTER_INSTRUCTION),
        (
            "human",
            "用户问题：\n{prompt}\n\n"
            "会话上下文摘要（可能为空）：\n{context_digest}",
        ),
    ]
)


def _route_with_llm(
    prompt: str,
    llm: Any,
    *,
    context_digest: str = "",
) -> PolicyDecision | None:
    if llm is None:
        return None
    try:
        if not hasattr(llm, "with_structured_output"):
            return None
        input_payload = {
            "prompt": prompt,
            "context_digest": context_digest.strip() or "(none)",
        }
        structured = _with_structured_output(llm, PolicyRouterOutput)
        try:
            payload = (POLICY_ROUTER_PROMPT | structured).invoke(input_payload)
        except Exception:
            # Compatibility fallback for lightweight stubs/mocks that cannot compose with prompt runnable.
            if not hasattr(structured, "invoke"):
                return None
            payload = structured.invoke(input_payload)
        else:
            if (
                not isinstance(payload, (dict, PolicyRouterOutput, PolicyDecision))
                and hasattr(structured, "invoke")
            ):
                payload = structured.invoke(input_payload)
        if isinstance(payload, PolicyDecision):
            return payload
        if isinstance(payload, dict):
            payload = PolicyRouterOutput.model_validate(payload)
        if not isinstance(payload, PolicyRouterOutput):
            return None
        return PolicyDecision(
            plan_enabled=payload.plan_enabled,
            team_enabled=payload.team_enabled,
            reason=payload.reason.strip() or "LLM 路由未提供原因。",
            confidence=float(payload.confidence),
            source="llm",
        )
    except Exception:
        logger.exception("Policy router LLM failed, fallback to heuristic")
        return None


def _with_structured_output(llm: Any, schema: type[BaseModel]) -> Any:
    # Prefer function-calling path to avoid provider parse payload warnings.
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:
        return llm.with_structured_output(schema)


def _heuristic_route(prompt: str, *, context_digest: str = "") -> PolicyDecision:
    settings = load_agent_settings()
    normalized = prompt.strip()
    text_len = len(normalized)
    sentence_count = len(
        [part for part in re.split(r"[。！？!?;\n]+", normalized) if part.strip()]
    )
    enumerated_steps = len(re.findall(r"\b\d+[\.\)]\s*", normalized))
    comma_count = normalized.count("，") + normalized.count(",")
    question_mark_count = normalized.count("?") + normalized.count("？")
    conjunction_count = (
        normalized.count("并")
        + normalized.count("且")
        + normalized.count("同时")
        + normalized.lower().count(" and ")
    )

    complexity_score = 0
    if text_len >= settings.agent_policy_text_len_medium:
        complexity_score += 1
    if text_len >= settings.agent_policy_text_len_high:
        complexity_score += 1
    if sentence_count >= settings.agent_policy_sentence_threshold:
        complexity_score += 1
    if comma_count >= settings.agent_policy_comma_threshold:
        complexity_score += 1
    if question_mark_count >= settings.agent_policy_question_threshold:
        complexity_score += 1
    if enumerated_steps >= settings.agent_policy_enum_steps_threshold:
        complexity_score += 1
    if conjunction_count >= settings.agent_policy_conjunction_threshold:
        complexity_score += 1
    digest = context_digest.strip()
    if digest:
        if len(digest) >= settings.agent_policy_context_chars_threshold:
            complexity_score += 1
        digest_lines = len([line for line in digest.splitlines() if line.strip()])
        if digest_lines >= settings.agent_policy_context_lines_threshold:
            complexity_score += 1

    plan_threshold = max(1, settings.agent_policy_score_plan)
    team_threshold = max(plan_threshold + 1, settings.agent_policy_score_team)

    if complexity_score >= team_threshold:
        return PolicyDecision(
            plan_enabled=True,
            team_enabled=True,
            reason="结构复杂度高（长度/句数/步骤密度），启用 plan + team。",
            source="heuristic",
        )

    if complexity_score >= plan_threshold:
        return PolicyDecision(
            plan_enabled=True,
            team_enabled=False,
            reason="结构复杂度中等，启用 plan。",
            source="heuristic",
        )

    return PolicyDecision(
        plan_enabled=False,
        team_enabled=False,
        reason="结构复杂度低，走主 agent 快速路径。",
        source="heuristic",
    )


_LOW_CONFIDENCE_THRESHOLD = 0.6


def _merge_conservative(
    llm_decision: PolicyDecision,
    heuristic_decision: PolicyDecision,
) -> PolicyDecision:
    """低置信度时保守合并：两者均同意才开启 plan/team。"""
    plan_enabled = llm_decision.plan_enabled and heuristic_decision.plan_enabled
    team_enabled = llm_decision.team_enabled and heuristic_decision.team_enabled
    reason = (
        f"低置信度合并（LLM conf={llm_decision.confidence:.2f}）"
        f"：LLM={llm_decision.reason} | 启发式={heuristic_decision.reason}"
    )
    return PolicyDecision(
        plan_enabled=plan_enabled,
        team_enabled=team_enabled,
        reason=reason,
        confidence=llm_decision.confidence,
        source="merged",
    )


def decide_execution_policy(
    prompt: str,
    *,
    llm: Any | None = None,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    context_digest: str = "",
) -> PolicyDecision:
    llm_decision = _route_with_llm(prompt, llm, context_digest=context_digest)
    heuristic_decision = _heuristic_route(prompt, context_digest=context_digest)

    if llm_decision is None:
        decision = heuristic_decision
    elif (
        llm_decision.confidence is not None
        and llm_decision.confidence < _LOW_CONFIDENCE_THRESHOLD
    ):
        # LLM 路由置信度不足，与启发式保守合并
        logger.info(
            "Policy router low confidence (%.2f), merging with heuristic: llm=%s heuristic=%s",
            llm_decision.confidence,
            llm_decision.reason,
            heuristic_decision.reason,
        )
        decision = _merge_conservative(llm_decision, heuristic_decision)
    else:
        decision = llm_decision

    if force_plan is not None:
        decision = PolicyDecision(
            plan_enabled=bool(force_plan),
            team_enabled=decision.team_enabled,
            reason=f"手动覆盖 plan={bool(force_plan)} | {decision.reason}",
            confidence=decision.confidence,
            source="manual",
        )
    if force_team is not None:
        decision = PolicyDecision(
            plan_enabled=decision.plan_enabled,
            team_enabled=bool(force_team),
            reason=f"手动覆盖 team={bool(force_team)} | {decision.reason}",
            confidence=decision.confidence,
            source="manual",
        )

    if decision.team_enabled and not decision.plan_enabled:
        decision = PolicyDecision(
            plan_enabled=True,
            team_enabled=True,
            reason=f"{decision.reason}（team 启用时自动开启 plan）",
            confidence=decision.confidence,
            source=decision.source,
        )
    return decision


def intercept(
    ctx: RequestContext,
    *,
    llm: Any | None = None,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    emit_info_log: bool = True,
) -> PolicyDecision:
    """请求前拦截器：携带结构化 RequestContext 统一决策执行策略。

    以 prompt + context_digest 作为路由输入，优先走 LLM 判断，
    LLM 不可用或失败时降级到启发式评分。
    """
    try:
        llm_decision = _route_with_llm(
            ctx.prompt,
            llm,
            context_digest=ctx.context_digest,
        )
    except Exception:
        logger.exception("Interceptor LLM routing failed, fallback to heuristic")
        llm_decision = None
    heuristic_decision = _heuristic_route(
        ctx.prompt,
        context_digest=ctx.context_digest,
    )

    if llm_decision is None:
        decision = heuristic_decision
    elif (
        llm_decision.confidence is not None
        and llm_decision.confidence < _LOW_CONFIDENCE_THRESHOLD
    ):
        logger.info(
            "Interceptor low confidence (%.2f), merging with heuristic: llm=%s heuristic=%s",
            llm_decision.confidence,
            llm_decision.reason,
            heuristic_decision.reason,
        )
        decision = _merge_conservative(llm_decision, heuristic_decision)
    else:
        decision = llm_decision

    if force_plan is not None:
        decision = PolicyDecision(
            plan_enabled=bool(force_plan),
            team_enabled=decision.team_enabled,
            reason=f"手动覆盖 plan={bool(force_plan)} | {decision.reason}",
            confidence=decision.confidence,
            source="manual",
        )
    if force_team is not None:
        decision = PolicyDecision(
            plan_enabled=decision.plan_enabled,
            team_enabled=bool(force_team),
            reason=f"手动覆盖 team={bool(force_team)} | {decision.reason}",
            confidence=decision.confidence,
            source="manual",
        )

    if decision.team_enabled and not decision.plan_enabled:
        decision = PolicyDecision(
            plan_enabled=True,
            team_enabled=True,
            reason=f"{decision.reason}（team 启用时自动开启 plan）",
            confidence=decision.confidence,
            source=decision.source,
        )

    log_fn = logger.info if emit_info_log else logger.debug
    log_fn(
        "Interceptor decision: plan=%s team=%s source=%s reason=%s",
        decision.plan_enabled,
        decision.team_enabled,
        decision.source,
        decision.reason,
    )
    return decision
