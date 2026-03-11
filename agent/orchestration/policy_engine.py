import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..domain.orchestration import PolicyDecision
from ..domain.request_context import RequestContext

logger = logging.getLogger(__name__)
POLICY_ROUTER_MAX_RETRIES = 2


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
            payload = _invoke_structured_compat(structured, input_payload)
        else:
            if (
                not isinstance(payload, (dict, PolicyRouterOutput, PolicyDecision))
                and hasattr(structured, "invoke")
            ):
                payload = _invoke_structured_compat(structured, input_payload)
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
        logger.exception("Policy router LLM failed")
        return None


def _with_structured_output(llm: Any, schema: type[BaseModel]) -> Any:
    # Prefer function-calling path to avoid provider parse payload warnings.
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:
        return llm.with_structured_output(schema)


def _invoke_structured_compat(structured: Any, input_payload: dict[str, str]) -> Any:
    """Invoke structured output runnable with cross-version compatible inputs."""
    prompt_value = POLICY_ROUTER_PROMPT.invoke(input_payload)
    candidates: list[Any] = [prompt_value]
    try:
        candidates.append(prompt_value.to_messages())
    except Exception:
        pass
    try:
        candidates.append(prompt_value.to_string())
    except Exception:
        pass
    # Keep dict as the final fallback for simple local mocks.
    candidates.append(input_payload)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return structured.invoke(candidate)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return None


def _manual_only_decision(
    *,
    force_plan: bool | None,
    force_team: bool | None,
) -> PolicyDecision | None:
    if force_plan is None and force_team is None:
        return None
    return PolicyDecision(
        plan_enabled=False,
        team_enabled=False,
        reason="仅应用手动路由覆盖。",
        confidence=1.0,
        source="manual",
    )


def _resolve_policy_decision(
    *,
    prompt: str,
    llm: Any | None,
    context_digest: str,
    max_retries: int = POLICY_ROUTER_MAX_RETRIES,
) -> PolicyDecision:
    if llm is None:
        raise RuntimeError("Policy router LLM is required when no manual override is provided.")
    retry_budget = max(0, int(max_retries))
    total_attempts = retry_budget + 1
    for attempt in range(1, total_attempts + 1):
        llm_decision = _route_with_llm(prompt, llm, context_digest=context_digest)
        if llm_decision is not None:
            return llm_decision
        if attempt < total_attempts:
            logger.warning(
                "Policy router returned empty decision, retrying (%s/%s)",
                attempt,
                total_attempts,
            )
    raise RuntimeError(
        f"Policy router failed after {total_attempts} attempts."
    )


def _apply_manual_overrides(
    decision: PolicyDecision,
    *,
    force_plan: bool | None,
    force_team: bool | None,
) -> PolicyDecision:
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
    return decision


def _ensure_team_plan_invariant(decision: PolicyDecision) -> PolicyDecision:
    if decision.team_enabled and not decision.plan_enabled:
        return PolicyDecision(
            plan_enabled=True,
            team_enabled=True,
            reason=f"{decision.reason}（team 启用时自动开启 plan）",
            confidence=decision.confidence,
            source=decision.source,
        )
    return decision


def decide_execution_policy(
    prompt: str,
    *,
    llm: Any | None = None,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    context_digest: str = "",
) -> PolicyDecision:
    manual_decision = _manual_only_decision(
        force_plan=force_plan,
        force_team=force_team,
    )
    if manual_decision is not None:
        return _ensure_team_plan_invariant(
            _apply_manual_overrides(
                manual_decision,
                force_plan=force_plan,
                force_team=force_team,
            )
        )
    decision = _resolve_policy_decision(
        prompt=prompt,
        llm=llm,
        context_digest=context_digest,
    )
    return _ensure_team_plan_invariant(decision)


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
    LLM 失败会自动重试，超过重试次数后抛出异常。
    """
    manual_decision = _manual_only_decision(
        force_plan=force_plan,
        force_team=force_team,
    )
    if manual_decision is not None:
        decision = _ensure_team_plan_invariant(
            _apply_manual_overrides(
                manual_decision,
                force_plan=force_plan,
                force_team=force_team,
            )
        )
    else:
        decision = _resolve_policy_decision(
            prompt=ctx.prompt,
            llm=llm,
            context_digest=ctx.context_digest,
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
