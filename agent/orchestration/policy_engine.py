import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..domain.orchestration import PolicyDecision
from ..domain.request_context import RequestContext

logger = logging.getLogger(__name__)


POLICY_ROUTER_INSTRUCTION = """
你是执行策略路由器。目标：在时延、成本、质量之间做平衡，决定是否启用 `plan` 和 `team`。

决策约束：
1) 如果任务可在一次检索+一次回答中完成，应偏向低成本路径
2) 若存在明确多步骤依赖，应开启 plan
3) 若存在并行子问题、交叉验证需求或多视角产出，应开启 team
4) 禁止因"用户语气复杂"误判复杂任务，必须基于任务结构判断

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
            "用户问题：\n{prompt}\n\n会话上下文摘要（可能为空）：\n{context_digest}",
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
            if not hasattr(structured, "invoke"):
                return None
            payload = _invoke_structured_compat(structured, input_payload)
        else:
            if not isinstance(payload, (dict, PolicyRouterOutput, PolicyDecision)) and hasattr(
                structured, "invoke"
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
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:
        return llm.with_structured_output(schema)


def _invoke_structured_compat(structured: Any, input_payload: dict[str, str]) -> Any:
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


def decide_execution_policy(
    prompt: str,
    *,
    llm: Any | None = None,
    context_digest: str = "",
) -> PolicyDecision:
    llm_decision = _route_with_llm(prompt, llm, context_digest=context_digest)
    if llm_decision is None:
        raise RuntimeError("Policy router LLM is required and failed to return a decision.")

    decision = llm_decision

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
    emit_info_log: bool = True,
) -> PolicyDecision:
    try:
        llm_decision = _route_with_llm(
            ctx.prompt,
            llm,
            context_digest=ctx.context_digest,
        )
    except Exception:
        logger.exception("Interceptor LLM routing failed")
        llm_decision = None

    if llm_decision is None:
        raise RuntimeError("Policy router LLM is required and failed to return a decision.")

    decision = llm_decision

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
