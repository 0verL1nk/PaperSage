import json
import re
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from ..capabilities import build_agent_tools
from ..domain.orchestration import (
    TeamExecution,
    TeamRole,
    TraceContext,
    TraceEvent,
    build_trace_event,
    create_trace_context,
)
from ..stream import extract_result_text


ROLE_ROUTER_INSTRUCTION = """
你是团队角色规划器。请为当前任务设计若干互补角色，避免职责重复。

要求：
1) 角色名使用英文小写短词（如 researcher/reviewer/writer）
2) 每个 goal 必须可执行、可检查
3) 角色之间职责互补，不要同义重复
4) 每个角色必须直接服务于“高质量最终答案”，不要创建泛化管理角色
"""


class RoleOutputItem(BaseModel):
    name: str = Field(min_length=1, description="英文小写角色名")
    goal: str = Field(min_length=1, description="角色执行目标")


class RoleRouterOutput(BaseModel):
    roles: list[RoleOutputItem] = Field(min_length=1)


ROLE_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROLE_ROUTER_INSTRUCTION),
        ("human", "用户问题：\n{prompt}\n\n最多可分配角色数：{max_members}"),
    ]
)


def _fallback_roles() -> list[TeamRole]:
    return [
        TeamRole(name="researcher", goal="检索并提取与问题最相关的证据"),
        TeamRole(name="reviewer", goal="检查证据一致性与潜在遗漏"),
        TeamRole(name="writer", goal="整理成简洁结构化结论"),
    ]


ROUND_ROUTER_INSTRUCTION = """
你是团队协作轮次规划器。请判断是否需要多轮团队协作。

判定原则：
1) 若任务信息明确、目标单一、一次协作可完成 -> rounds=1
2) 若任务存在多目标冲突、需要交叉校验或高风险结论 -> rounds=2

输出要求：rounds 为正整数，reason 给出一句话理由。
"""


class RoundRouterOutput(BaseModel):
    rounds: int = Field(ge=1)
    reason: str = Field(min_length=1)


ROUND_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROUND_ROUTER_INSTRUCTION),
        ("human", "用户问题：\n{prompt}\n\n允许的最大协作轮次：{max_rounds}"),
    ]
)

ROLE_NAME_PATTERN = re.compile(r"[^a-z0-9_-]+")
RESERVED_ROLE_NAMES = {
    "user",
    "leader",
    "planner",
    "policy_engine",
    "team_runtime",
    "coordinator",
}


def _extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    return text[start : end + 1]


def _llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _normalize_role_name(raw_name: str, *, fallback_index: int) -> str:
    candidate = str(raw_name or "").strip().lower()
    candidate = ROLE_NAME_PATTERN.sub("_", candidate).strip("_")
    if candidate in RESERVED_ROLE_NAMES:
        candidate = f"{candidate}_role"
    if not candidate:
        return f"role_{fallback_index}"
    return candidate[:32]


def _normalize_roles(raw_roles: list[TeamRole], *, max_members: int) -> list[TeamRole]:
    normalized: list[TeamRole] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_roles, start=1):
        if len(normalized) >= max_members:
            break
        name = _normalize_role_name(item.name, fallback_index=index)
        if name in seen:
            continue
        seen.add(name)
        goal = str(item.goal or "").strip() or "完成分配任务"
        normalized.append(TeamRole(name=name, goal=goal))
    return normalized


def _decide_team_rounds(
    *,
    prompt: str,
    llm: Any | None,
    max_rounds: int,
) -> int:
    bounded_rounds = max(1, int(max_rounds))
    if bounded_rounds <= 1 or llm is None:
        return 1
    try:
        if not hasattr(llm, "with_structured_output"):
            return 1
        chain = ROUND_ROUTER_PROMPT | _with_structured_output(llm, RoundRouterOutput)
        payload = chain.invoke({"prompt": prompt, "max_rounds": bounded_rounds})
        if isinstance(payload, dict):
            payload = RoundRouterOutput.model_validate(payload)
        if not isinstance(payload, RoundRouterOutput):
            return 1
        rounds = int(payload.rounds)
        return max(1, min(rounds, bounded_rounds))
    except Exception:
        try:
            result = llm.invoke(
                f"{ROUND_ROUTER_INSTRUCTION}\n\n用户问题：{prompt}\n允许的最大协作轮次：{bounded_rounds}"
            )
            text = _llm_content_to_text(getattr(result, "content", result))
            block = _extract_json_block(text)
            if not block:
                return 1
            payload = json.loads(block)
            if not isinstance(payload, dict):
                return 1
            rounds = int(payload.get("rounds", 1))
            return max(1, min(rounds, bounded_rounds))
        except Exception:
            return 1


def generate_dynamic_roles(
    prompt: str,
    llm: Any | None,
    *,
    max_members: int = 3,
) -> list[TeamRole]:
    bounded_members = max(1, int(max_members))
    if llm is None:
        return _fallback_roles()[:bounded_members]
    try:
        if not hasattr(llm, "with_structured_output"):
            return _fallback_roles()[:bounded_members]
        chain = ROLE_ROUTER_PROMPT | _with_structured_output(llm, RoleRouterOutput)
        payload = chain.invoke({"prompt": prompt, "max_members": bounded_members})
        if isinstance(payload, dict):
            payload = RoleRouterOutput.model_validate(payload)
        if not isinstance(payload, RoleRouterOutput):
            return _fallback_roles()[:bounded_members]
        roles: list[TeamRole] = []
        for item in payload.roles[:bounded_members]:
            name = str(item.name or "").strip().lower()
            goal = str(item.goal or "").strip()
            if not name:
                continue
            roles.append(TeamRole(name=name, goal=goal or "完成分配任务"))
        return roles if roles else _fallback_roles()[:bounded_members]
    except Exception:
        try:
            result = llm.invoke(
                f"{ROLE_ROUTER_INSTRUCTION}\n\n用户问题：{prompt}\n最多可分配角色数：{bounded_members}"
            )
            text = _llm_content_to_text(getattr(result, "content", result))
            block = _extract_json_block(text)
            if not block:
                return _fallback_roles()[:bounded_members]
            payload = json.loads(block)
            if not isinstance(payload, dict):
                return _fallback_roles()[:bounded_members]
            roles_raw = payload.get("roles")
            roles: list[TeamRole] = []
            if isinstance(roles_raw, list):
                for item in roles_raw[:bounded_members]:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip().lower()
                    goal = str(item.get("goal") or "").strip()
                    if not name:
                        continue
                    roles.append(TeamRole(name=name, goal=goal or "完成分配任务"))
            return roles if roles else _fallback_roles()[:bounded_members]
        except Exception:
            return _fallback_roles()[:bounded_members]


def _with_structured_output(llm: Any, schema: type[BaseModel]) -> Any:
    # Prefer function-calling path to avoid provider parse payload warnings.
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:
        return llm.with_structured_output(schema)


def _invoke_role_agent(
    *,
    llm: Any,
    role: TeamRole,
    prompt: str,
    plan_text: str,
    notes: str,
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None,
) -> str:
    system_prompt = (
        f"你是团队中的 {role.name}。\n"
        f"角色目标：{role.goal}\n\n"
        "执行约束：\n"
        "1) 优先使用文档检索工具，结论必须尽量有证据支撑\n"
        "2) 只输出面向队长的执行结果，不输出思考过程\n"
        "3) 输出格式必须包含以下三段：\n"
        "[结论]\n"
        "[证据]\n"
        "[待验证点]"
    )
    tools = build_agent_tools(
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
    )
    member_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    task_prompt = (
        f"用户问题：{prompt}\n"
        f"当前计划：\n{plan_text or '无'}\n"
        f"当前团队笔记：\n{notes or '无'}\n"
        "请输出你作为该角色的执行结果。"
    )
    result = member_agent.invoke(
        {"messages": [{"role": "user", "content": task_prompt}]},
        config={"configurable": {"thread_id": f"team:{role.name}"}},
    )
    return extract_result_text(result) if isinstance(result, dict) else str(result)


def run_team_tasks(
    *,
    prompt: str,
    plan_text: str,
    llm: Any | None,
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None,
    max_members: int = 3,
    max_rounds: int = 2,
    trace_context: TraceContext | None = None,
    parent_span_id: str | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> TeamExecution:
    bounded_members = max(1, int(max_members))
    bounded_rounds = max(1, int(max_rounds))
    if llm is None:
        return TeamExecution(
            enabled=False,
            fallback_reason="llm_unavailable",
        )

    roles = _normalize_roles(
        generate_dynamic_roles(prompt, llm, max_members=bounded_members),
        max_members=bounded_members,
    )
    if not roles:
        return TeamExecution(
            enabled=False,
            fallback_reason="roles_empty",
        )

    actual_rounds = _decide_team_rounds(
        prompt=prompt,
        llm=llm,
        max_rounds=bounded_rounds,
    )
    notes: list[str] = []
    trace_events: list[TraceEvent] = []
    effective_context = trace_context or create_trace_context(channel="internal.team_runtime")
    current_parent_span = str(parent_span_id or "")

    for round_idx in range(1, actual_rounds + 1):
        for role in roles:
            dispatch_event = build_trace_event(
                context=effective_context,
                sender="leader",
                receiver=role.name,
                performative="dispatch",
                content=f"[round={round_idx}] {role.goal}",
                parent_span_id=current_parent_span,
                metadata={"round": round_idx, "role": role.name},
            )
            trace_events.append(dispatch_event)
            current_parent_span = str(dispatch_event.get("span_id") or current_parent_span)
            if on_event is not None:
                on_event(dispatch_event)
            output = _invoke_role_agent(
                llm=llm,
                role=role,
                prompt=prompt,
                plan_text=plan_text,
                notes="\n".join(notes[-6:]),
                search_document_fn=search_document_fn,
                search_document_evidence_fn=search_document_evidence_fn,
            )
            result_event = build_trace_event(
                context=effective_context,
                sender=role.name,
                receiver="leader",
                performative="review",
                content=f"[round={round_idx}] {output}",
                parent_span_id=current_parent_span,
                metadata={"round": round_idx, "role": role.name},
            )
            trace_events.append(result_event)
            current_parent_span = str(result_event.get("span_id") or current_parent_span)
            if on_event is not None:
                on_event(result_event)
            notes.append(f"{role.name}: {output}")

    summary = "\n".join(notes[-12:])
    return TeamExecution(
        enabled=True,
        roles=[item.name for item in roles],
        member_count=len(roles),
        rounds=actual_rounds,
        summary=summary,
        trace_events=trace_events,
    )
