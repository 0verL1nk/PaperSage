import json
import re
from datetime import UTC, datetime
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from ..capabilities import build_agent_tools, build_progressive_tool_middleware
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
TEAM_TODO_STATUSES = ("todo", "in_progress", "done", "blocked", "canceled")


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


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _append_todo_history(
    record: dict[str, Any],
    *,
    action: str,
    status: str,
    note: str = "",
) -> None:
    history = record.get("history")
    if not isinstance(history, list):
        history = []
        record["history"] = history
    history.append(
        {
            "ts": _now_iso(),
            "action": action,
            "status": status,
            "note": note.strip(),
        }
    )


def _build_team_todo_records(
    *,
    roles: list[TeamRole],
    rounds: int,
    plan_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_id_by_key: dict[tuple[int, str], str] = {}
    for round_idx in range(1, rounds + 1):
        same_round_task_ids: list[str] = []
        for role_idx, role in enumerate(roles, start=1):
            task_id = f"team_r{round_idx}_{role.name}"
            deps: list[str] = []
            previous_task_id = task_id_by_key.get((round_idx - 1, role.name))
            if previous_task_id:
                deps.append(previous_task_id)
            if role_idx == len(roles) and len(roles) > 1:
                deps.extend(same_round_task_ids)
            # 保持依赖顺序同时去重
            unique_deps: list[str] = []
            seen: set[str] = set()
            for dep in deps:
                if dep and dep not in seen:
                    unique_deps.append(dep)
                    seen.add(dep)
            now_text = _now_iso()
            record: dict[str, Any] = {
                "id": task_id,
                "title": f"[Round {round_idx}] {role.name}",
                "details": role.goal,
                "status": "todo",
                "priority": "high" if role_idx == len(roles) else "medium",
                "assignee": role.name,
                "dependencies": unique_deps,
                "plan_id": plan_id,
                "step_ref": f"r{round_idx}:{role.name}",
                "round": round_idx,
                "created_at": now_text,
                "updated_at": now_text,
                "history": [],
            }
            _append_todo_history(
                record,
                action="upsert",
                status="todo",
                note="team runtime task initialized",
            )
            records.append(record)
            task_id_by_key[(round_idx, role.name)] = task_id
            same_round_task_ids.append(task_id)
    return records


def _todo_dependencies_done(
    record: dict[str, Any],
    *,
    records_by_id: dict[str, dict[str, Any]],
) -> bool:
    dependencies = record.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        return True
    for dep in dependencies:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        dep_record = records_by_id.get(dep_id)
        if not isinstance(dep_record, dict):
            return False
        if str(dep_record.get("status") or "").strip().lower() != "done":
            return False
    return True


def _build_todo_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    stats = {name: 0 for name in TEAM_TODO_STATUSES}
    for record in records:
        status = str(record.get("status") or "todo").strip().lower()
        if status not in stats:
            continue
        stats[status] += 1
    return stats


def _has_dependency_cycle(records: list[dict[str, Any]]) -> bool:
    records_by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        todo_id = str(item.get("id") or "").strip()
        if not todo_id:
            continue
        records_by_id[todo_id] = item

    visiting: set[str] = set()
    visited: set[str] = set()

    def _walk(todo_id: str) -> bool:
        if todo_id in visiting:
            return True
        if todo_id in visited:
            return False
        visiting.add(todo_id)
        record = records_by_id.get(todo_id, {})
        deps = record.get("dependencies")
        dep_ids = deps if isinstance(deps, list) else []
        for dep in dep_ids:
            dep_id = str(dep or "").strip()
            if not dep_id or dep_id not in records_by_id:
                continue
            if _walk(dep_id):
                return True
        visiting.remove(todo_id)
        visited.add(todo_id)
        return False

    for todo_id in records_by_id:
        if _walk(todo_id):
            return True
    return False


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
    middleware = build_progressive_tool_middleware(tools)
    create_kwargs: dict[str, Any] = {
        "model": llm,
        "tools": tools,
        "system_prompt": system_prompt,
        "checkpointer": InMemorySaver(),
    }
    if middleware:
        create_kwargs["middleware"] = middleware
    member_agent = create_agent(**create_kwargs)
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
    policy_checkpoint_fn: Callable[[list[dict[str, Any]]], tuple[bool, str | None]] | None = None,
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
    fallback_reason: str | None = None
    effective_context = trace_context or create_trace_context(channel="internal.team_runtime")
    current_parent_span = str(parent_span_id or "")
    todo_records = _build_team_todo_records(
        roles=roles,
        rounds=actual_rounds,
        plan_id=f"team:{effective_context.run_id}",
    )
    if _has_dependency_cycle(todo_records):
        for item in todo_records:
            if not isinstance(item, dict):
                continue
            item["status"] = "blocked"
            item["updated_at"] = _now_iso()
            _append_todo_history(
                item,
                action="blocked",
                status="blocked",
                note="dependency cycle detected",
            )
        return TeamExecution(
            enabled=True,
            roles=[item.name for item in roles],
            member_count=len(roles),
            rounds=actual_rounds,
            summary="",
            fallback_reason="todo_dependency_cycle",
            trace_events=trace_events,
            todo_records=todo_records,
            todo_stats=_build_todo_stats(todo_records),
        )
    records_by_id: dict[str, dict[str, Any]] = {
        str(item.get("id") or ""): item for item in todo_records if isinstance(item, dict)
    }
    role_map = {item.name: item for item in roles}
    role_index = {item.name: idx for idx, item in enumerate(roles)}

    while True:
        if policy_checkpoint_fn is not None:
            try:
                should_continue, checkpoint_reason = policy_checkpoint_fn(todo_records)
            except Exception:
                should_continue, checkpoint_reason = True, None
            if not should_continue:
                fallback_reason = (
                    str(checkpoint_reason).strip() if checkpoint_reason else "policy_interrupted"
                )
                break

        ready_records = [
            item
            for item in todo_records
            if str(item.get("status") or "").strip().lower() == "todo"
            and _todo_dependencies_done(item, records_by_id=records_by_id)
        ]
        if not ready_records:
            break
        ready_records.sort(
            key=lambda item: (
                int(item.get("round") or 0),
                role_index.get(str(item.get("assignee") or ""), 999),
                str(item.get("id") or ""),
            )
        )
        current_task = ready_records[0]
        todo_id = str(current_task.get("id") or "").strip()
        round_idx = int(current_task.get("round") or 0)
        role_name = str(current_task.get("assignee") or "").strip()
        role = role_map.get(role_name)
        if role is None:
            current_task["status"] = "blocked"
            current_task["updated_at"] = _now_iso()
            _append_todo_history(
                current_task,
                action="blocked",
                status="blocked",
                note="role not found during dispatch",
            )
            continue

        current_task["status"] = "in_progress"
        current_task["updated_at"] = _now_iso()
        _append_todo_history(
            current_task,
            action="update_status",
            status="in_progress",
            note="leader dispatched task",
        )
        dispatch_event = build_trace_event(
            context=effective_context,
            sender="leader",
            receiver=role.name,
            performative="dispatch",
            content=f"[round={round_idx}] {role.goal}",
            parent_span_id=current_parent_span,
            metadata={
                "round": round_idx,
                "role": role.name,
                "todo_id": todo_id,
                "dependencies": list(current_task.get("dependencies") or []),
                "step_ref": str(current_task.get("step_ref") or ""),
            },
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
        current_task["status"] = "done"
        current_task["output"] = output
        current_task["updated_at"] = _now_iso()
        _append_todo_history(
            current_task,
            action="update_status",
            status="done",
            note="role completed task",
        )
        result_event = build_trace_event(
            context=effective_context,
            sender=role.name,
            receiver="leader",
            performative="review",
            content=f"[round={round_idx}] {output}",
            parent_span_id=current_parent_span,
            metadata={
                "round": round_idx,
                "role": role.name,
                "todo_id": todo_id,
                "step_ref": str(current_task.get("step_ref") or ""),
            },
        )
        trace_events.append(result_event)
        current_parent_span = str(result_event.get("span_id") or current_parent_span)
        if on_event is not None:
            on_event(result_event)
        notes.append(f"{role.name}: {output}")

    unresolved_records = [
        item
        for item in todo_records
        if str(item.get("status") or "").strip().lower() in {"todo", "in_progress"}
    ]
    if unresolved_records:
        if not fallback_reason:
            fallback_reason = "todo_dependency_unresolved"
        unresolved_status = (
            "blocked" if fallback_reason == "todo_dependency_unresolved" else "canceled"
        )
        for item in unresolved_records:
            item["status"] = unresolved_status
            item["updated_at"] = _now_iso()
            _append_todo_history(
                item,
                action="blocked" if unresolved_status == "blocked" else "update_status",
                status=unresolved_status,
                note=(
                    "dependencies unresolved or cyclic"
                    if unresolved_status == "blocked"
                    else f"stopped by policy checkpoint: {fallback_reason}"
                ),
            )

    summary = "\n".join(notes[-12:])
    return TeamExecution(
        enabled=True,
        roles=[item.name for item in roles],
        member_count=len(roles),
        rounds=actual_rounds,
        summary=summary,
        fallback_reason=fallback_reason,
        trace_events=trace_events,
        todo_records=todo_records,
        todo_stats=_build_todo_stats(todo_records),
    )
