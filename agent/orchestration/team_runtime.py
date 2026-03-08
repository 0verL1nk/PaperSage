import json
import re
from datetime import datetime, timezone
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

# ---------------------------------------------------------------------------
# Leader Todo Planning — leader 先想清楚再分工
# ---------------------------------------------------------------------------

LEADER_TODO_PLAN_INSTRUCTION = """
你是团队领导者（leader）。请根据用户问题、执行计划和团队角色，制定一份可执行的任务清单。

要求：
1) 每个任务分配给已有的某个角色（assignee 必须来自 available_roles 列表）
2) 每个任务有唯一的 id（格式：t1, t2, t3…）
3) deps 字段列出该任务依赖的其他任务 id（空列表表示无依赖，可并行执行）
4) details 必须具体、可执行，不能只是重复角色目标
5) round 表示执行轮次（从1开始），需要迭代校验的任务可以在第2轮
6) priority: "high" 表示关键路径，"medium" 表示普通
7) 整体依赖关系必须是有向无环图（DAG）

注意：任务数量应与角色数 × 轮数匹配，不要遗漏角色，不要产生孤立任务。
"""


class LeaderTodoItemSchema(BaseModel):
    id: str = Field(min_length=1, description="任务唯一 id，如 t1/t2/t3")
    title: str = Field(min_length=1, description="任务标题，简短描述")
    details: str = Field(min_length=1, description="具体可执行的任务描述")
    assignee: str = Field(min_length=1, description="分配角色名（必须在 available_roles 中）")
    deps: list[str] = Field(default_factory=list, description="依赖的任务 id 列表")
    round: int = Field(default=1, ge=1, description="执行轮次")
    priority: str = Field(default="medium", description="优先级: high / medium")


class LeaderTodoPlanSchema(BaseModel):
    todos: list[LeaderTodoItemSchema] = Field(min_length=1)


LEADER_TODO_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", LEADER_TODO_PLAN_INSTRUCTION),
        (
            "human",
            "用户问题：\n{prompt}\n\n"
            "执行计划：\n{plan_text}\n\n"
            "可用角色（每行: name | goal）：\n{roles_text}\n\n"
            "允许最大轮次：{max_rounds}\n\n"
            "{cycle_hint}",
        ),
    ]
)


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
            roles_fallback: list[TeamRole] = []
            if isinstance(roles_raw, list):
                for item in roles_raw[:bounded_members]:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip().lower()
                    goal = str(item.get("goal") or "").strip()
                    if not name:
                        continue
                    roles_fallback.append(TeamRole(name=name, goal=goal or "完成分配任务"))
            return roles_fallback if roles_fallback else _fallback_roles()[:bounded_members]
        except Exception:
            return _fallback_roles()[:bounded_members]


def _with_structured_output(llm: Any, schema: type[BaseModel]) -> Any:
    # Prefer function-calling path to avoid provider parse payload warnings.
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:
        return llm.with_structured_output(schema)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _leader_plan_todo(
    *,
    prompt: str,
    plan_text: str,
    roles: list[TeamRole],
    actual_rounds: int,
    llm: Any,
    plan_id: str,
    cycle_hint: str = "",
) -> list[dict[str, Any]]:
    """让 leader LLM 根据任务和角色主动规划 todo 列表（带语义依赖）。

    cycle_hint: 若上次规划存在依赖环，将环内节点信息传入，引导 leader 修正。
    失败时 fallback 到机械生成的 _build_team_todo_records()。
    """
    roles_text = "\n".join(f"{r.name} | {r.goal}" for r in roles)
    # 将 cycle_hint 格式化为 prompt 片段（无提示时为空字符串，不影响 prompt）
    cycle_hint_text = (
        f"⚠️ 上次规划检测到依赖环，请务必修正！\n"
        f"涉及环的任务 id：{cycle_hint}\n"
        f"请检查这些任务的 deps 字段，消除循环依赖后重新输出完整任务列表。\n"
        if cycle_hint
        else ""
    )
    try:
        if not hasattr(llm, "with_structured_output"):
            raise ValueError("llm lacks with_structured_output")
        chain = LEADER_TODO_PLAN_PROMPT | _with_structured_output(llm, LeaderTodoPlanSchema)
        payload = chain.invoke(
            {
                "prompt": prompt,
                "plan_text": plan_text or "（无执行计划）",
                "roles_text": roles_text,
                "max_rounds": actual_rounds,
                "cycle_hint": cycle_hint_text,
            }
        )
        if isinstance(payload, dict):
            payload = LeaderTodoPlanSchema.model_validate(payload)
        if not isinstance(payload, LeaderTodoPlanSchema) or not payload.todos:
            raise ValueError("empty todo plan from leader")
    except Exception:
        # fallback: 尝试非结构化 LLM + JSON 解析
        try:
            result = llm.invoke(
                f"{LEADER_TODO_PLAN_INSTRUCTION}\n\n"
                f"用户问题：{prompt}\n\n"
                f"执行计划：{plan_text or '无'}\n\n"
                f"可用角色：\n{roles_text}\n\n"
                f"允许最大轮次：{actual_rounds}\n\n"
                + (f"{cycle_hint_text}\n\n" if cycle_hint_text else "")
                + "请以 JSON 格式输出 {\"todos\":[...]} 结构。"
            )
            text = _llm_content_to_text(getattr(result, "content", result))
            # 找最外层 { }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                raw = json.loads(text[start : end + 1])
                payload = LeaderTodoPlanSchema.model_validate(raw)
            else:
                raise ValueError("no json block")
        except Exception:
            # 最终 fallback：机械生成
            return _build_team_todo_records_mechanical(
                roles=roles, rounds=actual_rounds, plan_id=plan_id
            )

    # ── 第一遍：收集全部合法 id，用于后续 deps 引用校验 ──────────────────────
    valid_role_names = {r.name for r in roles}
    now_text = _now_iso()

    # 先做 id 合法性过滤（assignee 必须在角色列表中，id 不能重复）
    candidate_ids: set[str] = set()
    for item in payload.todos:
        raw_id = str(item.id or "").strip()
        assignee = str(item.assignee or "").strip().lower()
        if raw_id and assignee in valid_role_names and raw_id not in candidate_ids:
            candidate_ids.add(raw_id)

    # ── 第二遍：构建 records，deps 保留所有在 candidate_ids 内的引用 ──────────
    # DAG 校验由下游 _has_dependency_cycle() 做程序判定，这里不做静默裁剪
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for item in payload.todos:
        raw_id = str(item.id or "").strip()
        assignee = str(item.assignee or "").strip().lower()
        if not raw_id or assignee not in valid_role_names:
            continue
        if raw_id in seen_ids:
            continue
        seen_ids.add(raw_id)
        # 只过滤引用了不存在 id 的 dep（不做前向引用裁剪，DAG 由程序算法校验）
        deps = [
            str(d).strip()
            for d in (item.deps or [])
            if str(d).strip() and str(d).strip() in candidate_ids and str(d).strip() != raw_id
        ]
        round_val = max(1, min(int(item.round or 1), actual_rounds))
        priority = str(item.priority or "medium").strip().lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        record: dict[str, Any] = {
            "id": raw_id,
            "title": str(item.title or raw_id).strip(),
            "details": str(item.details or "").strip() or "完成分配任务",
            "status": "todo",
            "priority": priority,
            "assignee": assignee,
            "dependencies": deps,
            "plan_id": plan_id,
            "step_ref": f"r{round_val}:{assignee}",
            "round": round_val,
            "created_at": now_text,
            "updated_at": now_text,
            "history": [],
            "leader_planned": True,
        }
        _append_todo_history(record, action="upsert", status="todo", note="leader planned task")
        records.append(record)

    # 若规划结果为空（全部被过滤），fallback 机械生成
    if not records:
        return _build_team_todo_records_mechanical(
            roles=roles, rounds=actual_rounds, plan_id=plan_id
        )
    return records


def _build_team_todo_records_mechanical(
    *,
    roles: list[TeamRole],
    rounds: int,
    plan_id: str,
) -> list[dict[str, Any]]:
    """机械 fallback：roles × rounds 笛卡尔积，仅在 leader 规划失败时使用。"""
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
                "leader_planned": False,
            }
            _append_todo_history(
                record,
                action="upsert",
                status="todo",
                note="team runtime task initialized (mechanical fallback)",
            )
            records.append(record)
            task_id_by_key[(round_idx, role.name)] = task_id
            same_round_task_ids.append(task_id)
    return records


def _build_team_todo_records(
    *,
    roles: list[TeamRole],
    rounds: int,
    plan_id: str,
) -> list[dict[str, Any]]:
    """向后兼容入口，委托给 _build_team_todo_records_mechanical()。

    run_team_tasks() 现在优先调用 _leader_plan_todo()，
    此函数仅供外部单元测试或旧调用路径使用。
    """
    return _build_team_todo_records_mechanical(roles=roles, rounds=rounds, plan_id=plan_id)


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


def _find_cycle_ids(records: list[dict[str, Any]]) -> list[str]:
    """返回所有处于依赖环中的 todo id 列表（DFS 着色法）。

    不在任何环中的节点不会出现在返回列表里，方便给 leader 精准提示。
    """
    records_by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        todo_id = str(item.get("id") or "").strip()
        if todo_id:
            records_by_id[todo_id] = item

    # 0=white, 1=gray(visiting), 2=black(done)
    color: dict[str, int] = {k: 0 for k in records_by_id}
    on_stack: list[str] = []
    in_cycle: set[str] = set()

    def _dfs(node: str) -> bool:
        """Returns True if a cycle is detected going through `node`."""
        color[node] = 1
        on_stack.append(node)
        deps = records_by_id.get(node, {}).get("dependencies") or []
        for dep in deps:
            dep_id = str(dep or "").strip()
            if dep_id not in records_by_id:
                continue
            if color[dep_id] == 1:
                # Found a back-edge → mark all nodes on stack from dep_id onward
                idx = on_stack.index(dep_id)
                for n in on_stack[idx:]:
                    in_cycle.add(n)
                return True
            if color[dep_id] == 0:
                _dfs(dep_id)
        on_stack.pop()
        color[node] = 2
        return False

    for node in list(records_by_id):
        if color[node] == 0:
            _dfs(node)

    return sorted(in_cycle)


def _invoke_role_agent(
    *,
    llm: Any,
    role: TeamRole,
    prompt: str,
    plan_text: str,
    notes: str,
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None,
    round_idx: int = 1,
    prior_output: str = "",
    task_details: str = "",
) -> str:
    round_instruction = (
        f"当前为第 {round_idx} 轮协作。\n"
        + (
            f"你在上一轮的输出如下，请在此基础上深化、修正或补充：\n{prior_output}\n\n"
            if round_idx > 1 and prior_output
            else "这是首轮执行，请完成初始分析。\n\n"
        )
    )
    # leader 规划的具体任务描述优先于通用角色目标
    effective_task = task_details.strip() if task_details.strip() else role.goal
    system_prompt = (
        f"你是团队中的 {role.name}。\n"
        f"角色目标：{role.goal}\n\n"
        f"{round_instruction}"
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
        f"本次具体任务（leader 分配）：\n{effective_task}\n"
        f"当前计划：\n{plan_text or '无'}\n"
        f"当前团队笔记（其他成员已完成的内容）：\n{notes or '无'}\n"
        "请输出你作为该角色在本轮的执行结果。"
    )
    result = member_agent.invoke(
        {"messages": [{"role": "user", "content": task_prompt}]},
        config={"configurable": {"thread_id": f"team:{role.name}:r{round_idx}"}},
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

    # ── Phase 1: leader 主动规划 todo（带语义依赖）──────────────────────────
    plan_id = f"team:{effective_context.run_id}"
    max_todo_plan_retries = 3
    todo_plan_retry = 0
    cycle_hint: str = ""           # 精准的环内节点信息，逐次传给 leader
    while True:
        todo_records = _leader_plan_todo(
            prompt=prompt,
            plan_text=plan_text,
            roles=roles,
            actual_rounds=actual_rounds,
            llm=llm,
            plan_id=plan_id,
            cycle_hint=cycle_hint,
        )
        # 发布 leader 规划的 todo trace 事件
        todo_summary_lines: list[str] = []
        for rec in todo_records:
            deps_str = ",".join(rec.get("dependencies") or []) or "无"
            todo_summary_lines.append(
                f"  [{rec['id']}] {rec['title']} | assignee={rec['assignee']}"
                f" | round={rec['round']} | deps={deps_str}"
                f" | {rec['details'][:60]}{'...' if len(rec.get('details',''))>60 else ''}"
            )
        todo_plan_event = build_trace_event(
            context=effective_context,
            sender="leader",
            receiver="leader",
            performative="plan_todo",
            content=(
                f"leader 已规划 {len(todo_records)} 个任务（leader_planned="
                f"{todo_records[0].get('leader_planned', False) if todo_records else False}"
                + (f"，第 {todo_plan_retry} 次修正后重规划" if todo_plan_retry else "")
                + "）：\n"
                + "\n".join(todo_summary_lines)
            ),
            parent_span_id=current_parent_span,
            metadata={
                "todo_count": len(todo_records),
                "leader_planned": bool(todo_records and todo_records[0].get("leader_planned")),
                "todo_plan_retry": todo_plan_retry,
            },
        )
        trace_events.append(todo_plan_event)
        current_parent_span = str(todo_plan_event.get("span_id") or current_parent_span)
        if on_event is not None:
            on_event(todo_plan_event)
        # ── Phase 1 end ──────────────────────────────────────────────────────────

        # 检查依赖环（程序算法 DFS，非 LLM 判定）
        if not _has_dependency_cycle(todo_records):
            break

        # 用算法精准找出环内节点，构造 cycle_hint 反馈给 leader
        cycle_ids = _find_cycle_ids(todo_records)
        cycle_hint = ", ".join(cycle_ids) if cycle_ids else str([r.get("id") for r in todo_records])
        cycle_event = build_trace_event(
            context=effective_context,
            sender="system",
            receiver="leader",
            performative="plan_todo_reject",
            content=(
                f"⚠️ 程序检测到 todo 依赖存在环，涉及任务：{cycle_hint}。"
                f"请 leader 修改这些任务的 deps，消除循环依赖。"
                f"（第 {todo_plan_retry + 1} 次重试）"
            ),
            parent_span_id=current_parent_span,
            metadata={
                "todo_plan_retry": todo_plan_retry + 1,
                "cycle_ids": cycle_ids,
            },
        )
        trace_events.append(cycle_event)
        if on_event is not None:
            on_event(cycle_event)
        todo_plan_retry += 1
        if todo_plan_retry >= max_todo_plan_retries:
            # 超过最大重试次数，全部 blocked
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
    max_iterations = len(todo_records) + 1  # 最多执行 N+1 次迭代，防止依赖链计算异常导致死循环
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            fallback_reason = "max_iterations_exceeded"
            break
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

        # 查找该角色上一轮的输出，用于多轮迭代时提供上下文
        # 兼容 leader 规划的任意 id 格式（t1/t2）和 mechanical 格式（team_r1_xxx）
        prior_output = ""
        if round_idx > 1:
            for rec in todo_records:
                if (
                    str(rec.get("assignee") or "").strip() == role_name
                    and int(rec.get("round") or 0) == round_idx - 1
                    and str(rec.get("status") or "") == "done"
                ):
                    prior_output = str(rec.get("output") or "")
                    break

        # dispatch 消息内容：首轮说明目标，后续轮次带入上一轮产出摘要，体现真实迭代
        if round_idx == 1 or not prior_output:
            dispatch_content = f"[round={round_idx}] 目标：{role.goal}"
        else:
            prior_snippet = prior_output[:300].replace("\n", " ")
            dispatch_content = (
                f"[round={round_idx}] 目标：{role.goal}\n"
                f"  → 上轮产出摘要：{prior_snippet}{'...' if len(prior_output) > 300 else ''}\n"
                f"  → 请在上轮基础上深化或修正结论"
            )

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
            content=dispatch_content,
            parent_span_id=current_parent_span,
            metadata={
                "round": round_idx,
                "role": role.name,
                "todo_id": todo_id,
                "dependencies": list(current_task.get("dependencies") or []),
                "step_ref": str(current_task.get("step_ref") or ""),
                "has_prior_output": bool(prior_output),
            },
        )
        trace_events.append(dispatch_event)
        current_parent_span = str(dispatch_event.get("span_id") or current_parent_span)
        if on_event is not None:
            on_event(dispatch_event)

        # 取出 leader 规划的具体任务描述，传给子 agent
        task_details = str(current_task.get("details") or "").strip()

        try:
            output = _invoke_role_agent(
                llm=llm,
                role=role,
                prompt=prompt,
                plan_text=plan_text,
                notes="\n".join(notes[-6:]),
                search_document_fn=search_document_fn,
                search_document_evidence_fn=search_document_evidence_fn,
                round_idx=round_idx,
                prior_output=prior_output,
                task_details=task_details,
            )
        except Exception as exc:
            current_task["status"] = "blocked"
            current_task["updated_at"] = _now_iso()
            _append_todo_history(
                current_task,
                action="blocked",
                status="blocked",
                note=f"role agent invocation failed: {exc}",
            )
            continue
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
