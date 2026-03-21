from __future__ import annotations

from typing import Any

from .contracts import AgentEvalCase, FinalAnswerJudge, FinalAnswerJudgeResult, ProcessContract
from .feedback import build_case_feedback


def _stable_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return value if isinstance(value, dict) else None
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, dict) else None
    return None


def _stable_list_of_dicts(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in payload:
        stable_item = _stable_dict(item)
        if stable_item is not None:
            normalized.append(stable_item)
    return normalized


def compute_execution_completion_ratio(
    *,
    todos: list[dict[str, Any]],
    runtime_state: dict[str, Any] | None,
) -> float | None:
    def _todo_completion_ratio() -> float | None:
        total_todos = len(todos)
        if total_todos == 0:
            return None
        completed = 0
        for item in todos:
            status = str(item.get("status") or "").strip().lower()
            if status in {"completed", "done"}:
                completed += 1
        return completed / total_todos

    def _runtime_plan_completion_ratio() -> float | None:
        if not isinstance(runtime_state, dict):
            return None
        current_plan = runtime_state.get("current_plan")
        if not isinstance(current_plan, dict):
            return None
        steps = current_plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return None
        completed_ids = runtime_state.get("completed_step_ids")
        if not isinstance(completed_ids, list):
            return None
        normalized_completed = {
            str(item).strip() for item in completed_ids if str(item).strip()
        }
        total_steps = 0
        completed_steps = 0
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id") or "").strip()
            if not step_id:
                continue
            total_steps += 1
            if step_id in normalized_completed:
                completed_steps += 1
        if total_steps == 0:
            return None
        return completed_steps / total_steps

    todo_ratio = _todo_completion_ratio()
    runtime_ratio = _runtime_plan_completion_ratio()

    if runtime_ratio is not None and todo_ratio is not None:
        return max(todo_ratio, runtime_ratio)
    if runtime_ratio is not None:
        return runtime_ratio
    if todo_ratio is None:
        return None
    if todo_ratio > 0.0:
        return todo_ratio
    # Todos can remain stale in end-to-end runs because write_todos does not
    # guarantee completion updates. Treat all-noncompleted todo state as
    # unavailable unless a runtime plan explicitly confirms zero progress.
    return None


def normalize_turn_result(turn_result: dict[str, Any]) -> dict[str, Any]:
    answer = str(turn_result.get("answer") or "").strip()
    evidence_items = turn_result.get("evidence_items")
    normalized_evidence = _stable_list_of_dicts(evidence_items)
    todos = turn_result.get("todos")
    normalized_todos = _stable_list_of_dicts(todos)
    plan = _stable_dict(turn_result.get("plan"))
    runtime_state = _stable_dict(turn_result.get("runtime_state"))
    agent_plan = _stable_dict(turn_result.get("agent_plan"))
    phase_path = str(turn_result.get("phase_path") or "").strip()
    trace_payload = turn_result.get("trace_payload")
    normalized_trace = trace_payload if isinstance(trace_payload, list) else []
    raw_output_messages = turn_result.get("output_messages")
    output_messages: list[Any] = raw_output_messages if isinstance(raw_output_messages, list) else []
    used_tool_names: list[str] = []
    for item in output_messages:
        tool_calls = (
            item.get("tool_calls")
            if isinstance(item, dict)
            else getattr(item, "tool_calls", None)
        )
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_name = str(tool_call.get("name") or "").strip()
            if tool_name and tool_name not in used_tool_names:
                used_tool_names.append(tool_name)
    execution_completion_ratio = compute_execution_completion_ratio(
        todos=normalized_todos,
        runtime_state=runtime_state,
    )
    return {
        "answer": answer,
        "evidence_items": normalized_evidence,
        "evidence_count": len(normalized_evidence),
        "todos": normalized_todos,
        "todo_count": len(normalized_todos),
        "plan": plan,
        "runtime_state": runtime_state,
        "agent_plan": agent_plan,
        "plan_present": bool(plan or agent_plan or (runtime_state or {}).get("current_plan")),
        "phase_path": phase_path,
        "trace_payload": normalized_trace,
        "trace_diagnostic_count": len(normalized_trace),
        "output_messages": output_messages,
        "used_tool_names": used_tool_names,
        "execution_completion_ratio": execution_completion_ratio,
        "run_latency_ms": float(turn_result.get("run_latency_ms") or 0.0),
        "used_document_rag": bool(turn_result.get("used_document_rag", False)),
        "leader_tool_names": (
            turn_result.get("leader_tool_names")
            if isinstance(turn_result.get("leader_tool_names"), list)
            else []
        ),
    }


def _evaluate_judge_answer(
    case: AgentEvalCase,
    normalized_result: dict[str, Any],
    judge: FinalAnswerJudge | None,
) -> dict[str, Any]:
    if judge is None:
        raise ValueError("A final-answer LLM judge is required for task-completion evals.")
    judge_result = judge(case, normalized_result)
    if not isinstance(judge_result, FinalAnswerJudgeResult):
        raise TypeError("Final answer judge must return FinalAnswerJudgeResult.")
    return {
        "mode": "rubric",
        "passed": bool(judge_result.passed),
        "score": judge_result.score,
        "reasoning": str(judge_result.reasoning or "").strip(),
    }


def _phase_labels_pass(contract: ProcessContract, phase_path: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for label in contract.required_phase_labels:
        if label not in phase_path:
            missing.append(label)
    return not missing, missing


def evaluate_case_result(
    case: AgentEvalCase,
    turn_result: dict[str, Any],
    *,
    judge: FinalAnswerJudge | None,
) -> dict[str, Any]:
    normalized_result = normalize_turn_result(turn_result)
    answer = str(normalized_result["answer"])

    final_checks = [_evaluate_judge_answer(case, normalized_result, judge)]
    final_success = all(bool(item.get("passed", False)) for item in final_checks)

    process_contract = case.process_contract
    evidence_count = int(normalized_result["evidence_count"])
    required_evidence_count = max(
        process_contract.min_evidence_count,
        1 if process_contract.requires_evidence else 0,
    )
    evidence_pass = evidence_count >= required_evidence_count
    plan_present = bool(normalized_result["plan_present"])
    plan_pass = (not process_contract.require_plan) or plan_present
    todo_count = int(normalized_result["todo_count"])
    todo_pass = (not process_contract.require_todos) or todo_count > 0

    execution_completion_ratio = normalized_result["execution_completion_ratio"]
    ratio_pass = True
    if process_contract.min_execution_completion_ratio is not None:
        ratio_pass = (
            execution_completion_ratio is None
            or float(execution_completion_ratio) >= process_contract.min_execution_completion_ratio
        )

    phase_pass, missing_phase_labels = _phase_labels_pass(
        process_contract,
        str(normalized_result["phase_path"]),
    )
    used_tool_names = normalized_result["used_tool_names"]
    tool_names_pass = all(
        tool_name in used_tool_names for tool_name in process_contract.required_tool_names
    )

    process_success = all([evidence_pass, plan_pass, todo_pass, ratio_pass, phase_pass, tool_names_pass])
    completed = final_success and process_success
    process_checks = {
        "plan_passed": plan_pass,
        "todo_passed": todo_pass,
        "tool_names_passed": tool_names_pass,
        "used_tool_names": used_tool_names,
        "phase_labels_passed": phase_pass,
        "missing_phase_labels": missing_phase_labels,
        "ratio_passed": ratio_pass,
    }
    evidence_coverage = {
        "passed": evidence_pass,
        "count": evidence_count,
        "required_count": required_evidence_count,
    }
    feedback = build_case_feedback(
        final_success=final_success,
        process_success=process_success,
        final_checks=final_checks,
        process_checks=process_checks,
        evidence_coverage=evidence_coverage,
    )

    return {
        "case_id": case.case_id,
        "category": case.category,
        "prompt": case.prompt,
        "completed": completed,
        "final_success": final_success,
        "process_success": process_success,
        "execution_completion_ratio": execution_completion_ratio,
        "evidence_coverage": evidence_coverage,
        "final_checks": final_checks,
        "process_checks": process_checks,
        "feedback": feedback,
        "answer": answer,
        "diagnostics": {
            "phase_path": normalized_result["phase_path"],
            "trace_diagnostic_count": normalized_result["trace_diagnostic_count"],
            "run_latency_ms": normalized_result["run_latency_ms"],
            "used_document_rag": normalized_result["used_document_rag"],
            "leader_tool_names": normalized_result["leader_tool_names"],
        },
    }
