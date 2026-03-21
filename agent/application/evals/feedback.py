from __future__ import annotations

from typing import Any


def build_case_feedback(
    *,
    final_success: bool,
    process_success: bool,
    final_checks: list[dict[str, Any]],
    process_checks: dict[str, Any],
    evidence_coverage: dict[str, Any],
) -> dict[str, Any]:
    remediation_area: list[str] = []
    failure_reason_parts: list[str] = []
    recommended_actions: list[str] = []

    if not final_success:
        remediation_area.append("prompt")
        reasoning = "; ".join(
            str(item.get("reasoning") or "").strip()
            for item in final_checks
            if str(item.get("reasoning") or "").strip()
        )
        failure_reason_parts.append(reasoning or "final answer did not satisfy the task contract")
        recommended_actions.append(
            "Tighten the task prompt or evaluation rubric so the answer must cover the required outcome."
        )

    if not bool(evidence_coverage.get("passed", False)):
        if "retrieval/tooling" not in remediation_area:
            remediation_area.append("retrieval/tooling")
        failure_reason_parts.append("insufficient evidence coverage")
        recommended_actions.append(
            "Strengthen retrieval requirements or tool guidance so the turn gathers enough grounded evidence."
        )

    if not bool(process_checks.get("tool_names_passed", True)):
        if "retrieval/tooling" not in remediation_area:
            remediation_area.append("retrieval/tooling")
        failure_reason_parts.append("required stable tools were not used")
        recommended_actions.append(
            "Adjust prompt or routing rules so the turn uses the required stable tools for this task type."
        )

    if not bool(process_checks.get("plan_passed", True)) or not bool(process_checks.get("todo_passed", True)):
        if "prompt" not in remediation_area:
            remediation_area.append("prompt")
        failure_reason_parts.append("required planning or task tracking contract was not met")
        recommended_actions.append(
            "Strengthen planning instructions or routing guidance so multi-step tasks expose the required stable outputs."
        )

    if not bool(process_checks.get("ratio_passed", True)):
        if "architecture" not in remediation_area:
            remediation_area.append("architecture")
        failure_reason_parts.append("execution completion ratio was below the required threshold")
        recommended_actions.append(
            "Improve multi-step execution reliability so planned work completes before the final answer is produced."
        )

    if not bool(process_checks.get("phase_labels_passed", True)):
        if "architecture" not in remediation_area:
            remediation_area.append("architecture")
        failure_reason_parts.append("required stable phase labels were missing")
        recommended_actions.append(
            "Check whether the stable eval contract should expose this process signal or whether the task flow needs adjustment."
        )

    deduped_actions: list[str] = []
    for action in recommended_actions:
        if action not in deduped_actions:
            deduped_actions.append(action)

    return {
        "failure_reason": "; ".join(item for item in failure_reason_parts if item),
        "feedback_summary": (
            "Stable end-to-end contract satisfied."
            if final_success and process_success
            else "Review the recommended remediation areas and actions for this failed contract."
        ),
        "remediation_area": remediation_area,
        "recommended_actions": deduped_actions,
        "confidence": "medium" if remediation_area else "high",
    }
