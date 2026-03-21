from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_eval_report(
    *,
    fixture_path: str,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_cases = len(case_results)
    completed_cases = sum(1 for item in case_results if bool(item.get("completed", False)))
    final_success_cases = sum(1 for item in case_results if bool(item.get("final_success", False)))
    process_success_cases = sum(1 for item in case_results if bool(item.get("process_success", False)))
    evidence_success_cases = sum(
        1 for item in case_results if bool((item.get("evidence_coverage") or {}).get("passed", False))
    )

    ratios: list[float] = []
    for item in case_results:
        ratio = item.get("execution_completion_ratio")
        if isinstance(ratio, (int, float)):
            ratios.append(float(ratio))
    average_execution_completion_ratio = (
        sum(ratios) / len(ratios) if ratios else 0.0
    )

    failed_case_ids = [
        str(item.get("case_id") or "").strip()
        for item in case_results
        if not bool(item.get("completed", False))
    ]
    failed_case_ids = [item for item in failed_case_ids if item]
    remediation_area_counts: dict[str, int] = {}
    for item in case_results:
        feedback = item.get("feedback")
        if not isinstance(feedback, dict):
            continue
        remediation_areas = feedback.get("remediation_area")
        if not isinstance(remediation_areas, list):
            continue
        for area in remediation_areas:
            normalized_area = str(area or "").strip()
            if not normalized_area:
                continue
            remediation_area_counts[normalized_area] = remediation_area_counts.get(normalized_area, 0) + 1

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fixture_path": fixture_path,
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "completion_rate": (completed_cases / total_cases) if total_cases else 0.0,
        "final_success_rate": (final_success_cases / total_cases) if total_cases else 0.0,
        "process_success_rate": (process_success_cases / total_cases) if total_cases else 0.0,
        "evidence_coverage_rate": (evidence_success_cases / total_cases) if total_cases else 0.0,
        "average_execution_completion_ratio": average_execution_completion_ratio,
        "failed_case_ids": failed_case_ids,
        "remediation_area_counts": remediation_area_counts,
        "cases": case_results,
    }
