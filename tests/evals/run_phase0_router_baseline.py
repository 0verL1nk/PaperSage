import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage

from agent.middlewares.orchestration import OrchestrationMiddleware

WORKFLOW_REACT = "react"
WORKFLOW_PLAN_ACT = "plan_act"
WORKFLOW_PLAN_ACT_REPLAN = "plan_act_replan"

VALID_WORKFLOWS = {WORKFLOW_REACT, WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN}


class _FixtureRouterLLM:
    def invoke(self, prompt: str) -> SimpleNamespace:
        prompt_text = str(prompt)
        is_complex = len(prompt_text) > 100 or "请完成" in prompt_text or "任务" in prompt_text
        needs_team = any(token in prompt_text for token in ("协作", "多角色", "对比", "比较"))
        payload = {
            "is_complex": is_complex or needs_team,
            "needs_team": needs_team,
            "reason": "middleware baseline heuristic",
        }
        return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))


def _predict_workflow_mode(prompt: str) -> tuple[str, str]:
    middleware = OrchestrationMiddleware(llm=_FixtureRouterLLM())
    middleware.before_model(
        {"messages": [HumanMessage(content=prompt)]},
        runtime=None,
        config={"configurable": {"state": {}}},
    )
    analysis = middleware._last_analysis or {}
    if analysis.get("needs_team"):
        return WORKFLOW_PLAN_ACT_REPLAN, str(analysis.get("reason") or "")
    if analysis.get("is_complex"):
        return WORKFLOW_PLAN_ACT, str(analysis.get("reason") or "")
    return WORKFLOW_REACT, str(analysis.get("reason") or "")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_no} is not a JSON object.")
        expected = payload.get("expected_workflow")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Line {line_no} missing valid 'prompt'.")
        if not isinstance(expected, str) or expected not in VALID_WORKFLOWS:
            raise ValueError(f"Line {line_no} has invalid 'expected_workflow': {expected}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"No rows found in fixture: {path}")
    return rows


def run_router_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    distribution: Counter[str] = Counter()
    matches = 0
    latency_ms_values: list[float] = []

    for record in records:
        prompt = record["prompt"]
        expected = record["expected_workflow"]
        started = time.perf_counter()
        predicted, reason = _predict_workflow_mode(prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latency_ms_values.append(latency_ms)
        distribution[predicted] += 1
        is_match = predicted == expected
        if is_match:
            matches += 1
        predictions.append(
            {
                "id": record.get("id"),
                "category": record.get("category"),
                "expected_workflow": expected,
                "predicted_workflow": predicted,
                "is_match": is_match,
                "latency_ms": round(latency_ms, 3),
                "reason": reason,
            }
        )

    total = len(records)
    mismatches = [item for item in predictions if not item["is_match"]]
    average_latency_ms = sum(latency_ms_values) / total if total else 0.0
    p95_latency_ms = sorted(latency_ms_values)[int(max(0, total * 0.95 - 1))] if total else 0.0

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "total_cases": total,
        "matched_cases": matches,
        "match_rate": (matches / total) if total else 0.0,
        "route_distribution": {
            WORKFLOW_REACT: distribution.get(WORKFLOW_REACT, 0),
            WORKFLOW_PLAN_ACT: distribution.get(WORKFLOW_PLAN_ACT, 0),
            WORKFLOW_PLAN_ACT_REPLAN: distribution.get(WORKFLOW_PLAN_ACT_REPLAN, 0),
        },
        "router_latency_ms": {
            "average": round(average_latency_ms, 3),
            "p95": round(p95_latency_ms, 3),
        },
        "mismatches": mismatches,
        "predictions": predictions,
    }


def _default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("docs/plans/baselines") / f"phase0-router-baseline-{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase-0 router baseline for workflow routing.")
    parser.add_argument(
        "--fixture",
        default="tests/evals/fixtures/multi_agent_eval_set_v1.jsonl",
        help="Path to evaluation JSONL fixture.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Defaults to docs/plans/baselines/phase0-router-baseline-<timestamp>.json",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(fixture_path)
    report = run_router_baseline(records)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"fixture: {fixture_path}")
    print(f"cases: {report['total_cases']}")
    print(f"match_rate: {report['match_rate']:.3f}")
    print(f"route_distribution: {report['route_distribution']}")
    print(f"latency_ms(avg/p95): {report['router_latency_ms']['average']}/{report['router_latency_ms']['p95']}")
    print(f"report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
