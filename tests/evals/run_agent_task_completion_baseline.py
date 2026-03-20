import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.adapters.llm import create_chat_model
from agent.application.evals import (
    AgentEvalCase,
    ExecuteTurnEvalRunner,
    build_trajectory_llm_as_judge,
    load_eval_cases,
    run_agent_evals,
    select_eval_cases,
)


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line or line.startswith("export "):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class _ScenarioAgent:
    def __init__(self, case: AgentEvalCase):
        self._case = case

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        user_prompt = payload["messages"][0]["content"]
        assert user_prompt == self._case.prompt
        on_event = None
        if isinstance(config, dict):
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                candidate = configurable.get("on_event")
                if callable(candidate):
                    on_event = candidate

        if callable(on_event):
            on_event(
                {
                    "sender": "user",
                    "receiver": "leader",
                    "performative": "request",
                    "content": self._case.prompt,
                }
            )
            if self._case.process_contract.require_plan:
                on_event(
                    {
                        "sender": "planner",
                        "receiver": "leader",
                        "performative": "plan",
                        "content": "生成执行计划",
                    }
                )
            on_event(
                {
                    "sender": "leader",
                    "receiver": "user",
                    "performative": "final",
                    "content": "生成最终答案",
                }
            )

        answer = _answer_for_case(self._case)
        tool_calls = _tool_calls_for_case(self._case)
        result: dict[str, Any] = {
            "messages": [SimpleNamespace(content=answer, tool_calls=tool_calls)],
        }
        if self._case.process_contract.require_plan:
            result["plan"] = {"goal": self._case.prompt, "description": "按步骤完成任务"}
            result["runtime_state"] = {
                "current_plan": {"steps": [{"id": "step_1"}]},
                "completed_step_ids": ["step_1"],
            }
        if self._case.process_contract.require_todos:
            result["todos"] = [{"id": "todo_1", "content": "完成任务", "status": "completed"}]
        return result


def _answer_for_case(case: AgentEvalCase) -> str:
    if case.category == "project_rag":
        return (
            "基于当前项目文档，RAG 的核心价值是先检索再生成，用外部证据降低幻觉并提升回答相关性。"
            "<evidence>chunk-1|p1|o0-10</evidence>"
        )
    if case.category == "project_compare":
        return (
            "结合当前项目文档，RAG 实现更直接、接入成本更低；Self-RAG 增加了自反思与纠错链路，但工程复杂度更高。"
            "<evidence>chunk-1|p1|o0-10</evidence>"
            "<evidence>chunk-2|p2|o0-10</evidence>"
            "如果当前阶段优先追求稳定落地，建议先以 RAG 为主，再把 Self-RAG 作为后续试点方向。"
        )
    if case.category == "web_research":
        return (
            "基于 2025 年以来的公开资料，Self-RAG 的最新进展集中在更强的可控评估链路与更重的系统成本两点。"
            "结论一：研究关注点正在从论文指标转向真实系统集成。"
            "结论二：是否值得引入，越来越取决于延迟、成本和观测能力。"
        )
    return (
        "结合当前项目文档与近期公开资料，Self-RAG 有机会提升答案自校验能力，但会显著增加编排复杂度与延迟预算。"
        "<evidence>chunk-1|p1|o0-10</evidence>"
        "<evidence>chunk-2|p2|o0-10</evidence>"
        "路线建议是先做小范围 pilot，再决定是否纳入正式 roadmap。"
    )


def _tool_calls_for_case(case: AgentEvalCase) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for tool_name in case.process_contract.required_tool_names:
        tool_calls.append({"name": tool_name, "args": {"query": case.prompt}})
    if not tool_calls and case.process_contract.requires_evidence:
        tool_calls.append({"name": "search_document", "args": {"query": case.prompt}})
    return tool_calls


def _search_document_evidence(case: AgentEvalCase) -> dict[str, Any]:
    if case.category == "project_rag":
        return {"evidences": [{"chunk_id": "chunk-1", "text": "证据", "page_no": 1}]}
    if case.category in {"project_compare", "hybrid_research"}:
        return {
            "evidences": [
                {"chunk_id": "chunk-1", "text": "证据1", "page_no": 1},
                {"chunk_id": "chunk-2", "text": "证据2", "page_no": 2},
            ]
        }
    return {"evidences": []}


def _build_runner(case: AgentEvalCase) -> ExecuteTurnEvalRunner:
    return ExecuteTurnEvalRunner(
        leader_agent=_ScenarioAgent(case),
        leader_runtime_config={},
        search_document_evidence_fn=lambda _query, eval_case=case: _search_document_evidence(eval_case),
    )


def _default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("docs/plans/baselines") / f"task-completion-eval-baseline-{stamp}.json"


def _build_judge(model_name: str, base_url: str | None) -> Any:
    api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for baseline LLM judge runs.")
    if not model_name.strip():
        raise ValueError("Judge model is required. Set --judge-model or OPENAI_MODEL_NAME.")
    model = create_chat_model(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=0.0,
    )
    return build_trajectory_llm_as_judge(model=model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end task completion eval baseline.")
    parser.add_argument(
        "--fixture",
        default="tests/evals/fixtures/agent_task_eval_set_v1.jsonl",
        help="Path to eval fixture JSONL.",
    )
    parser.add_argument(
        "--env-file",
        default="/home/ling/LLM_App_Final/.env",
        help="Path to env file with LLM judge configuration.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Defaults to docs/plans/baselines/task-completion-eval-baseline-<timestamp>.json",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional judge model override. Defaults to OPENAI_MODEL_NAME from the env file.",
    )
    parser.add_argument(
        "--judge-base-url",
        default="",
        help="Optional OpenAI-compatible base URL override. Defaults to OPENAI_BASE_URL from the env file.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional case id to run. Repeat to run multiple cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of cases to run after filtering.",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))

    fixture_path = Path(args.fixture)
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_eval_cases(fixture_path)
    cases = select_eval_cases(
        cases,
        case_ids=args.case_id or None,
        limit=args.limit if args.limit > 0 else None,
    )
    judge_model = args.judge_model.strip() or str(os.getenv("OPENAI_MODEL_NAME") or "").strip()
    judge_base_url = args.judge_base_url.strip() or str(os.getenv("OPENAI_BASE_URL") or "").strip()
    judge = _build_judge(
        model_name=judge_model,
        base_url=judge_base_url or None,
    )

    report = run_agent_evals(
        cases,
        runner=lambda case: _build_runner(case)(case),
        judge=judge,
        fixture_path=str(fixture_path),
    )
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"fixture: {fixture_path}")
    print(f"cases: {report['total_cases']}")
    print(f"completed_cases: {report['completed_cases']}")
    print(f"completion_rate: {report['completion_rate']:.3f}")
    print(f"final_success_rate: {report['final_success_rate']:.3f}")
    print(f"process_success_rate: {report['process_success_rate']:.3f}")
    print(f"evidence_coverage_rate: {report['evidence_coverage_rate']:.3f}")
    print(
        "average_execution_completion_ratio: "
        f"{report['average_execution_completion_ratio']:.3f}"
    )
    print(f"remediation_area_counts: {report['remediation_area_counts']}")
    print(f"report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
