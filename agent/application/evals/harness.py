from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..turn_engine import execute_turn_core
from .contracts import AgentEvalCase, FinalAnswerJudge
from .reporting import build_eval_report
from .scoring import evaluate_case_result


@dataclass(frozen=True)
class ExecuteTurnEvalRunner:
    leader_agent: Any
    leader_runtime_config: dict[str, Any]
    leader_llm: Any | None = None
    policy_llm: Any | None = None
    search_document_evidence_fn: Any | None = None
    leader_tool_specs: list[dict[str, Any]] = field(default_factory=list)
    routing_context: str = ""

    def __call__(self, case: AgentEvalCase) -> dict[str, Any]:
        return execute_turn_core(
            prompt=case.prompt,
            hinted_prompt=case.prompt,
            leader_agent=self.leader_agent,
            leader_runtime_config=dict(self.leader_runtime_config),
            leader_llm=self.leader_llm,
            policy_llm=self.policy_llm,
            search_document_evidence_fn=self.search_document_evidence_fn,
            leader_tool_specs=list(self.leader_tool_specs),
            routing_context=self.routing_context,
        )


def run_agent_evals(
    cases: list[AgentEvalCase],
    *,
    runner,
    judge: FinalAnswerJudge | None,
    fixture_path: str = "",
) -> dict[str, Any]:
    if judge is None:
        raise ValueError("A final-answer LLM judge is required for task-completion eval runs.")

    case_results: list[dict[str, Any]] = []
    for case in cases:
        turn_result = runner(case)
        case_results.append(evaluate_case_result(case, turn_result, judge=judge))
    return build_eval_report(fixture_path=fixture_path, case_results=case_results)
