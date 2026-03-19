from __future__ import annotations

from typing import Any

from .contracts import AgentEvalCase, FinalAnswerJudge, FinalAnswerJudgeResult


def build_trajectory_llm_as_judge(
    *,
    model: Any,
    prompt: str | None = None,
) -> FinalAnswerJudge:
    try:
        from agentevals.trajectory.llm import create_trajectory_llm_as_judge
    except ImportError as exc:
        raise RuntimeError(
            "agentevals is required for LLM-as-judge evaluation. Install dev eval dependencies first."
        ) from exc

    evaluator = create_trajectory_llm_as_judge(model=model, prompt=prompt)

    def _judge(case: AgentEvalCase, normalized_result: dict[str, Any]) -> FinalAnswerJudgeResult:
        outputs = normalized_result.get("output_messages")
        if not isinstance(outputs, list) or not outputs:
            return FinalAnswerJudgeResult(
                passed=False,
                score=0.0,
                reasoning="No output messages available for trajectory judge.",
            )

        reference_outputs = None
        metadata = case.metadata if isinstance(case.metadata, dict) else {}
        raw_reference_outputs = metadata.get("reference_outputs")
        if isinstance(raw_reference_outputs, list):
            reference_outputs = raw_reference_outputs
        elif isinstance(raw_reference_outputs, dict):
            messages = raw_reference_outputs.get("messages")
            if isinstance(messages, list):
                reference_outputs = messages

        result = (
            evaluator(outputs=outputs, reference_outputs=reference_outputs)
            if reference_outputs is not None
            else evaluator(outputs=outputs)
        )
        if not isinstance(result, dict):
            raise TypeError("Trajectory judge returned a non-dict result.")

        return FinalAnswerJudgeResult(
            passed=bool(result.get("score")),
            score=1.0 if bool(result.get("score")) else 0.0,
            reasoning=str(result.get("comment") or result.get("reasoning") or "").strip(),
        )

    return _judge
