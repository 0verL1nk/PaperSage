from __future__ import annotations

import json
from typing import Any

from .contracts import AgentEvalCase, FinalAnswerJudge, FinalAnswerJudgeResult

FINAL_ANSWER_JUDGE_PROMPT = """You are evaluating whether an AI research assistant successfully completed a user task.\n\nJudge only the user-visible final answer quality against the supplied task and success rubric.\nIgnore private middleware details, exact internal decomposition, and implementation-specific routing choices.\nDo not require exact keyword overlap unless the rubric explicitly requires it.\n\n<UserRequest>\n{inputs}\n</UserRequest>\n\n<SuccessRubric>\n{rubric}\n</SuccessRubric>\n\n<ReferenceOutputs>\n{reference_outputs}\n</ReferenceOutputs>\n\n<AgentOutputs>\n{outputs}\n</AgentOutputs>\n"""


def _stable_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return dict(payload)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return dict(value) if isinstance(value, dict) else None
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, dict) else None
    return None


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    payload = _stable_dict(tool_call)
    if payload is None:
        return None
    if "function" in payload:
        function_payload = payload.get("function")
        if not isinstance(function_payload, dict):
            return None
        normalized = dict(payload)
        normalized["id"] = str(normalized.get("id") or "")
        normalized["type"] = str(normalized.get("type") or "function")
        return normalized

    name = str(payload.get("name") or "").strip()
    if not name:
        return None
    arguments = payload.get("args")
    if isinstance(arguments, str):
        arguments_text = arguments
    else:
        arguments_text = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    return {
        "id": str(payload.get("id") or ""),
        "type": "function",
        "function": {"name": name, "arguments": arguments_text},
    }


def _normalize_output_message(message: Any) -> dict[str, Any] | None:
    payload = _stable_dict(message)
    if payload is None:
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        role = getattr(message, "role", None) or getattr(message, "type", None) or "assistant"
        payload = {
            "role": str(role or "assistant"),
            "content": "" if content is None else str(content),
        }
        if isinstance(tool_calls, list):
            payload["tool_calls"] = tool_calls

    role = str(payload.get("role") or payload.get("type") or "assistant").strip().lower()
    if role == "ai":
        role = "assistant"
    normalized: dict[str, Any] = {
        "role": role or "assistant",
        "content": "" if payload.get("content") is None else str(payload.get("content")),
    }

    raw_tool_calls = payload.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        normalized_tool_calls = [
            normalized_tool_call
            for item in raw_tool_calls
            if (normalized_tool_call := _normalize_tool_call(item)) is not None
        ]
        normalized["tool_calls"] = normalized_tool_calls

    if normalized["role"] == "tool":
        normalized["tool_call_id"] = str(payload.get("tool_call_id") or "")
    return normalized


def _normalize_output_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        normalized = _normalize_output_message(message)
        if normalized is not None:
            normalized_messages.append(normalized)
    return normalized_messages


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

    evaluator_kwargs: dict[str, Any] = {
        "prompt": prompt or FINAL_ANSWER_JUDGE_PROMPT,
    }
    if isinstance(model, str):
        evaluator_kwargs["model"] = model
    else:
        evaluator_kwargs["judge"] = model

    evaluator = create_trajectory_llm_as_judge(**evaluator_kwargs)

    def _judge(case: AgentEvalCase, normalized_result: dict[str, Any]) -> FinalAnswerJudgeResult:
        outputs = _normalize_output_messages(normalized_result.get("output_messages"))
        if not outputs:
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

        result = evaluator(
            outputs=outputs,
            reference_outputs=reference_outputs,
            inputs=case.prompt,
            rubric=case.final_answer_contract.success_rubric,
        )
        if not isinstance(result, dict):
            raise TypeError("Trajectory judge returned a non-dict result.")

        return FinalAnswerJudgeResult(
            passed=bool(result.get("score")),
            score=1.0 if bool(result.get("score")) else 0.0,
            reasoning=str(result.get("comment") or result.get("reasoning") or "").strip(),
        )

    return _judge
