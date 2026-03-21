from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LEGACY_KEYWORD_CONTRACT_KEYS = (
    "expected_answer_all_of",
    "expected_answer_any_of",
    "forbidden_answer_any_of",
)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


@dataclass(frozen=True)
class FinalAnswerContract:
    success_rubric: str = ""

    @property
    def has_success_contract(self) -> bool:
        return bool(self.success_rubric.strip())


@dataclass(frozen=True)
class ProcessContract:
    requires_evidence: bool = False
    min_evidence_count: int = 0
    require_plan: bool = False
    require_todos: bool = False
    min_execution_completion_ratio: float | None = None
    required_tool_names: list[str] = field(default_factory=list)
    required_phase_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentEvalCase:
    case_id: str
    category: str
    prompt: str
    final_answer_contract: FinalAnswerContract
    process_contract: ProcessContract
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentEvalCase":
        case_id = str(payload.get("id") or "").strip()
        category = str(payload.get("category") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        if not case_id or not category or not prompt:
            raise ValueError("Eval case must include non-empty id, category, and prompt.")

        legacy_keys = [key for key in LEGACY_KEYWORD_CONTRACT_KEYS if key in payload]
        if legacy_keys:
            raise ValueError(
                "Eval case "
                f"'{case_id}' uses deprecated keyword matching fields {legacy_keys}. "
                "Final-answer evals must use success_rubric and LLM judge only."
            )

        final_answer_contract = FinalAnswerContract(
            success_rubric=str(payload.get("success_rubric") or "").strip(),
        )
        if not final_answer_contract.has_success_contract:
            raise ValueError(
                f"Eval case '{case_id}' must define a non-empty success_rubric."
            )

        raw_min_ratio = payload.get("min_execution_completion_ratio")
        min_ratio: float | None
        if raw_min_ratio is None:
            min_ratio = None
        else:
            min_ratio = float(raw_min_ratio)

        process_contract = ProcessContract(
            requires_evidence=bool(payload.get("requires_evidence", False)),
            min_evidence_count=max(0, int(payload.get("min_evidence_count", 0))),
            require_plan=bool(payload.get("require_plan", False)),
            require_todos=bool(payload.get("require_todos", False)),
            min_execution_completion_ratio=min_ratio,
            required_tool_names=_string_list(payload.get("required_tool_names")),
            required_phase_labels=_string_list(payload.get("required_phase_labels")),
        )
        metadata = {
            str(key): value
            for key, value in payload.items()
            if key
            not in {
                "id",
                "category",
                "prompt",
                *LEGACY_KEYWORD_CONTRACT_KEYS,
                "success_rubric",
                "requires_evidence",
                "min_evidence_count",
                "require_plan",
                "require_todos",
                "min_execution_completion_ratio",
                "required_tool_names",
                "required_phase_labels",
            }
        }
        return cls(
            case_id=case_id,
            category=category,
            prompt=prompt,
            final_answer_contract=final_answer_contract,
            process_contract=process_contract,
            metadata=metadata,
        )


@dataclass(frozen=True)
class FinalAnswerJudgeResult:
    passed: bool
    score: float | None = None
    reasoning: str = ""


FinalAnswerJudge = Callable[[AgentEvalCase, dict[str, Any]], FinalAnswerJudgeResult]
