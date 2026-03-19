from __future__ import annotations

import json
from pathlib import Path

from .contracts import AgentEvalCase


def load_eval_cases(path: str | Path) -> list[AgentEvalCase]:
    fixture_path = Path(path)
    rows: list[AgentEvalCase] = []
    for line_no, raw_line in enumerate(
        fixture_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_no} is not a JSON object.")
        try:
            rows.append(AgentEvalCase.from_dict(payload))
        except ValueError as exc:
            raise ValueError(f"Invalid eval fixture at line {line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"No eval cases found in fixture: {fixture_path}")
    return rows
