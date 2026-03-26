#!/usr/bin/env python3
"""Block Claude-triggered git commits unless local quality gates pass."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

GATE_COMMANDS: tuple[str, ...] = (
    "bash scripts/quality_gate.sh core",
    "bash scripts/quality_gate.sh full",
    "uv run --extra dev python -m pytest tests/unit -q",
)

_GIT_COMMIT_PATTERN = re.compile(r"\bgit(?:\s+-\S+)*\s+commit\b")


def _is_git_commit(payload: dict[str, object]) -> bool:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command")
    if not isinstance(command, str):
        return False
    return bool(_GIT_COMMIT_PATTERN.search(command))


def _run_gate(command: str, cwd: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output.strip()


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    if not isinstance(payload, dict) or not _is_git_commit(payload):
        return 0

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())).resolve()

    for command in GATE_COMMANDS:
        success, output = _run_gate(command, project_dir)
        if success:
            continue

        print("Claude git commit blocked: required quality gates failed.", file=sys.stderr)
        print(f"Failed command: {command}", file=sys.stderr)
        if output:
            print(output, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
