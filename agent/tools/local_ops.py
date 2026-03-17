import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .types import ToolMetadata


class ReadFileInput(BaseModel):
    path: str = Field(description="Relative or absolute file path under workspace root.")
    offset: int = Field(
        default=0,
        ge=0,
        description="Character offset to start reading from.",
    )
    limit: int = Field(
        default=4000,
        ge=1,
        le=20000,
        description="Maximum characters to read.",
    )


class WriteFileInput(BaseModel):
    path: str = Field(description="Relative or absolute file path under workspace root.")
    content: str = Field(description="Full content to write (or append).")
    append: bool = Field(
        default=False,
        description="Whether to append content instead of overwrite.",
    )


class EditFileInput(BaseModel):
    path: str = Field(description="Relative or absolute file path under workspace root.")
    old_text: str = Field(description="Exact text to replace.")
    new_text: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False,
        description="Whether to replace all matches or only first match.",
    )


class UpdateFileInput(BaseModel):
    path: str = Field(description="Relative or absolute file path under workspace root.")
    start_line: int = Field(ge=1, description="1-based start line.")
    end_line: int = Field(ge=1, description="1-based end line (inclusive).")
    new_text: str = Field(description="Replacement text block.")


class BashInput(BaseModel):
    command: str = Field(description="Bash command to execute.")
    cwd: str = Field(
        default=".",
        description="Working directory under workspace root.",
    )
    timeout_seconds: int = Field(
        default=20,
        ge=1,
        le=120,
        description="Command timeout in seconds.",
    )
    max_output_chars: int = Field(
        default=8000,
        ge=200,
        le=50000,
        description="Maximum stdout/stderr chars captured in response.",
    )


class AskHumanInput(BaseModel):
    question: str = Field(description="Question that needs human confirmation or input.")
    context: str = Field(
        default="",
        description="Optional brief context for the question.",
    )
    urgency: str = Field(
        default="normal",
        description="Urgency level: low, normal, high.",
    )


DEFAULT_FILE_READ_LIMIT = 4000
_DANGEROUS_BASH_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r":\(\)\s*\{",
    r">\s*/dev/",
]


LOCAL_OPS_TOOL_METADATA = (
    ToolMetadata(
        name="read_file",
        description="Read file content from workspace file system with offset/limit.",
    ),
    ToolMetadata(
        name="write_file",
        description="Write or append text content to a workspace file.",
    ),
    ToolMetadata(
        name="edit_file",
        description="Edit a workspace file by replacing exact text snippets.",
    ),
    ToolMetadata(
        name="update_file",
        description="Update a workspace file by replacing a line range.",
    ),
    ToolMetadata(
        name="bash",
        description="Run bounded bash commands inside workspace for engineering tasks.",
    ),
    ToolMetadata(
        name="ask_human",
        description="Ask user for clarification/confirmation when autonomous action is risky.",
    ),
)


def _env_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _get_file_tools_root() -> Path:
    configured = _env_value("AGENT_FILE_TOOLS_ROOT", default="")
    root = Path(configured) if configured else Path.cwd()
    try:
        return root.resolve()
    except Exception:
        return root.absolute()


def _resolve_workspace_path(path_value: str) -> tuple[Path | None, str | None]:
    raw = str(path_value or "").strip()
    if not raw:
        return None, "Path is empty."

    root = _get_file_tools_root()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve()
    except Exception as exc:
        return None, f"Failed to resolve path: {exc}"

    try:
        resolved.relative_to(root)
    except Exception:
        return None, f"Blocked by file policy: path is outside workspace root ({root})."
    return resolved, None


def _display_workspace_path(path_obj: Path) -> str:
    root = _get_file_tools_root()
    try:
        return path_obj.relative_to(root).as_posix()
    except Exception:
        return path_obj.as_posix()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_dangerous_bash_command(command: str) -> bool:
    lowered = str(command or "").lower()
    return any(re.search(pattern, lowered) for pattern in _DANGEROUS_BASH_PATTERNS)


def build_local_ops_tools(*, enabled_tool_names: set[str]) -> list[Any]:
    tools: list[Any] = []

    if "read_file" in enabled_tool_names:
        @tool(
            "read_file",
            description="Read file content from workspace file system with offset/limit.",
            args_schema=ReadFileInput,
        )
        def read_file(path: str, offset: int = 0, limit: int = DEFAULT_FILE_READ_LIMIT) -> str:
            safe_limit = max(1, min(int(limit), 20000))
            safe_offset = max(0, int(offset))
            resolved_path, error_text = _resolve_workspace_path(path)
            if resolved_path is None:
                return str(error_text or "Invalid path.")
            if not resolved_path.exists():
                return f"File not found: {_display_workspace_path(resolved_path)}"
            if not resolved_path.is_file():
                return f"Not a file: {_display_workspace_path(resolved_path)}"
            try:
                content = resolved_path.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Failed to read file: {exc}"
            snippet = content[safe_offset : safe_offset + safe_limit]
            return (
                f"Path: {_display_workspace_path(resolved_path)}\n"
                f"Total chars: {len(content)}\n"
                f"Offset: {safe_offset}\n"
                f"Limit: {safe_limit}\n\n"
                f"{snippet}"
            )

        tools.append(read_file)

    if "write_file" in enabled_tool_names:
        @tool(
            "write_file",
            description="Write or append text content to a workspace file.",
            args_schema=WriteFileInput,
        )
        def write_file(path: str, content: str, append: bool = False) -> str:
            resolved_path, error_text = _resolve_workspace_path(path)
            if resolved_path is None:
                return str(error_text or "Invalid path.")
            try:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                if append:
                    with resolved_path.open("a", encoding="utf-8") as handle:
                        handle.write(content)
                else:
                    resolved_path.write_text(content, encoding="utf-8")
            except Exception as exc:
                return f"Failed to write file: {exc}"
            action = "Appended" if append else "Wrote"
            return (
                f"{action} {len(content)} chars to "
                f"{_display_workspace_path(resolved_path)}"
            )

        tools.append(write_file)

    if "edit_file" in enabled_tool_names:
        @tool(
            "edit_file",
            description="Edit a workspace file by replacing exact text snippets.",
            args_schema=EditFileInput,
        )
        def edit_file(
            path: str,
            old_text: str,
            new_text: str,
            replace_all: bool = False,
        ) -> str:
            if not old_text:
                return "old_text must not be empty."
            resolved_path, error_text = _resolve_workspace_path(path)
            if resolved_path is None:
                return str(error_text or "Invalid path.")
            if not resolved_path.exists() or not resolved_path.is_file():
                return f"File not found: {_display_workspace_path(resolved_path)}"
            try:
                content = resolved_path.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Failed to read file: {exc}"

            matches = content.count(old_text)
            if matches <= 0:
                return (
                    f"Target text not found in "
                    f"{_display_workspace_path(resolved_path)}."
                )
            if replace_all:
                updated = content.replace(old_text, new_text)
                replaced = matches
            else:
                updated = content.replace(old_text, new_text, 1)
                replaced = 1
            try:
                resolved_path.write_text(updated, encoding="utf-8")
            except Exception as exc:
                return f"Failed to write file: {exc}"
            return (
                f"Replaced {replaced} occurrence(s) in "
                f"{_display_workspace_path(resolved_path)}."
            )

        tools.append(edit_file)

    if "update_file" in enabled_tool_names:
        @tool(
            "update_file",
            description="Update a workspace file by replacing a line range.",
            args_schema=UpdateFileInput,
        )
        def update_file(path: str, start_line: int, end_line: int, new_text: str) -> str:
            safe_start = int(start_line)
            safe_end = int(end_line)
            if safe_end < safe_start:
                return "end_line must be greater than or equal to start_line."
            resolved_path, error_text = _resolve_workspace_path(path)
            if resolved_path is None:
                return str(error_text or "Invalid path.")
            if not resolved_path.exists() or not resolved_path.is_file():
                return f"File not found: {_display_workspace_path(resolved_path)}"

            try:
                content = resolved_path.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Failed to read file: {exc}"

            lines = content.splitlines()
            total_lines = len(lines)
            if safe_start < 1 or safe_start > max(total_lines, 1):
                return f"start_line out of range: {safe_start} (total lines: {total_lines})"
            if safe_end < 1 or safe_end > max(total_lines, 1):
                return f"end_line out of range: {safe_end} (total lines: {total_lines})"

            replacement_lines = new_text.splitlines()
            merged = lines[: safe_start - 1] + replacement_lines + lines[safe_end:]
            new_content = "\n".join(merged)
            if content.endswith("\n") and new_content:
                new_content += "\n"
            try:
                resolved_path.write_text(new_content, encoding="utf-8")
            except Exception as exc:
                return f"Failed to write file: {exc}"
            return (
                f"Updated lines {safe_start}-{safe_end} in "
                f"{_display_workspace_path(resolved_path)}."
            )

        tools.append(update_file)

    if "bash" in enabled_tool_names:
        @tool(
            "bash",
            description="Run bounded bash commands inside workspace for engineering tasks.",
            args_schema=BashInput,
        )
        def bash(
            command: str,
            cwd: str = ".",
            timeout_seconds: int = 20,
            max_output_chars: int = 8000,
        ) -> str:
            safe_command = str(command or "").strip()
            if not safe_command:
                return "Command is empty."
            if _is_dangerous_bash_command(safe_command):
                return "Blocked by bash policy: command appears destructive."

            resolved_cwd, error_text = _resolve_workspace_path(cwd)
            if resolved_cwd is None:
                return str(error_text or "Invalid cwd.")
            if not resolved_cwd.exists() or not resolved_cwd.is_dir():
                return f"Working directory not found: {_display_workspace_path(resolved_cwd)}"

            safe_timeout = max(1, min(int(timeout_seconds), 120))
            safe_max_output = max(200, min(int(max_output_chars), 50000))
            try:
                completed = subprocess.run(
                    ["bash", "-lc", safe_command],
                    cwd=str(resolved_cwd),
                    capture_output=True,
                    text=True,
                    timeout=safe_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return f"Command timed out after {safe_timeout}s."
            except Exception as exc:
                return f"Failed to execute command: {exc}"

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            combined_len = len(stdout) + len(stderr)
            if len(stdout) > safe_max_output:
                stdout = stdout[:safe_max_output] + "\n...(truncated)"
            if len(stderr) > safe_max_output:
                stderr = stderr[:safe_max_output] + "\n...(truncated)"
            payload = {
                "exit_code": int(completed.returncode),
                "cwd": _display_workspace_path(resolved_cwd),
                "command": safe_command,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": combined_len > (safe_max_output * 2),
            }
            return json.dumps(payload, ensure_ascii=False)

        tools.append(bash)

    if "ask_human" in enabled_tool_names:
        @tool(
            "ask_human",
            description="Ask user for clarification/confirmation when autonomous action is risky.",
            args_schema=AskHumanInput,
        )
        def ask_human(question: str, context: str = "", urgency: str = "normal") -> str:
            prompt = str(question or "").strip()
            if not prompt:
                return "question must not be empty."
            urgency_value = str(urgency or "normal").strip().lower()
            if urgency_value not in {"low", "normal", "high"}:
                urgency_value = "normal"
            payload = {
                "type": "ask_human",
                "question": prompt,
                "context": str(context or "").strip(),
                "urgency": urgency_value,
                "ts": _now_iso(),
            }
            return json.dumps(payload, ensure_ascii=False)

        tools.append(ask_human)

    return tools
