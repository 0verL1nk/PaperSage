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
        name="bash",
        description="Run bounded bash commands inside workspace for engineering tasks.",
    ),
    ToolMetadata(
        name="ask_human",
        description="Ask user for clarification/confirmation when autonomous action is risky.",
    ),
    ToolMetadata(
        name="read_file",
        description="Read file content from workspace with offset and limit support.",
    ),
    ToolMetadata(
        name="write_file",
        description="Write or append content to file in workspace.",
    ),
    ToolMetadata(
        name="edit_file",
        description="Replace text in file with exact string matching.",
    ),
    ToolMetadata(
        name="update_file",
        description="Update specific line range in file.",
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

    if "read_file" in enabled_tool_names:
        @tool(
            "read_file",
            description="Read file content from workspace.",
            args_schema=ReadFileInput,
        )
        def read_file(path: str, offset: int = 0, limit: int = 4000) -> str:
            resolved, error = _resolve_workspace_path(path)
            if resolved is None:
                return str(error or "Invalid path.")
            if not resolved.exists():
                return f"File not found: {_display_workspace_path(resolved)}"
            if not resolved.is_file():
                return f"Not a file: {_display_workspace_path(resolved)}"
            try:
                content = resolved.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Failed to read file: {exc}"
            safe_offset = max(0, int(offset))
            safe_limit = max(1, min(int(limit), 20000))
            return content[safe_offset:safe_offset + safe_limit]

        tools.append(read_file)

    if "write_file" in enabled_tool_names:
        @tool(
            "write_file",
            description="Write content to file in workspace.",
            args_schema=WriteFileInput,
        )
        def write_file(path: str, content: str, append: bool = False) -> str:
            resolved, error = _resolve_workspace_path(path)
            if resolved is None:
                return str(error or "Invalid path.")
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if append else "w"
                resolved.write_text(content, encoding="utf-8") if not append else resolved.open(mode, encoding="utf-8").write(content)
                return f"Written to {_display_workspace_path(resolved)}"
            except Exception as exc:
                return f"Failed to write file: {exc}"

        tools.append(write_file)

    if "edit_file" in enabled_tool_names:
        @tool(
            "edit_file",
            description="Replace text in file.",
            args_schema=EditFileInput,
        )
        def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
            resolved, error = _resolve_workspace_path(path)
            if resolved is None:
                return str(error or "Invalid path.")
            if not resolved.exists():
                return f"File not found: {_display_workspace_path(resolved)}"
            try:
                content = resolved.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Failed to read file: {exc}"
            if old_text not in content:
                return "Text not found in file."
            new_content = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
            try:
                resolved.write_text(new_content, encoding="utf-8")
                return f"Edited {_display_workspace_path(resolved)}"
            except Exception as exc:
                return f"Failed to write file: {exc}"

        tools.append(edit_file)

    if "update_file" in enabled_tool_names:
        @tool(
            "update_file",
            description="Update lines in file.",
            args_schema=UpdateFileInput,
        )
        def update_file(path: str, start_line: int, end_line: int, new_text: str) -> str:
            resolved, error = _resolve_workspace_path(path)
            if resolved is None:
                return str(error or "Invalid path.")
            if not resolved.exists():
                return f"File not found: {_display_workspace_path(resolved)}"
            try:
                lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)
            except Exception as exc:
                return f"Failed to read file: {exc}"
            if start_line < 1 or end_line < start_line or end_line > len(lines):
                return f"Invalid line range: {start_line}-{end_line} (file has {len(lines)} lines)"
            new_lines = lines[:start_line-1] + [new_text if new_text.endswith("\n") else new_text + "\n"] + lines[end_line:]
            try:
                resolved.write_text("".join(new_lines), encoding="utf-8")
                return f"Updated {_display_workspace_path(resolved)}"
            except Exception as exc:
                return f"Failed to write file: {exc}"

        tools.append(update_file)

    return tools
