import json
import os
import re
import subprocess
from datetime import UTC, datetime
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


class WriteTodoInput(BaseModel):
    action: str = Field(
        default="upsert",
        description="Todo action: upsert, update_status, append_note, complete.",
    )
    todo_id: str = Field(
        default="",
        description="Stable todo id. Leave empty to auto-generate on first insert.",
    )
    title: str = Field(
        default="",
        description="Todo title (required for creating new todo).",
    )
    details: str = Field(
        default="",
        description="Detailed task description.",
    )
    status: str = Field(
        default="todo",
        description="Task status: todo, in_progress, done, blocked, canceled.",
    )
    priority: str = Field(
        default="medium",
        description="Task priority: low, medium, high.",
    )
    assignee: str = Field(
        default="",
        description="Optional assignee (agent/user identifier).",
    )
    dependencies: list[str] | None = Field(
        default=None,
        description="Optional dependency todo ids. Empty list clears dependencies.",
    )
    plan_id: str = Field(
        default="",
        description="Optional plan identifier for plan-mode grouping.",
    )
    step_ref: str = Field(
        default="",
        description="Optional step reference (e.g. 1, step_2).",
    )
    note: str = Field(
        default="",
        description="Optional note to append into history.",
    )
    file_path: str = Field(
        default=".agent/todo.json",
        description="Todo JSON file path under workspace root.",
    )


class EditTodoInput(BaseModel):
    todo_id: str = Field(description="Existing todo id to edit.")
    title: str = Field(
        default="",
        description="Optional updated title.",
    )
    details: str = Field(
        default="",
        description="Optional updated details.",
    )
    status: str = Field(
        default="",
        description="Optional updated status: todo, in_progress, done, blocked, canceled.",
    )
    priority: str = Field(
        default="",
        description="Optional updated priority: low, medium, high.",
    )
    assignee: str = Field(
        default="",
        description="Optional updated assignee.",
    )
    dependencies: list[str] | None = Field(
        default=None,
        description="Optional updated dependencies. Empty list clears dependencies.",
    )
    plan_id: str = Field(
        default="",
        description="Optional updated plan id.",
    )
    step_ref: str = Field(
        default="",
        description="Optional updated step ref.",
    )
    note: str = Field(
        default="",
        description="Optional note appended to history.",
    )
    file_path: str = Field(
        default=".agent/todo.json",
        description="Todo JSON file path under workspace root.",
    )


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
VALID_TODO_STATUSES = {"todo", "in_progress", "done", "blocked", "canceled"}
VALID_TODO_PRIORITIES = {"low", "medium", "high"}
_TODO_ID_PATTERN = re.compile(r"[^a-z0-9_-]+")
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
        name="write_todo",
        description="Create or update structured todo items for plan-mode execution tracking.",
    ),
    ToolMetadata(
        name="edit_todo",
        description="Edit existing todo items by id with status/priority/notes updates.",
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


def _normalize_todo_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in VALID_TODO_STATUSES:
        return normalized
    return "todo"


def _normalize_todo_priority(priority: str) -> str:
    normalized = str(priority or "").strip().lower()
    if normalized in VALID_TODO_PRIORITIES:
        return normalized
    return "medium"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_todo_id(raw_id: str) -> str:
    normalized = _TODO_ID_PATTERN.sub("_", str(raw_id or "").strip().lower()).strip("_")
    return normalized[:64]


def _generate_todo_id(title: str) -> str:
    stem = _normalize_todo_id(title) or "todo"
    return f"{stem}_{int(datetime.now(UTC).timestamp())}"


def _normalize_todo_dependencies(
    dependencies: list[str] | None,
    *,
    self_id: str = "",
) -> list[str]:
    if dependencies is None:
        return []
    normalized_self = _normalize_todo_id(self_id)
    output: list[str] = []
    seen: set[str] = set()
    for raw in dependencies:
        dep_id = _normalize_todo_id(raw)
        if not dep_id:
            continue
        if normalized_self and dep_id == normalized_self:
            continue
        if dep_id in seen:
            continue
        seen.add(dep_id)
        output.append(dep_id)
    return output


def _load_todo_store(path_obj: Path) -> list[dict[str, Any]]:
    if not path_obj.exists():
        return []
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        records.append(item)
    return records


def _save_todo_store(path_obj: Path, records: list[dict[str, Any]]) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_todo_index(records: list[dict[str, Any]], todo_id: str) -> int:
    normalized_id = _normalize_todo_id(todo_id)
    if not normalized_id:
        return -1
    for idx, item in enumerate(records):
        if str(item.get("id") or "").strip() == normalized_id:
            return idx
    return -1


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

    if "write_todo" in enabled_tool_names:
        @tool(
            "write_todo",
            description="Append or update structured todo items for plan-mode execution tracking.",
            args_schema=WriteTodoInput,
        )
        def write_todo(
            action: str = "upsert",
            todo_id: str = "",
            title: str = "",
            details: str = "",
            status: str = "todo",
            priority: str = "medium",
            assignee: str = "",
            dependencies: list[str] | None = None,
            plan_id: str = "",
            step_ref: str = "",
            note: str = "",
            file_path: str = ".agent/todo.json",
        ) -> str:
            normalized_action = str(action or "upsert").strip().lower()
            if normalized_action not in {"upsert", "update_status", "append_note", "complete"}:
                return "Unsupported action. Use one of: upsert, update_status, append_note, complete."

            resolved_path, error_text = _resolve_workspace_path(file_path)
            if resolved_path is None:
                return str(error_text or "Invalid file_path.")

            records = _load_todo_store(resolved_path)
            normalized_id = _normalize_todo_id(todo_id)
            now = _now_iso()
            index = -1
            for idx, item in enumerate(records):
                if str(item.get("id") or "").strip() == normalized_id and normalized_id:
                    index = idx
                    break

            if normalized_action == "complete":
                normalized_action = "update_status"
                status = "done"

            if index < 0 and normalized_action in {"update_status", "append_note"}:
                return "Todo id not found. Use action=upsert to create first."

            if index < 0:
                todo_title = str(title or "").strip()
                if not todo_title:
                    return "title is required when creating a todo item."
                normalized_id = normalized_id or _generate_todo_id(todo_title)
                record = {
                    "id": normalized_id,
                    "title": todo_title,
                    "details": str(details or "").strip(),
                    "status": _normalize_todo_status(status),
                    "priority": _normalize_todo_priority(priority),
                    "assignee": str(assignee or "").strip(),
                    "dependencies": _normalize_todo_dependencies(
                        dependencies,
                        self_id=normalized_id,
                    ),
                    "plan_id": str(plan_id or "").strip(),
                    "step_ref": str(step_ref or "").strip(),
                    "created_at": now,
                    "updated_at": now,
                    "history": [],
                }
                records.append(record)
                index = len(records) - 1
            else:
                record = records[index]

            if normalized_action == "upsert":
                if str(title or "").strip():
                    record["title"] = str(title).strip()
                if str(details or "").strip():
                    record["details"] = str(details).strip()
                if str(plan_id or "").strip():
                    record["plan_id"] = str(plan_id).strip()
                if str(step_ref or "").strip():
                    record["step_ref"] = str(step_ref).strip()
                record["status"] = _normalize_todo_status(status)
                record["priority"] = _normalize_todo_priority(priority)
                if str(assignee or "").strip():
                    record["assignee"] = str(assignee).strip()
                if dependencies is not None:
                    record["dependencies"] = _normalize_todo_dependencies(
                        dependencies,
                        self_id=str(record.get("id") or ""),
                    )
            elif normalized_action == "update_status":
                record["status"] = _normalize_todo_status(status)
                if str(step_ref or "").strip():
                    record["step_ref"] = str(step_ref).strip()

            history = record.get("history")
            if not isinstance(history, list):
                history = []
                record["history"] = history

            note_text = str(note or "").strip()
            if normalized_action == "append_note" and not note_text:
                return "note is required when action=append_note."
            if note_text or normalized_action in {"update_status", "complete"}:
                history.append(
                    {
                        "ts": now,
                        "action": normalized_action,
                        "note": note_text,
                        "status": record.get("status", "todo"),
                    }
                )

            record["updated_at"] = now
            records[index] = record
            try:
                _save_todo_store(resolved_path, records)
            except Exception as exc:
                return f"Failed to persist todo store: {exc}"

            return (
                f"Todo saved: id={record.get('id')} status={record.get('status')} "
                f"priority={record.get('priority')} file={_display_workspace_path(resolved_path)}"
            )

        tools.append(write_todo)

    if "edit_todo" in enabled_tool_names:
        @tool(
            "edit_todo",
            description="Edit existing todo items by id with status/priority/notes updates.",
            args_schema=EditTodoInput,
        )
        def edit_todo(
            todo_id: str,
            title: str = "",
            details: str = "",
            status: str = "",
            priority: str = "",
            assignee: str = "",
            dependencies: list[str] | None = None,
            plan_id: str = "",
            step_ref: str = "",
            note: str = "",
            file_path: str = ".agent/todo.json",
        ) -> str:
            resolved_path, error_text = _resolve_workspace_path(file_path)
            if resolved_path is None:
                return str(error_text or "Invalid file_path.")
            records = _load_todo_store(resolved_path)
            index = _find_todo_index(records, todo_id)
            if index < 0:
                return "Todo id not found."
            record = records[index]
            now = _now_iso()

            if str(title or "").strip():
                record["title"] = str(title).strip()
            if str(details or "").strip():
                record["details"] = str(details).strip()
            if str(plan_id or "").strip():
                record["plan_id"] = str(plan_id).strip()
            if str(step_ref or "").strip():
                record["step_ref"] = str(step_ref).strip()
            if str(status or "").strip():
                record["status"] = _normalize_todo_status(status)
            if str(priority or "").strip():
                record["priority"] = _normalize_todo_priority(priority)
            if str(assignee or "").strip():
                record["assignee"] = str(assignee).strip()
            if dependencies is not None:
                record["dependencies"] = _normalize_todo_dependencies(
                    dependencies,
                    self_id=str(record.get("id") or ""),
                )

            history = record.get("history")
            if not isinstance(history, list):
                history = []
                record["history"] = history
            note_text = str(note or "").strip()
            if note_text:
                history.append(
                    {
                        "ts": now,
                        "action": "edit",
                        "note": note_text,
                        "status": record.get("status", "todo"),
                    }
                )
            record["updated_at"] = now
            records[index] = record
            try:
                _save_todo_store(resolved_path, records)
            except Exception as exc:
                return f"Failed to persist todo store: {exc}"
            return (
                f"Todo updated: id={record.get('id')} status={record.get('status')} "
                f"priority={record.get('priority')} file={_display_workspace_path(resolved_path)}"
            )

        tools.append(edit_todo)

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
