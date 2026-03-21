import json
import logging
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from .capabilities import build_profile_tools
from .profiles import AgentProfile
from .runtime_agent import create_runtime_agent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDependencies:
    search_document_fn: Callable[[str], str]
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None
    list_documents_fn: Callable[[], list[dict[str, Any]]] | None = None
    project_uid: str | None = None
    session_uid: str | None = None
    user_uuid: str | None = None


@dataclass(frozen=True)
class AgentRuntimeOptions:
    llm: Any
    document_name: str | None = None
    project_name: str | None = None
    scope_summary: str | None = None
    system_prompt: str | None = None
    enable_tool_selector: bool | None = None
    thread_id: str | None = None


@dataclass(frozen=True)
class AgentSession:
    agent: Any
    thread_id: str
    tool_specs: list[dict[str, str]]
    profile_name: str = ""

    @property
    def runtime_config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}


def create_agent_session(
    *,
    profile: AgentProfile,
    deps: AgentDependencies,
    options: AgentRuntimeOptions,
) -> AgentSession:
    logger.info(
        "Creating agent session: profile=%s project_name=%s document_name=%s",
        profile.name,
        options.project_name or "默认项目",
        options.document_name or "未知文档",
    )
    final_system_prompt = options.system_prompt or profile.prompt_builder(
        document_name=options.document_name,
        project_name=options.project_name,
        scope_summary=options.scope_summary,
    )

    tools = build_profile_tools(profile, deps)
    tool_specs = _build_tool_specs(tools)
    thread_id = _resolve_thread_id(deps, explicit_thread_id=options.thread_id)
    checkpointer = _build_checkpointer()
    enable_tool_selector = _resolve_enable_tool_selector(options.enable_tool_selector)

    agent = create_runtime_agent(
        model=options.llm,
        tools=tools,
        system_prompt=final_system_prompt,
        checkpointer=checkpointer,
        enable_tool_selector=enable_tool_selector,
        profile=profile,
        deps=deps,
    )
    logger.info(
        "Created agent session: profile=%s thread_id=%s tools=%s",
        profile.name,
        thread_id,
        len(tools),
    )
    return AgentSession(
        agent=agent,
        thread_id=thread_id,
        tool_specs=tool_specs,
        profile_name=profile.name,
    )


def _resolve_thread_id(deps: AgentDependencies, *, explicit_thread_id: str | None = None) -> str:
    if explicit_thread_id:
        return explicit_thread_id

    thread_id = f"paper-qa-{uuid4().hex}"
    if deps.project_uid and deps.session_uid and deps.user_uuid:
        from .adapters import get_or_create_thread_id_for_session

        return get_or_create_thread_id_for_session(
            project_uid=deps.project_uid,
            session_uid=deps.session_uid,
            uuid=deps.user_uuid,
        )
    return thread_id


def _build_checkpointer() -> SqliteSaver:
    checkpointer_db_path = os.getenv("CHECKPOINTER_DB_PATH", "./data/checkpoints.db")
    checkpointer_dir = os.path.dirname(checkpointer_db_path)
    if checkpointer_dir:
        os.makedirs(checkpointer_dir, exist_ok=True)
    conn = sqlite3.connect(checkpointer_db_path, check_same_thread=False)
    return SqliteSaver(conn)


def _resolve_enable_tool_selector(explicit_value: bool | None) -> bool:
    if explicit_value is not None:
        return explicit_value
    return os.getenv("ENABLE_TOOL_SELECTOR", "false").lower() in {"true", "1", "yes"}


def _build_tool_specs(tools: list[Any]) -> list[dict[str, str]]:
    tool_specs: list[dict[str, str]] = []
    schema_level = _tool_schema_level()
    for tool_item in tools:
        name = str(getattr(tool_item, "name", "") or "").strip()
        description = str(getattr(tool_item, "description", "") or "").strip()
        args_schema_text = _serialize_tool_args_schema(tool_item, schema_level=schema_level)
        if not name and not description and not args_schema_text:
            continue
        tool_specs.append(
            {
                "name": name,
                "description": description,
                "args_schema": args_schema_text,
                "schema_level": schema_level,
            }
        )
    return tool_specs


def _tool_schema_level() -> str:
    configured = str(os.getenv("AGENT_TOOL_SCHEMA_LEVEL", "manifest") or "").strip().lower()
    if configured in {"manifest", "compact", "full"}:
        return configured
    return "manifest"


def _serialize_tool_args_schema(tool_item: Any, *, schema_level: str) -> str:
    if schema_level == "manifest":
        return ""
    args_schema = getattr(tool_item, "args_schema", None)
    if args_schema is None or not hasattr(args_schema, "model_json_schema"):
        return ""
    try:
        schema_obj = args_schema.model_json_schema()
    except Exception:
        return ""
    if not isinstance(schema_obj, dict):
        return ""
    if schema_level == "full":
        return json.dumps(schema_obj, ensure_ascii=False)
    properties = schema_obj.get("properties")
    field_names: list[str] = []
    if isinstance(properties, dict):
        field_names = sorted(str(key).strip() for key in properties.keys() if str(key).strip())
    required = schema_obj.get("required")
    required_names: list[str] = []
    if isinstance(required, list):
        required_names = sorted(str(item).strip() for item in required if str(item).strip())
    compact_schema = {
        "type": str(schema_obj.get("type") or "object"),
        "fields": field_names,
        "required": required_names,
    }
    return json.dumps(compact_schema, ensure_ascii=False)
