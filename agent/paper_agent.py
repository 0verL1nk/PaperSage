import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver

from .runtime_agent import build_runtime_tools, create_runtime_agent

logger = logging.getLogger(__name__)


PAPER_QA_SYSTEM_PROMPT = """你是专业论文问答 Agent。

[核心目标]
在准确、可验证、简洁之间做平衡，输出可执行且有证据支撑的答案。

[工具使用策略]
1) 优先调用 search_document 获取当前项目文档证据（结构化 JSON）。
2) 若文档证据不足，再调用 search_papers；仍不足时才调用 search_web。
3) 需要总结/批判性阅读/方法比较/翻译/思维导图时，可调用 use_skill。
4) 生成思维导图时，先调用 use_skill("mindmap", task)，输出必须用 <mindmap> 标签包裹：
   <mindmap>{{"name":"主题","children":[...]}}</mindmap>
   禁止使用 markdown 代码块 ```json
5) 若任务需要显式拆步骤，调用 create_plan 请求规划运行时。
6) 若任务需要并行分析、交叉验证或多角色协作，调用 start_team(goal, reason, roles_hint) 请求团队运行时。

[答案约束]
1) 优先使用项目文档证据，避免无依据推断。
2) 结论尽量附上证据来源标注（文档/论文库/网络）。
   若引用文档证据，使用具体引用格式如 [chunk_id|p页码|o起止偏移]，禁止使用 [文档证据] 这类占位符。
3) 输出语言默认跟随用户输入语言。

当前对话项目：{project_name}
当前检索范围：{scope_summary}
当前对话文档（兼容字段）：{document_name}"""


def _build_system_prompt(
    document_name: str | None = None,
    project_name: str | None = None,
    scope_summary: str | None = None,
) -> str:
    """构建带上下文的 system prompt"""
    doc_name = document_name if document_name else "未知文档"
    proj_name = project_name if project_name else "默认项目"
    scope_text = scope_summary if scope_summary else "默认范围"
    prompt = ChatPromptTemplate.from_template(PAPER_QA_SYSTEM_PROMPT)
    return prompt.format(
        document_name=doc_name,
        project_name=proj_name,
        scope_summary=scope_text,
    )


@dataclass(frozen=True)
class PaperAgentSession:
    agent: Any
    thread_id: str
    tool_specs: list[dict[str, str]]

    @property
    def runtime_config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}


def create_paper_agent_session(
    *,
    llm: Any,
    search_document_fn,
    search_document_evidence_fn=None,
    read_document_fn=None,
    list_documents_fn=None,
    system_prompt: str = PAPER_QA_SYSTEM_PROMPT,
    document_name: str | None = None,
    project_name: str | None = None,
    scope_summary: str | None = None,
    project_uid: str | None = None,
    session_uid: str | None = None,
    user_uuid: str | None = None,
) -> PaperAgentSession:
    logger.info(
        "Creating paper agent session: project_name=%s document_name=%s",
        project_name or "默认项目",
        document_name or "未知文档",
    )
    final_system_prompt = _build_system_prompt(
        document_name=document_name,
        project_name=project_name,
        scope_summary=scope_summary,
    )

    tools = build_runtime_tools(
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        read_document_fn=read_document_fn,
        list_documents_fn=list_documents_fn,
    )

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

    thread_id = f"paper-qa-{uuid4().hex}"
    if project_uid and session_uid and user_uuid:
        from .adapters import get_or_create_thread_id_for_session
        thread_id = get_or_create_thread_id_for_session(
            project_uid=project_uid,
            session_uid=session_uid,
            uuid=user_uuid,
        )

    checkpointer_db_path = os.getenv("CHECKPOINTER_DB_PATH", "./data/checkpoints.db")
    os.makedirs(os.path.dirname(checkpointer_db_path), exist_ok=True)
    conn = sqlite3.connect(checkpointer_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    agent = create_runtime_agent(
        model=llm,
        tools=tools,
        system_prompt=final_system_prompt,
        checkpointer=checkpointer,
    )
    logger.info("Created paper agent session: thread_id=%s tools=%s", thread_id, len(tools))
    return PaperAgentSession(agent=agent, thread_id=thread_id, tool_specs=tool_specs)


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
