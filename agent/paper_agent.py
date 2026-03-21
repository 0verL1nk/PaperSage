import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver

from .runtime_agent import build_runtime_tools, create_runtime_agent

logger = logging.getLogger(__name__)


DocumentAccessMode = Literal["scoped", "none"]

PAPER_QA_SYSTEM_PROMPT = """你是专业论文问答 Agent。

[核心目标]
提供准确、有证据支撑的答案。每个结论都必须有文档证据支持。

[基本原则]
- 对于日常寒暄（如"你好"、"谢谢"），直接回答即可
- 对于任何需要查询文档的问题，必须使用 search_document 工具
- 必须使用 <evidence> 标签引用所有证据

[检索策略 - 重要]
1) 使用 search_document 多轮检索，直到获得充分证据：
   - 第一次检索：获取初步相关内容
   - 根据初步结果，调整关键词继续检索
   - 重复检索直到找到足够的证据支持结论
2) 若文档证据不足，再调用 search_papers
3) 仍不足时才调用 search_web
4) 如果用户明确要求“在当前项目文档范围内”“仅基于项目文档”“不要联网”或语义等价约束：
   - 只能使用 search_document
   - 不要调用 search_papers 或 search_web
   - 若项目文档证据不足，要明确说明当前项目范围内证据不足，而不是擅自扩展到外部来源
5) 一旦已经拿到足够的项目文档证据，就直接回答，不要继续额外调用外部检索工具

[证据引用 - 必须遵守]
回答中的每个关键结论都必须用 <evidence> 标签引用证据：
- 格式：<evidence>chunk_id|p页码|o起止偏移</evidence>
- 从 search_document 返回的 JSON 中提取：chunk_id、page_no、offset_start、offset_end
- 示例：根据研究<evidence>doc123|p5|o100-200</evidence>，该方法有效<evidence>doc456|p8|o300-400</evidence>
- 禁止使用 [文档证据]、[证据] 等占位符
- 禁止使用空的 <evidence/> 标签

[其他工具]
- 需要总结/批判性阅读/方法比较/翻译时，可调用 use_skill
- 生成思维导图时：调用 use_skill("mindmap", task)，然后直接输出 <mindmap>{{"name":"主题","children":[...]}}</mindmap>
- 生成思维导图时严禁输出 Mermaid、Markdown 代码块、标题、解释文字或任何 <mindmap> 标签外的额外文本

[复杂任务处理]
仅当遇到明确的复杂多步骤任务时才使用计划工具（如文献综述、对比分析、系统性调研等）：
1) 调用 create_plan 工具创建执行计划
2) 使用 write_todos 工具跟踪任务进度
3) 完成后调用 delete_plan 工具清理计划

不要对以下情况使用计划工具：
- 简单问答（如"你好"、"这是什么"）
- 单一查询任务
- 日常对话

[输出要求]
1) 输出语言默认跟随用户输入语言
2) 每个结论都必须有证据支持
3) 避免无依据推断

当前对话项目：{project_name}
当前检索范围：{scope_summary}
当前对话文档（兼容字段）：{document_name}"""


EXTERNAL_ONLY_SYSTEM_PROMPT = """你是专业论文问答 Agent。

[核心目标]
基于当前会话允许的外部检索能力，给出准确、可核对的答案。

[基本原则]
- 对于日常寒暄（如"你好"、"谢谢"），直接回答即可
- 当前会话不提供项目文档，不要调用 search_document、read_document、list_document
- 可以根据问题需要使用 search_papers 或 search_web
- 不要把外部资料伪装成“当前项目文档”或“项目内证据”

[检索策略 - 重要]
1) 优先根据用户问题选择 search_web 或 search_papers
2) 如果用户要求最新进展、近年变化、当前实践，优先使用 search_web
3) 如果外部公开资料仍不足，要明确说明证据不足，而不是伪造项目文档结论

[其他工具]
- 需要总结/批判性阅读/方法比较/翻译时，可调用 use_skill
- 生成思维导图时：调用 use_skill("mindmap", task)，然后直接输出 <mindmap>{{"name":"主题","children":[...]}}</mindmap>
- 生成思维导图时严禁输出 Mermaid、Markdown 代码块、标题、解释文字或任何 <mindmap> 标签外的额外文本

[复杂任务处理]
仅当遇到明确的复杂多步骤任务时才使用计划工具：
1) 调用 create_plan 工具创建执行计划
2) 使用 write_todos 工具跟踪任务进度
3) 完成后调用 delete_plan 工具清理计划

[输出要求]
1) 输出语言默认跟随用户输入语言
2) 明确区分外部公开资料与项目内材料
3) 避免无依据推断

当前对话项目：{project_name}
当前检索范围：{scope_summary}"""


def _normalize_document_access(document_access: str | None) -> DocumentAccessMode:
    normalized = str(document_access or "scoped").strip().lower()
    return "none" if normalized == "none" else "scoped"


def _build_system_prompt(
    document_name: str | None = None,
    project_name: str | None = None,
    scope_summary: str | None = None,
    document_access: str | None = None,
) -> str:
    """构建带上下文的 system prompt"""
    access_mode = _normalize_document_access(document_access)
    proj_name = project_name if project_name else "默认项目"
    scope_text = scope_summary if scope_summary else "默认范围"
    if access_mode == "none":
        prompt = ChatPromptTemplate.from_template(EXTERNAL_ONLY_SYSTEM_PROMPT)
        return prompt.format(project_name=proj_name, scope_summary=scope_text)

    doc_name = document_name if document_name else "未知文档"
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
    search_document_fn=None,
    search_document_evidence_fn=None,
    read_document_fn=None,
    list_documents_fn=None,
    system_prompt: str = PAPER_QA_SYSTEM_PROMPT,
    document_name: str | None = None,
    project_name: str | None = None,
    scope_summary: str | None = None,
    document_access: str | None = None,
    project_uid: str | None = None,
    session_uid: str | None = None,
    user_uuid: str | None = None,
) -> PaperAgentSession:
    del system_prompt
    access_mode = _normalize_document_access(document_access)
    logger.info(
        "Creating paper agent session: project_name=%s document_name=%s document_access=%s",
        project_name or "默认项目",
        document_name or "未知文档",
        access_mode,
    )
    final_system_prompt = _build_system_prompt(
        document_name=document_name,
        project_name=project_name,
        scope_summary=scope_summary,
        document_access=access_mode,
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

    enable_tool_selector = os.getenv("ENABLE_TOOL_SELECTOR", "false").lower() in {"true", "1", "yes"}

    agent = create_runtime_agent(
        model=llm,
        tools=tools,
        system_prompt=final_system_prompt,
        checkpointer=checkpointer,
        enable_tool_selector=enable_tool_selector,
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
