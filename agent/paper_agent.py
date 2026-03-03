import logging
import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate

from .capabilities import build_agent_tools

logger = logging.getLogger(__name__)


PAPER_QA_SYSTEM_PROMPT = """你是专业论文问答 Agent。
优先调用 search_document 检索文档证据（该工具返回结构化 JSON 证据列表），再组织答案。
当文档证据不足时，先调用 search_papers 检索学术来源，再视情况调用 search_web 获取补充信息。
当用户要求总结、批判性阅读、方法比较、翻译、思维导图等能力时，可调用 use_skill。
当用户要求生成思维导图时，优先调用 use_skill("mindmap", task) 获取约束，再输出严格 JSON（仅 JSON，无额外解释），格式为 {{"name":"主题","children":[{{"name":"子主题","children":[...]}}]}}。
答案要简洁、结构化，并标注证据来源（文档或网络）。
输出语言默认跟随用户输入语言。
禁止输出你的思考过程、规划过程、工具调用过程。
只输出面向用户的最终答案。
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
    system_prompt: str = PAPER_QA_SYSTEM_PROMPT,
    document_name: str | None = None,
    project_name: str | None = None,
    scope_summary: str | None = None,
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

    tools = build_agent_tools(
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        read_document_fn=read_document_fn,
    )

    tool_specs: list[dict[str, str]] = []
    for tool_item in tools:
        name = str(getattr(tool_item, "name", "") or "").strip()
        description = str(getattr(tool_item, "description", "") or "").strip()
        args_schema = getattr(tool_item, "args_schema", None)
        args_schema_text = ""
        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            try:
                args_schema_text = json.dumps(args_schema.model_json_schema(), ensure_ascii=False)
            except Exception:
                args_schema_text = ""
        if not name and not description and not args_schema_text:
            continue
        tool_specs.append(
            {
                "name": name,
                "description": description,
                "args_schema": args_schema_text,
            }
        )

    thread_id = f"paper-qa-{uuid4().hex}"
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=final_system_prompt,
        checkpointer=InMemorySaver(),
    )
    logger.info("Created paper agent session: thread_id=%s tools=%s", thread_id, len(tools))
    return PaperAgentSession(agent=agent, thread_id=thread_id, tool_specs=tool_specs)
