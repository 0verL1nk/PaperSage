from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from .capabilities import build_agent_tools


PAPER_QA_SYSTEM_PROMPT = """你是专业论文问答 Agent。
优先调用 search_document 检索文档证据，再组织答案。
仅当文档证据不足时，再调用 search_web 获取补充信息。
当用户要求总结、批判性阅读、方法比较、翻译等能力时，可调用 use_skill。
当用户要求生成思维导图时，输出严格 JSON（仅 JSON，无额外解释），格式为 $JSON_SCHEMA$。
答案要简洁、结构化，并标注证据来源（文档或网络）。
输出语言默认跟随用户输入语言。
禁止输出你的思考过程、规划过程、工具调用过程。
只输出面向用户的最终答案。
当前对话的文档：{document_name}"""


def _build_system_prompt(document_name: str | None = None) -> str:
    """构建带有文档名称的 system prompt"""
    doc_name = document_name if document_name else "未知文档"
    return PAPER_QA_SYSTEM_PROMPT.format(document_name=doc_name)


@dataclass(frozen=True)
class PaperAgentSession:
    agent: Any
    thread_id: str

    @property
    def runtime_config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}


def create_paper_agent_session(
    *,
    llm: Any,
    search_document_fn,
    read_document_fn=None,
    system_prompt: str = PAPER_QA_SYSTEM_PROMPT,
    document_name: str | None = None,
) -> PaperAgentSession:
    # 构建带有文档名称的 system prompt
    final_system_prompt = _build_system_prompt(document_name)

    tools = build_agent_tools(
        search_document_fn=search_document_fn,
        read_document_fn=read_document_fn,
    )
    thread_id = f"paper-qa-{uuid4().hex}"
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=final_system_prompt,
        checkpointer=InMemorySaver(),
    )
    return PaperAgentSession(agent=agent, thread_id=thread_id)
