import logging
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate

from .paper_prompt import PAPER_QA_SYSTEM_PROMPT, build_paper_system_prompt
from .profiles import paper_leader_profile
from .session_factory import (
    AgentDependencies,
    AgentRuntimeOptions,
    create_agent_session,
)
from .session_factory import (
    AgentSession as PaperAgentSession,
)

logger = logging.getLogger(__name__)


DocumentAccessMode = Literal["scoped", "none"]

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
    """构建带上下文的 system prompt。"""
    access_mode = _normalize_document_access(document_access)
    proj_name = project_name if project_name else "默认项目"
    scope_text = scope_summary if scope_summary else "默认范围"
    if access_mode == "none":
        prompt = ChatPromptTemplate.from_template(EXTERNAL_ONLY_SYSTEM_PROMPT)
        return prompt.format(project_name=proj_name, scope_summary=scope_text)

    prompt = build_paper_system_prompt(
        document_name=document_name,
        project_name=project_name,
        scope_summary=scope_summary,
    )
    if scope_summary and any(keyword in scope_summary for keyword in ("项目内", "仅限", "仅基于项目文档", "不要联网")):
        prompt += (
            "\n\n[项目范围约束]\n"
            "- 当前项目文档范围内回答\n"
            "- 不要调用 search_papers 或 search_web\n"
        )
    return prompt


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
    access_mode = _normalize_document_access(document_access)
    logger.info(
        "Creating paper agent session: project_name=%s document_name=%s document_access=%s",
        project_name or "默认项目",
        document_name or "未知文档",
        access_mode,
    )
    final_system_prompt = (
        system_prompt
        if system_prompt != PAPER_QA_SYSTEM_PROMPT
        else _build_system_prompt(
            document_name=document_name,
            project_name=project_name,
            scope_summary=scope_summary,
            document_access=access_mode,
        )
    )
    runtime_options = AgentRuntimeOptions(
        llm=llm,
        document_name=document_name,
        project_name=project_name,
        scope_summary=scope_summary,
        system_prompt=final_system_prompt,
    )
    dependencies = AgentDependencies(
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        read_document_fn=read_document_fn,
        list_documents_fn=list_documents_fn,
        project_uid=project_uid,
        session_uid=session_uid,
        user_uuid=user_uuid,
    )
    session = create_agent_session(
        profile=paper_leader_profile,
        deps=dependencies,
        options=runtime_options,
    )
    return PaperAgentSession(
        agent=session.agent,
        thread_id=session.thread_id,
        tool_specs=session.tool_specs,
        profile_name=session.profile_name,
    )
