from collections.abc import Callable

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field


SKILL_LIBRARY = {
    "summary": "Summarize long context into concise bullets and cite evidence.",
    "critical_reading": "Evaluate claims, assumptions, and evidence quality before final conclusions.",
    "method_compare": "Compare two methods across objective, setup, metrics, and trade-offs.",
    "translation": "Preserve technical terms and structure while translating academic text.",
}


class SearchDocumentInput(BaseModel):
    query: str = Field(
        description="Specific paper question or keyword used to retrieve evidence snippets."
    )


class ReadDocumentInput(BaseModel):
    offset: int = Field(
        default=0,
        description="Character offset to start reading from (0 means start from beginning).",
    )
    limit: int = Field(
        default=2000,
        description="Maximum number of characters to read (recommended 1000-3000 for context).",
    )
    include_rag: bool = Field(
        default=False,
        description="Whether to use RAG to retrieve relevant context around the reading position.",
    )


class SearchWebInput(BaseModel):
    query: str = Field(
        description="Supplemental public-web query used only when document evidence is insufficient."
    )


class SkillInput(BaseModel):
    skill_name: str = Field(
        description="Skill template name. Allowed values: summary, critical_reading, method_compare, translation."
    )
    task: str = Field(
        description="Current user task where the selected skill guidance should be applied."
    )


def build_agent_tools(
    search_document_fn: Callable[[str], str],
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None,
):
    """构建 Agent 工具集

    Args:
        search_document_fn: RAG 检索函数
        read_document_fn: 分块读取函数，接收 (offset, limit)，返回 (content, total_length)
    """
    @tool(
        "search_document",
        description="Search uploaded paper content for relevant evidence snippets using RAG.",
        args_schema=SearchDocumentInput,
    )
    def search_document(query: str) -> str:
        return search_document_fn(query)

    if read_document_fn is not None:
        @tool(
            "read_document",
            description="Read a specific portion of the uploaded paper with pagination. "
            "Use offset to skip to a position, limit to control chunk size. "
            "Set include_rag=True to get relevant context around the position.",
            args_schema=ReadDocumentInput,
        )
        def read_document(offset: int = 0, limit: int = 2000, include_rag: bool = False) -> str:
            content, total = read_document_fn(offset, limit)

            # 如果启用 RAG，获取相关上下文
            rag_context = ""
            if include_rag:
                # 基于位置获取相关段落
                # 计算当前位置的大致位置（用于检索）
                query = f"position_{offset}"
                rag_context = search_document_fn(query)

            result = f"=== 文档阅读 (字符位置 {offset} - {offset + limit}) ===\n"
            result += f"总长度: {total} 字符\n"
            result += f"当前 chunk: {len(content)} 字符\n\n"
            if rag_context:
                result += f"=== 相关上下文 (RAG) ===\n{rag_context}\n\n"
            result += f"=== 内容 ===\n{content}"
            return result
    else:
        read_document = None

    try:
        web_search_client = DuckDuckGoSearchRun(name="search_web")
    except Exception:
        web_search_client = None

    @tool(
        "search_web",
        description="Search public web content for supplemental context.",
        args_schema=SearchWebInput,
    )
    def search_web(query: str) -> str:
        if web_search_client is None:
            return "Web search is unavailable in current environment. Install ddgs to enable it."
        try:
            return web_search_client.run(query)
        except Exception as exc:
            return f"Web search failed: {exc}"

    @tool(
        "use_skill",
        description="Apply a named skill template to the current task and return operational guidance.",
        args_schema=SkillInput,
    )
    def use_skill(skill_name: str, task: str) -> str:
        normalized_name = skill_name.strip().lower()
        guidance = SKILL_LIBRARY.get(normalized_name)
        if not guidance:
            options = ", ".join(sorted(SKILL_LIBRARY.keys()))
            return f"Unknown skill '{skill_name}'. Available skills: {options}."
        return f"Skill: {normalized_name}\nGuidance: {guidance}\nTask: {task}"

    tools = [search_document, search_web, use_skill]
    if read_document is not None:
        tools.insert(1, read_document)  # 在 search_document 后插入 read_document

    return tools
