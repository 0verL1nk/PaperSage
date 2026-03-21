from agent.paper_prompt import build_paper_system_prompt
from agent.prompts.base import build_base_agent_prompt
from agent.prompts.paper_domain import build_paper_domain_prompt


def test_base_prompt_is_generic_and_not_paper_specific():
    prompt = build_base_agent_prompt()

    assert "专业论文问答 Agent" not in prompt
    assert "search_document" not in prompt
    assert "<evidence>" not in prompt
    assert "输出语言默认跟随用户输入语言" in prompt


def test_paper_domain_prompt_carries_paper_specific_retrieval_rules():
    prompt = build_paper_domain_prompt(
        document_name="文档A",
        project_name="项目A",
        scope_summary="范围A",
    )

    assert "专业论文问答 Agent" not in prompt
    assert "search_document" in prompt
    assert "<evidence>" in prompt
    assert "不要再次检索" in prompt
    assert "should_stop" in prompt
    assert "sleep" in prompt
    assert "当前对话项目：项目A" in prompt


def test_paper_system_prompt_combines_generic_base_domain_and_leader_role():
    prompt = build_paper_system_prompt(
        document_name="文档A",
        project_name="项目A",
        scope_summary="范围A",
    )

    assert "你是通用智能 Agent" in prompt
    assert "search_document" in prompt
    assert "你负责调度与最终回答" in prompt
