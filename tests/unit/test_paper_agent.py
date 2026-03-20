from agent import paper_agent as paper_agent_module


def test_build_system_prompt_does_not_raise():
    result = paper_agent_module._build_system_prompt(
        document_name="测试文档",
        project_name="测试项目",
        scope_summary="测试范围",
    )
    assert "你是通用智能 Agent" in result
    assert "你正在处理论文阅读与文档问答任务" in result
    assert "测试文档" in result
    assert "测试项目" in result
    assert "<mindmap>{" in result
    assert "严禁输出 Mermaid" in result


def test_build_system_prompt_explicitly_blocks_external_search_for_project_only_queries():
    result = paper_agent_module._build_system_prompt(
        document_name="测试文档",
        project_name="测试项目",
        scope_summary="仅限项目内 4 篇论文",
    )

    assert "当前项目文档范围内" in result
    assert "不要调用 search_papers 或 search_web" in result


def test_build_system_prompt_for_document_free_session_omits_project_document_instructions():
    result = paper_agent_module._build_system_prompt(
        project_name="测试项目",
        scope_summary="仅允许外部检索，不提供项目文档",
        document_access="none",
    )

    assert "当前会话不提供项目文档" in result
    assert "不要调用 search_document" in result
    assert "必须使用 search_document" not in result
    assert "当前对话文档（兼容字段）" not in result


def test_create_paper_agent_session_delegates_to_generic_factory(monkeypatch):
    captured = {}

    def fake_create_agent_session(*, profile, deps, options):
        captured["profile"] = profile
        captured["deps"] = deps
        captured["options"] = options
        return paper_agent_module.PaperAgentSession(
            agent="agent",
            thread_id="thread-1",
            tool_specs=[],
            profile_name=profile.name,
        )

    monkeypatch.setattr(paper_agent_module, "create_agent_session", fake_create_agent_session)

    session = paper_agent_module.create_paper_agent_session(
        llm="fake-llm",
        search_document_fn=lambda q: q,
        document_name="文档A",
        project_name="项目A",
        scope_summary="范围A",
    )

    assert session.thread_id == "thread-1"
    assert session.profile_name == "paper_leader"
    assert captured["profile"].name == "paper_leader"
    assert captured["deps"].search_document_fn("q") == "q"
    assert captured["options"].llm == "fake-llm"
    assert captured["options"].document_name == "文档A"
    assert "项目A" in captured["options"].system_prompt


def test_paper_agent_session_runtime_config_contains_thread_id():
    session = paper_agent_module.PaperAgentSession(
        agent=object(),
        thread_id="thread-1",
        tool_specs=[],
    )
    assert session.runtime_config == {"configurable": {"thread_id": "thread-1"}}
