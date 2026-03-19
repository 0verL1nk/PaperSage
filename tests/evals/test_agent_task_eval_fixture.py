from agent.application.evals import load_eval_cases

FIXTURE_PATH = "tests/evals/fixtures/agent_task_eval_set_v1.jsonl"


def test_agent_task_eval_fixture_has_project_and_web_cases() -> None:
    cases = load_eval_cases(FIXTURE_PATH)

    assert len(cases) >= 4
    categories = {item.category for item in cases}
    assert {"project_rag", "project_compare", "web_research", "hybrid_research"}.issubset(categories)


def test_agent_task_eval_fixture_uses_stable_process_contracts() -> None:
    cases = load_eval_cases(FIXTURE_PATH)
    hybrid_case = next(item for item in cases if item.case_id == "hybrid_research_001")

    assert hybrid_case.process_contract.requires_evidence is True
    assert hybrid_case.process_contract.required_tool_names == ["search_document", "search_web"]
    assert hybrid_case.process_contract.required_phase_labels == ["规划", "输出最终答案"]
