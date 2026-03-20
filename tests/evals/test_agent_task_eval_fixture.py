from agent.application.evals import load_eval_cases

FIXTURE_PATH = "tests/evals/fixtures/agent_task_eval_set_v1.jsonl"


def test_agent_task_eval_fixture_has_broad_project_and_web_case_coverage() -> None:
    cases = load_eval_cases(FIXTURE_PATH)

    assert len(cases) >= 12
    categories = {item.category for item in cases}
    assert {
        "project_rag",
        "project_compare",
        "project_scope_boundary",
        "project_gap",
        "web_research",
        "web_tradeoff",
        "hybrid_research",
        "hybrid_rollout",
        "hybrid_guardrail",
        "hybrid_reject",
    }.issubset(categories)


def test_agent_task_eval_fixture_uses_judge_rubrics_and_stable_process_contracts() -> None:
    cases = load_eval_cases(FIXTURE_PATH)
    hybrid_case = next(item for item in cases if item.case_id == "hybrid_research_001")
    boundary_case = next(item for item in cases if item.case_id == "project_scope_boundary_001")
    rollout_case = next(item for item in cases if item.case_id == "hybrid_rollout_001")

    assert hybrid_case.final_answer_contract.success_rubric
    assert hybrid_case.process_contract.requires_evidence is True
    assert hybrid_case.process_contract.required_tool_names == ["search_document", "search_web"]
    assert boundary_case.process_contract.required_tool_names == ["search_document"]
    assert rollout_case.process_contract.require_plan is True
    assert rollout_case.process_contract.min_execution_completion_ratio == 1.0



def test_default_task_eval_fixture_avoids_brittle_phase_label_contracts() -> None:
    cases = load_eval_cases(FIXTURE_PATH)

    assert all(item.process_contract.required_phase_labels == [] for item in cases)
