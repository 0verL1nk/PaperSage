import json
from pathlib import Path
from types import SimpleNamespace

from agent.application.evals import (
    AgentEvalCase,
    ExecuteTurnEvalRunner,
    FinalAnswerJudgeResult,
    build_eval_report,
    evaluate_case_result,
    load_eval_cases,
    run_agent_evals,
    select_eval_cases,
)


def test_load_eval_cases_supports_stable_process_contracts(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "fact_001",
                        "category": "fact",
                        "prompt": "论文结论是什么？",
                        "expected_answer_all_of": ["核心结论"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "multi_001",
                        "category": "multi_step",
                        "prompt": "请比较方法并给出建议",
                        "success_rubric": "Answer should compare methods and provide a recommendation.",
                        "requires_evidence": True,
                        "min_evidence_count": 2,
                        "require_plan": True,
                        "min_execution_completion_ratio": 1.0,
                        "required_tool_names": ["search_document", "search_web"],
                        "required_phase_labels": ["规划", "输出最终答案"],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(fixture)

    assert [item.case_id for item in cases] == ["fact_001", "multi_001"]
    assert cases[1].process_contract.requires_evidence is True
    assert cases[1].process_contract.min_evidence_count == 2
    assert cases[1].process_contract.required_tool_names == ["search_document", "search_web"]
    assert cases[1].process_contract.required_phase_labels == ["规划", "输出最终答案"]


def test_load_eval_cases_rejects_rows_without_success_contract(tmp_path: Path) -> None:
    fixture = tmp_path / "invalid.jsonl"
    fixture.write_text(
        json.dumps({"id": "broken", "category": "fact", "prompt": "x"}, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        load_eval_cases(fixture)
    except ValueError as exc:
        assert "success contract" in str(exc)
    else:
        raise AssertionError("Expected fixture validation failure")


def test_evaluate_case_result_combines_reference_and_process_checks() -> None:
    case = AgentEvalCase.from_dict(
        {
            "id": "compare_001",
            "category": "compare",
            "prompt": "请比较两种方法",
            "expected_answer_all_of": ["方法A", "方法B", "建议"],
            "requires_evidence": True,
            "min_evidence_count": 2,
            "require_plan": True,
            "min_execution_completion_ratio": 1.0,
            "required_tool_names": ["search_document", "search_web"],
            "required_phase_labels": ["规划", "输出最终答案"],
        }
    )

    turn_result = {
        "answer": "方法A 更快，方法B 更稳健，建议先选方法B。",
        "evidence_items": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        "plan": {"goal": "compare"},
        "runtime_state": {
            "current_plan": {"steps": [{"id": "s1"}, {"id": "s2"}]},
            "completed_step_ids": ["s1", "s2"],
        },
        "todos": [],
        "phase_path": "接收请求 -> 规划 -> 输出最终答案",
        "trace_payload": [
            {"performative": "complexity_analysis"},
            {"performative": "final"},
        ],
        "output_messages": [
            SimpleNamespace(tool_calls=[{"name": "search_document"}, {"name": "search_web"}])
        ],
        "run_latency_ms": 12.5,
        "used_document_rag": True,
        "leader_tool_names": ["search_document"],
    }

    result = evaluate_case_result(case, turn_result, judge=None)

    assert result["completed"] is True
    assert result["final_success"] is True
    assert result["process_success"] is True
    assert result["execution_completion_ratio"] == 1.0
    assert result["evidence_coverage"]["passed"] is True
    assert result["process_checks"]["tool_names_passed"] is True
    assert result["feedback"]["remediation_area"] == []


def test_evaluate_case_result_uses_rubric_judge_when_present() -> None:
    case = AgentEvalCase.from_dict(
        {
            "id": "rubric_001",
            "category": "multi_step",
            "prompt": "请完成任务",
            "success_rubric": "Answer must clearly finish the task.",
        }
    )

    turn_result = {
        "answer": "任务已完成",
        "evidence_items": [],
        "phase_path": "输出最终答案",
        "trace_payload": [{"performative": "internal_private_event"}],
        "run_latency_ms": 9.0,
    }

    def _judge(eval_case: AgentEvalCase, normalized_result: dict[str, object]) -> FinalAnswerJudgeResult:
        assert eval_case.case_id == "rubric_001"
        assert normalized_result["answer"] == "任务已完成"
        return FinalAnswerJudgeResult(passed=True, score=0.9, reasoning="rubric pass")

    result = evaluate_case_result(case, turn_result, judge=_judge)

    assert result["completed"] is True
    assert result["final_success"] is True
    assert result["process_success"] is True
    assert result["diagnostics"]["trace_diagnostic_count"] == 1


def test_evaluate_case_result_adds_actionable_feedback_for_failures() -> None:
    case = AgentEvalCase.from_dict(
        {
            "id": "hybrid_001",
            "category": "hybrid_research",
            "prompt": "请结合项目文档和联网检索给出建议",
            "expected_answer_all_of": ["项目文档", "最新进展", "建议"],
            "requires_evidence": True,
            "min_evidence_count": 2,
            "require_plan": True,
            "required_tool_names": ["search_document", "search_web"],
        }
    )

    turn_result = {
        "answer": "建议继续研究。",
        "evidence_items": [],
        "phase_path": "输出最终答案",
        "trace_payload": [{"performative": "internal_private_event"}],
        "output_messages": [SimpleNamespace(tool_calls=[{"name": "search_document"}])],
        "run_latency_ms": 15.0,
    }

    result = evaluate_case_result(case, turn_result, judge=None)

    assert result["completed"] is False
    assert "prompt" in result["feedback"]["remediation_area"]
    assert "retrieval/tooling" in result["feedback"]["remediation_area"]
    assert "architecture" not in result["feedback"]["remediation_area"]
    assert result["feedback"]["recommended_actions"]


def test_run_agent_evals_executes_cases_through_turn_engine() -> None:
    class _FakeAgent:
        def invoke(self, payload, config=None):
            assert payload["messages"][0]["role"] == "user"
            return {
                "messages": [
                    SimpleNamespace(
                        content="核心结论 <evidence>chunk-1|p1|o0-10</evidence>",
                        tool_calls=[{"name": "search_document", "args": {"query": "q"}}],
                    )
                ],
                "plan": {"goal": "answer"},
                "runtime_state": {
                    "current_plan": {"steps": [{"id": "s1"}]},
                    "completed_step_ids": ["s1"],
                },
            }

    case = AgentEvalCase.from_dict(
        {
            "id": "fact_001",
            "category": "fact",
            "prompt": "论文结论是什么？",
            "expected_answer_all_of": ["核心结论"],
            "requires_evidence": True,
            "min_evidence_count": 1,
            "require_plan": True,
            "min_execution_completion_ratio": 1.0,
        }
    )

    runner = ExecuteTurnEvalRunner(
        leader_agent=_FakeAgent(),
        leader_runtime_config={},
        search_document_evidence_fn=lambda _query: {
            "evidences": [{"chunk_id": "chunk-1", "text": "证据", "page_no": 1}]
        },
    )

    report = run_agent_evals([case], runner=runner, judge=None)

    assert report["total_cases"] == 1
    assert report["completed_cases"] == 1
    assert report["completion_rate"] == 1.0
    assert report["cases"][0]["completed"] is True
    assert report["cases"][0]["evidence_coverage"]["count"] == 1


def test_build_eval_report_summarizes_case_results() -> None:
    report = build_eval_report(
        fixture_path="tests/evals/fixtures/agent_task_eval_set_v1.jsonl",
        case_results=[
            {
                "case_id": "a",
                "completed": True,
                "final_success": True,
                "process_success": True,
                "execution_completion_ratio": 1.0,
                "evidence_coverage": {"passed": True, "count": 2},
                "feedback": {"remediation_area": []},
            },
            {
                "case_id": "b",
                "completed": False,
                "final_success": True,
                "process_success": False,
                "execution_completion_ratio": 0.5,
                "evidence_coverage": {"passed": False, "count": 0},
                "feedback": {"remediation_area": ["retrieval/tooling", "prompt"]},
            },
        ],
    )

    assert report["total_cases"] == 2
    assert report["completed_cases"] == 1
    assert report["completion_rate"] == 0.5
    assert report["final_success_rate"] == 1.0
    assert report["process_success_rate"] == 0.5
    assert report["evidence_coverage_rate"] == 0.5
    assert report["average_execution_completion_ratio"] == 0.75
    assert report["failed_case_ids"] == ["b"]
    assert report["remediation_area_counts"] == {"prompt": 1, "retrieval/tooling": 1}


def test_select_eval_cases_supports_case_ids_and_limit() -> None:
    cases = [
        AgentEvalCase.from_dict(
            {
                "id": "a",
                "category": "fact",
                "prompt": "A",
                "expected_answer_all_of": ["A"],
            }
        ),
        AgentEvalCase.from_dict(
            {
                "id": "b",
                "category": "fact",
                "prompt": "B",
                "expected_answer_all_of": ["B"],
            }
        ),
        AgentEvalCase.from_dict(
            {
                "id": "c",
                "category": "fact",
                "prompt": "C",
                "expected_answer_all_of": ["C"],
            }
        ),
    ]

    filtered = select_eval_cases(cases, case_ids=["c", "a"])
    assert [item.case_id for item in filtered] == ["a", "c"]

    limited = select_eval_cases(cases, limit=2)
    assert [item.case_id for item in limited] == ["a", "b"]
