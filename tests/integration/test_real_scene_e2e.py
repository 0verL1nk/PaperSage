import os
from pathlib import Path

import pytest

from agent.adapters.document import extract_document_payload
from agent.adapters.rag import create_project_evidence_retriever
from agent.application.turn_engine import execute_turn_core
from agent.domain.orchestration import (
    OrchestratedTurn,
    PolicyDecision,
    TeamExecution,
    build_trace_event,
    create_trace_context,
)

PAPER_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "papers" / "rag_agentic_reasoning"


def _paper_id_from_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if "-" not in stem:
        return stem
    return stem.split("-", 1)[0].strip()


def _load_documents_from_fixture(max_chars_per_doc: int = 25000) -> list[dict[str, str]]:
    pdf_paths = sorted(PAPER_FIXTURE_DIR.glob("*.pdf"))
    if not pdf_paths:
        return []

    cache_dir = PAPER_FIXTURE_DIR / "_extracted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, str]] = []
    for pdf_path in pdf_paths:
        cache_path = cache_dir / f"{pdf_path.stem}.txt"
        if cache_path.exists():
            extracted = cache_path.read_text(encoding="utf-8", errors="replace")
        else:
            payload = extract_document_payload(str(pdf_path))
            if not isinstance(payload, dict) or payload.get("result") != 1:
                continue
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            extracted = text.strip()
            cache_path.write_text(extracted, encoding="utf-8")

        paper_id = _paper_id_from_name(pdf_path.name)
        enriched = (
            f"[paper_id] {paper_id}\n"
            f"[file_name] {pdf_path.name}\n\n"
            f"{extracted[:max_chars_per_doc]}"
        )
        docs.append(
            {
                "doc_uid": paper_id,
                "doc_name": pdf_path.name,
                "text": enriched,
            }
        )
    return docs


@pytest.fixture(scope="module")
def real_scene_documents() -> list[dict[str, str]]:
    enabled = os.getenv("RUN_REAL_SCENARIO_E2E", "0").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        pytest.skip("Real scenario E2E disabled. Set RUN_REAL_SCENARIO_E2E=1 to run.")
    docs = _load_documents_from_fixture()
    if len(docs) < 5:
        pytest.skip("Real paper fixtures not ready. Please download enough PDFs into tests/fixtures/papers/rag_agentic_reasoning.")
    return docs


@pytest.fixture(scope="module")
def project_retriever(real_scene_documents):
    return create_project_evidence_retriever(
        documents=real_scene_documents,
        project_uid="proj-real-scene-e2e",
    )


def test_real_scene_survey_retrieval_hits_multiple_papers(project_retriever) -> None:
    payload = project_retriever(
        "请比较 arxiv:2005.11401、arxiv:2310.11511、arxiv:2210.03629 在检索增强与工具调用上的差异"
    )
    evidences = payload.get("evidences") if isinstance(payload, dict) else None
    assert isinstance(evidences, list) and len(evidences) >= 3
    hit_doc_uids = {
        str(item.get("doc_uid") or "").strip()
        for item in evidences
        if isinstance(item, dict)
    }
    hit_doc_uids.discard("")
    assert len(hit_doc_uids) >= 3


def test_real_scene_turn_engine_replaces_evidence_placeholders(project_retriever) -> None:
    trace_ctx = create_trace_context(channel="integration.real_scene")

    def _fake_orchestrator(**kwargs):
        on_event = kwargs.get("on_event")
        if callable(on_event):
            for sender, receiver, performative, content in (
                ("user", "leader", "request", "需要综述结论"),
                ("policy_engine", "leader", "policy", "plan=True,team=True"),
                ("planner", "leader", "plan", "step1,step2"),
                ("leader", "researcher", "dispatch", "收集证据"),
                ("researcher", "leader", "member_output", "返回证据摘要"),
                ("leader", "user", "final", "结论 [文档证据]，对比 [证据]"),
            ):
                on_event(
                    build_trace_event(
                        context=trace_ctx,
                        sender=sender,
                        receiver=receiver,
                        performative=performative,
                        content=content,
                    )
                )
        return OrchestratedTurn(
            answer="结论 [文档证据]，对比 [证据]",
            policy_decision=PolicyDecision(
                plan_enabled=True,
                team_enabled=True,
                reason="integration test",
                confidence=0.9,
                source="heuristic",
            ),
            team_execution=TeamExecution(
                enabled=True,
                roles=["researcher"],
                member_count=1,
                rounds=1,
                summary="ok",
            ),
            trace_payload=[],
            leader_tool_names=["search_document"],
        )

    result = execute_turn_core(
        prompt="请比较 RAG 与 Self-RAG，并给选型建议",
        hinted_prompt="请比较 RAG 与 Self-RAG，并给选型建议",
        leader_agent=object(),
        leader_runtime_config={},
        search_document_evidence_fn=project_retriever,
        orchestrated_turn_executor=_fake_orchestrator,
    )

    assert result["used_document_rag"] is True
    assert isinstance(result["evidence_items"], list) and result["evidence_items"]
    assert "[文档证据]" not in result["answer"]
    assert "[证据]" not in result["answer"]
    assert "[chunk_" in result["answer"] or ":chunk_" in result["answer"]
    hit_doc_uids = {
        str(item.get("doc_uid") or "").strip()
        for item in result["evidence_items"]
        if isinstance(item, dict)
    }
    hit_doc_uids.discard("")
    assert len(hit_doc_uids) >= 2


def test_real_scene_project_scope_isolation(real_scene_documents, project_retriever) -> None:
    full_retriever = project_retriever
    subset_docs = [
        doc
        for doc in real_scene_documents
        if doc["doc_uid"] in {"2005.11401", "2210.03629"}
    ]
    subset_retriever = create_project_evidence_retriever(
        documents=subset_docs,
        project_uid="proj-subset",
    )

    query = "比较 2310.11511 与 2210.03629 的差异"
    full_payload = full_retriever(query)
    subset_payload = subset_retriever(query)

    full_evidences = full_payload.get("evidences") if isinstance(full_payload, dict) else []
    subset_evidences = subset_payload.get("evidences") if isinstance(subset_payload, dict) else []
    assert isinstance(full_evidences, list) and full_evidences
    assert isinstance(subset_evidences, list) and subset_evidences

    full_doc_uids = {
        str(item.get("doc_uid") or "").strip()
        for item in full_evidences
        if isinstance(item, dict)
    }
    subset_doc_uids = {
        str(item.get("doc_uid") or "").strip()
        for item in subset_evidences
        if isinstance(item, dict)
    }
    full_doc_uids.discard("")
    subset_doc_uids.discard("")

    assert "2310.11511" in full_doc_uids
    assert "2310.11511" not in subset_doc_uids
    assert subset_doc_uids.issubset({"2005.11401", "2210.03629"})
