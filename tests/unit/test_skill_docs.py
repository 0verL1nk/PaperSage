from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_summary_skill_uses_canonical_evidence_format() -> None:
    content = (ROOT / "agent/skills/summary/SKILL.md").read_text(encoding="utf-8")
    assert "<evidence>chunk_id|p页码|o起止偏移</evidence>" in content
    assert "[Section 2.1]" not in content
    assert "[Page 3]" not in content
    assert "[citation]" not in content


def test_agentic_search_skill_documents_canonical_document_citations() -> None:
    expected = "<evidence>chunk_id|p页码|o起止偏移</evidence>"
    skill_content = (ROOT / "agent/skills/agentic_search/SKILL.md").read_text(encoding="utf-8")
    blueprint_content = (
        ROOT / "agent/skills/agentic_search/references/workflow_blueprint.md"
    ).read_text(encoding="utf-8")
    schema_content = (
        ROOT / "agent/skills/agentic_search/references/output_schema.md"
    ).read_text(encoding="utf-8")

    assert expected in skill_content
    assert expected in blueprint_content
    assert expected in schema_content
