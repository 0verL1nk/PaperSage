from agent.rag.evidence import EvidenceItem, EvidencePayload


def test_evidence_item_schema_fields():
    item = EvidenceItem(
        project_uid="project-1",
        doc_uid="doc-1",
        doc_name="paper-a.pdf",
        chunk_id="chunk_2",
        text="example evidence",
        score=0.87,
        page_no=3,
        offset_start=100,
        offset_end=140,
    )

    assert item.project_uid == "project-1"
    assert item.doc_uid == "doc-1"
    assert item.doc_name == "paper-a.pdf"
    assert item.chunk_id == "chunk_2"
    assert item.score == 0.87
    assert item.page_no == 3
    assert item.offset_start == 100
    assert item.offset_end == 140


def test_evidence_payload_defaults():
    payload = EvidencePayload()
    assert payload.evidences == []
    assert payload.trace == {}
