from pathlib import Path

from agent.archive import list_agent_outputs, save_agent_output


def test_save_and_list_agent_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"

    save_agent_output(
        uuid="u1",
        doc_uid="d1",
        doc_name="paper.pdf",
        output_type="summary",
        content="summary text",
        db_name=str(db_path),
    )
    save_agent_output(
        uuid="u1",
        doc_uid="d1",
        doc_name="paper.pdf",
        output_type="mindmap",
        content='{"name":"root","children":[]}',
        db_name=str(db_path),
    )

    items = list_agent_outputs(uuid="u1", doc_uid="d1", db_name=str(db_path))

    assert len(items) == 2
    assert items[0]["output_type"] == "mindmap"
    assert items[1]["output_type"] == "summary"


def test_list_agent_outputs_filters_by_uuid(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"

    save_agent_output(
        uuid="u1",
        output_type="qa",
        content="a1",
        db_name=str(db_path),
    )
    save_agent_output(
        uuid="u2",
        output_type="qa",
        content="a2",
        db_name=str(db_path),
    )

    items = list_agent_outputs(uuid="u1", db_name=str(db_path))
    assert len(items) == 1
    assert items[0]["content"] == "a1"
