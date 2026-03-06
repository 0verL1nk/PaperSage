from agent.application.agent_center.runtime_state import (
    clear_project_runtime,
    has_cached_agent_session,
    load_document_text,
)


class _Logger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


def test_has_cached_agent_session_true():
    session_state = {
        "paper_agent_sessions": {"leader:p1:s1": {}},
        "paper_evidence_retrievers": {"p1": object()},
        "paper_project_scope_signatures": {"p1": "sig"},
    }
    assert has_cached_agent_session(
        session_state=session_state,
        build_session_key_fn=lambda p, s, m: f"{m}:{p}:{s}",
        mode_leader="leader",
        project_uid="p1",
        session_uid="s1",
        scope_signature="sig",
    )


def test_clear_project_runtime_removes_project_items():
    session_state = {
        "paper_agent_sessions": {
            "leader:p1:s1": {},
            "leader:p2:s1": {},
        },
        "paper_evidence_retrievers": {"p1": 1, "p2": 2},
        "paper_project_llms": {"p1": 1, "p2": 2},
        "paper_project_search_document_fns": {"p1": 1, "p2": 2},
    }
    clear_project_runtime(
        session_state=session_state,
        project_uid="p1",
        mode_leader="leader",
    )
    assert "leader:p1:s1" not in session_state["paper_agent_sessions"]
    assert "leader:p2:s1" in session_state["paper_agent_sessions"]
    assert "p1" not in session_state["paper_evidence_retrievers"]


def test_load_document_text_session_hit():
    session_state = {"document_text_cache": {"u1": "cached"}}
    text, source, err = load_document_text(
        session_state=session_state,
        logger=_Logger(),
        selected_uid="u1",
        file_path="/tmp/a.pdf",
        load_cached_extraction_fn=lambda _uid: None,
        extract_document_fn=lambda _path: {"result": 1, "text": "x"},
        save_cached_extraction_fn=lambda **_kwargs: None,
    )
    assert text == "cached"
    assert source == "session_hit"
    assert err is None


def test_load_document_text_db_restore():
    session_state = {}
    text, source, err = load_document_text(
        session_state=session_state,
        logger=_Logger(),
        selected_uid="u1",
        file_path="/tmp/a.pdf",
        load_cached_extraction_fn=lambda _uid: "db-cache",
        extract_document_fn=lambda _path: {"result": 1, "text": "x"},
        save_cached_extraction_fn=lambda **_kwargs: None,
    )
    assert text == "db-cache"
    assert source == "db_restore"
    assert err is None


def test_load_document_text_extract_error():
    session_state = {}
    text, source, err = load_document_text(
        session_state=session_state,
        logger=_Logger(),
        selected_uid="u1",
        file_path="/tmp/a.pdf",
        load_cached_extraction_fn=lambda _uid: None,
        extract_document_fn=lambda _path: {"result": 0, "text": "boom"},
        save_cached_extraction_fn=lambda **_kwargs: None,
    )
    assert text is None
    assert source == "error"
    assert "文档加载失败" in str(err)


def test_load_document_text_extract_success_persists():
    session_state = {}
    saved = {"called": 0}

    def _save(**_kwargs):
        saved["called"] += 1

    text, source, err = load_document_text(
        session_state=session_state,
        logger=_Logger(),
        selected_uid="u1",
        file_path="/tmp/a.pdf",
        load_cached_extraction_fn=lambda _uid: None,
        extract_document_fn=lambda _path: {"result": 1, "text": "fresh"},
        save_cached_extraction_fn=_save,
    )
    assert text == "fresh"
    assert source == "extracted"
    assert err is None
    assert saved["called"] == 1
    assert session_state["document_text_cache"]["u1"] == "fresh"


def test_load_document_text_prunes_cache_when_over_limit(monkeypatch):
    monkeypatch.setenv("AGENT_DOCUMENT_TEXT_CACHE_MAX_CHARS", "10")
    session_state = {"document_text_cache": {"old": "12345678"}}
    text, source, err = load_document_text(
        session_state=session_state,
        logger=_Logger(),
        selected_uid="u2",
        file_path="/tmp/b.pdf",
        load_cached_extraction_fn=lambda _uid: "abcde",
        extract_document_fn=lambda _path: {"result": 1, "text": "x"},
        save_cached_extraction_fn=lambda **_kwargs: None,
    )
    assert text == "abcde"
    assert source == "db_restore"
    assert err is None
    assert "u2" in session_state["document_text_cache"]
    assert "old" not in session_state["document_text_cache"]
