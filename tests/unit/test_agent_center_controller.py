from agent.application.agent_center.controller import (
    build_hinted_prompt,
    build_scope_cache_caption,
    load_scope_docs_with_text,
    resolve_archive_target,
    resolve_runtime_session_id,
    resolve_selected_doc_uid_for_logging,
    serialize_output_content,
    validate_runtime_prerequisites,
)


def test_validate_runtime_prerequisites():
    assert validate_runtime_prerequisites(api_key="", model_name="m") == "missing_api_key"
    assert validate_runtime_prerequisites(api_key="k", model_name="") == "missing_model_name"
    assert validate_runtime_prerequisites(api_key="k", model_name="m") is None


def test_load_scope_docs_with_text_and_cache_caption():
    docs = [
        {"uid": "d1", "file_path": "/tmp/a", "file_name": "A"},
        {"uid": "d2", "file_path": "/tmp/b", "file_name": "B"},
    ]
    loaded = {"d1": ("t1", "session_hit"), "d2": ("t2", "db_restore")}

    out, stats, failed_uid = load_scope_docs_with_text(
        scope_docs=docs,
        load_document_text_fn=lambda uid, _path: loaded[uid],
    )
    assert failed_uid is None
    assert len(out) == 2
    assert out[0]["text"] == "t1"
    assert stats["session_hit"] == 1
    assert stats["db_restore"] == 1
    assert "数据库缓存恢复" in build_scope_cache_caption(stats)

    _out, _stats, failed_uid = load_scope_docs_with_text(
        scope_docs=docs,
        load_document_text_fn=lambda uid, _path: (None, "error") if uid == "d2" else ("ok", "session_hit"),
    )
    assert failed_uid == "d2"


def test_build_hinted_prompt_and_runtime_helpers():
    memories = [{"memory_type": "semantic", "content": "m1"}]
    hinted = build_hinted_prompt(
        prompt="hello",
        compact_summary="summary",
        user_uuid="u1",
        project_uid="p1",
        detect_language_fn=lambda _text: "en",
        with_language_hint_fn=lambda text, _detector: f"{text}::lang",
        inject_compact_summary_fn=lambda text, summary: f"{text}::{summary}",
        search_project_memory_items_fn=lambda **_kwargs: memories,
        inject_long_term_memory_fn=lambda text, mem: f"{text}::{len(mem)}",
        memory_limit=4,
    )
    assert hinted.startswith("CTXv1\n")
    assert "\nI:sys=paper_qa,col=a2a,tools=-,skills=-\n" in hinted
    assert "\nS:summary\n" in hinted
    assert "\nM:semantic:m1\n" in hinted
    assert "\nL:lang\n" in hinted
    assert hinted.endswith("\nQ:hello")
    assert resolve_runtime_session_id({"configurable": {"thread_id": "tid"}}) == "tid"
    assert resolve_runtime_session_id({}) == "-"
    assert resolve_selected_doc_uid_for_logging([{"uid": "d1"}]) == "d1"
    assert resolve_selected_doc_uid_for_logging([]) == ""


def test_build_hinted_prompt_includes_compact_tool_metadata():
    hinted = build_hinted_prompt(
        prompt="hello",
        compact_summary="summary",
        user_uuid="u1",
        project_uid="p1",
        detect_language_fn=lambda _text: "en",
        with_language_hint_fn=lambda text, _detector: text,
        inject_compact_summary_fn=lambda text, summary: f"{text}::{summary}",
        search_project_memory_items_fn=lambda **_kwargs: [],
        inject_long_term_memory_fn=lambda text, mem: f"{text}::{len(mem)}",
        tool_specs=[
            {
                "name": "start_team",
                "description": "Request sub-agent team runtime",
                "args_schema": '{"fields":["goal","reason","roles_hint"]}',
            }
        ],
        memory_limit=4,
    )

    assert "tools=start_team:Request sub-agent team runtime" in hinted
    assert '{"fields":["goal","reason"' in hinted


def test_archive_target_and_serialization():
    uid, name = resolve_archive_target(
        scope_docs_with_text=[{"uid": "d1", "file_name": "docA"}],
        project_name="P",
    )
    assert uid == "d1"
    assert name == "docA"
    uid, name = resolve_archive_target(
        scope_docs_with_text=[{"uid": "d1"}, {"uid": "d2"}],
        project_name="P",
    )
    assert uid is None
    assert name == "P"
    assert serialize_output_content(answer="a", mindmap_data=None, json_dumps_fn=lambda payload, **_kwargs: payload) == "a"
    payload = serialize_output_content(
        answer="a",
        mindmap_data={"name": "root"},
        json_dumps_fn=lambda body, **_kwargs: f"json:{body['name']}",
    )
    assert payload == "json:root"
