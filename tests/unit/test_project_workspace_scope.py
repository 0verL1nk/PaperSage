from ui.project_workspace import reconcile_scope_selection


def test_reconcile_scope_selection_defaults_to_all_docs():
    selected, known = reconcile_scope_selection(
        all_uids=["d1", "d2", "d3"],
        persisted_selected=None,
        known_uids=None,
    )
    assert selected == ["d1", "d2", "d3"]
    assert known == ["d1", "d2", "d3"]


def test_reconcile_scope_selection_auto_includes_new_docs_when_previously_full():
    selected, known = reconcile_scope_selection(
        all_uids=["d1", "d2", "d3"],
        persisted_selected=["d1", "d2"],
        known_uids=["d1", "d2"],
    )
    assert selected == ["d1", "d2", "d3"]
    assert known == ["d1", "d2", "d3"]


def test_reconcile_scope_selection_keeps_custom_subset_on_new_docs():
    selected, known = reconcile_scope_selection(
        all_uids=["d1", "d2", "d3"],
        persisted_selected=["d1"],
        known_uids=["d1", "d2"],
    )
    assert selected == ["d1"]
    assert known == ["d1", "d2", "d3"]


def test_reconcile_scope_selection_prunes_removed_docs():
    selected, known = reconcile_scope_selection(
        all_uids=["d1", "d3"],
        persisted_selected=["d1", "d2"],
        known_uids=["d1", "d2", "d3"],
    )
    assert selected == ["d1"]
    assert known == ["d1", "d3"]
