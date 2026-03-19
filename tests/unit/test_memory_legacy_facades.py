import utils.utils as legacy_utils


def test_legacy_memory_facades_are_not_exposed() -> None:
    assert not hasattr(legacy_utils, "upsert_project_memory_item")
    assert not hasattr(legacy_utils, "list_project_memory_items")
    assert not hasattr(legacy_utils, "search_project_memory_items")
    assert not hasattr(legacy_utils, "touch_memory_items")
