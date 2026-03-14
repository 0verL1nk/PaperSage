import importlib
from collections.abc import Callable
from datetime import datetime
from typing import Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver


class SnapshotLike(Protocol):
    thread_id: str
    checkpoint_id: str
    created_at: datetime
    metadata: dict[str, object]


class SnapshotManagerLike(Protocol):
    def save_snapshot(
        self,
        thread_id: str,
        checkpoint_id: str,
        metadata: dict[str, object] | None = None,
    ) -> SnapshotLike: ...

    def list_snapshots(self, thread_id: str) -> list[SnapshotLike]: ...

    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> Checkpoint | None: ...


def _create_manager(checkpointer: InMemorySaver) -> SnapshotManagerLike:
    module = importlib.import_module("agent.orchestration.snapshot_manager")
    manager_class = cast(
        Callable[[InMemorySaver], SnapshotManagerLike], getattr(module, "SnapshotManager")
    )
    return manager_class(checkpointer)


def _put_checkpoint(checkpointer: InMemorySaver, thread_id: str, source: str) -> str:
    config = cast(
        RunnableConfig,
        cast(object, {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}),
    )
    metadata = cast(
        CheckpointMetadata,
        cast(object, {"source": source, "step": 1, "writes": {}, "score": 1}),
    )
    checkpoint_config = checkpointer.put(config, empty_checkpoint(), metadata, {})
    configurable = checkpoint_config.get("configurable")
    if not isinstance(configurable, dict):
        raise AssertionError("missing configurable checkpoint data")
    checkpoint_id = configurable.get("checkpoint_id")
    assert isinstance(checkpoint_id, str)
    return checkpoint_id


def test_snapshot_manager_save_returns_snapshot_metadata() -> None:
    manager = _create_manager(InMemorySaver())

    snapshot = manager.save_snapshot("thread_1", "checkpoint_1", {"note": "test"})

    assert snapshot.thread_id == "thread_1"
    assert snapshot.checkpoint_id == "checkpoint_1"
    assert snapshot.metadata == {"note": "test"}
    assert isinstance(snapshot.created_at, datetime)


def test_snapshot_manager_list_returns_snapshots_from_checkpointer() -> None:
    checkpointer = InMemorySaver()
    expected_thread_1_ids = {
        _put_checkpoint(checkpointer, "thread_1", "input"),
        _put_checkpoint(checkpointer, "thread_1", "loop"),
    }
    _ = _put_checkpoint(checkpointer, "thread_2", "other")
    manager = _create_manager(checkpointer)

    snapshots = manager.list_snapshots("thread_1")

    assert isinstance(snapshots, list)
    assert {snapshot.checkpoint_id for snapshot in snapshots} == expected_thread_1_ids
    assert all(snapshot.thread_id == "thread_1" for snapshot in snapshots)


def test_snapshot_manager_get_checkpoint_by_thread_and_checkpoint_id() -> None:
    checkpointer = InMemorySaver()
    checkpoint_id = _put_checkpoint(checkpointer, "thread_1", "input")
    manager = _create_manager(checkpointer)

    checkpoint = manager.get_checkpoint("thread_1", checkpoint_id)

    assert checkpoint is not None
    assert checkpoint["id"] == checkpoint_id


def test_snapshot_manager_get_checkpoint_returns_none_when_missing() -> None:
    manager = _create_manager(InMemorySaver())

    checkpoint = manager.get_checkpoint("thread_1", "missing")

    assert checkpoint is None
