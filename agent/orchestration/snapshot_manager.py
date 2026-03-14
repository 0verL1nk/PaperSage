"""State snapshot management backed by a LangGraph checkpointer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointTuple,
)


@dataclass(frozen=True)
class Snapshot:
    """Snapshot metadata for a thread checkpoint."""

    thread_id: str
    checkpoint_id: str
    created_at: datetime
    metadata: dict[str, object]


class SnapshotManager:
    """Provide save/list/get snapshot primitives for rollback-adjacent flows."""

    def __init__(self, checkpointer: BaseCheckpointSaver[str]) -> None:
        self._checkpointer: BaseCheckpointSaver[str] = checkpointer

    def save_snapshot(
        self,
        thread_id: str,
        checkpoint_id: str,
        metadata: dict[str, object] | None = None,
    ) -> Snapshot:
        """Create snapshot metadata for an existing checkpoint reference."""
        return Snapshot(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(tz=timezone.utc),
            metadata=dict(metadata or {}),
        )

    def list_snapshots(self, thread_id: str) -> list[Snapshot]:
        """List snapshots available for a thread via checkpointer history."""
        config = _checkpoint_config(thread_id)
        snapshots: list[Snapshot] = []

        for checkpoint_tuple in self._checkpointer.list(config):
            tuple_config = checkpoint_tuple.config or {}
            tuple_cfg_obj = tuple_config.get("configurable")
            if not isinstance(tuple_cfg_obj, dict):
                continue
            thread_value = tuple_cfg_obj.get("thread_id")
            checkpoint_value = tuple_cfg_obj.get("checkpoint_id")
            cp_thread_id = thread_value if isinstance(thread_value, str) else thread_id
            checkpoint_id = checkpoint_value if isinstance(checkpoint_value, str) else ""
            if cp_thread_id != thread_id or not checkpoint_id:
                continue

            checkpoint = checkpoint_tuple.checkpoint or {}
            timestamp = checkpoint.get("ts")
            created_at = _parse_checkpoint_timestamp(timestamp)
            metadata = {key: value for key, value in (checkpoint_tuple.metadata or {}).items()}
            snapshots.append(
                Snapshot(
                    thread_id=cp_thread_id,
                    checkpoint_id=checkpoint_id,
                    created_at=created_at,
                    metadata=metadata,
                )
            )

        return snapshots

    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint payload by thread and checkpoint id."""
        checkpoint_tuple = self._get_checkpoint_tuple(thread_id, checkpoint_id)
        if checkpoint_tuple is not None:
            return checkpoint_tuple.checkpoint

        return self._get_checkpoint_payload(thread_id, checkpoint_id)

    def _get_checkpoint_tuple(self, thread_id: str, checkpoint_id: str) -> CheckpointTuple | None:
        config = _checkpoint_config(thread_id, checkpoint_id)
        get_tuple = getattr(self._checkpointer, "get_tuple", None)
        if callable(get_tuple):
            return cast(CheckpointTuple | None, get_tuple(config))

        return None

    def _get_checkpoint_payload(self, thread_id: str, checkpoint_id: str) -> Checkpoint | None:
        config = _checkpoint_config(thread_id, checkpoint_id)

        legacy_get = getattr(self._checkpointer, "get", None)
        if callable(legacy_get):
            try:
                return cast(
                    Checkpoint | None,
                    legacy_get(_checkpoint_config(thread_id), checkpoint_id),
                )
            except TypeError:
                return cast(Checkpoint | None, legacy_get(config))
        return None


def _checkpoint_config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, str] = {"thread_id": thread_id}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return cast(RunnableConfig, cast(object, {"configurable": configurable}))


def _parse_checkpoint_timestamp(timestamp: object) -> datetime:
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)
