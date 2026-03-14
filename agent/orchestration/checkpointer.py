"""LangGraph Checkpointer Factory.

Provides factory helpers for creating LangGraph checkpointers (memory / sqlite).
"""

import sqlite3
from typing import Literal, final

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


# Supported checkpointer types
CheckpointerType = Literal["memory", "sqlite"]


@final
class UnsupportedCheckpointerTypeError(ValueError):
    """Raised when an unsupported checkpointer type is requested."""

    def __init__(self, checkpointer_type: str) -> None:
        self.checkpointer_type: str = checkpointer_type
        supported = ["memory", "sqlite"]
        super().__init__(
            f"Unsupported checkpointer type: {checkpointer_type!r}. Supported types: {supported}"
        )


def create_checkpointer(
    checkpointer_type: CheckpointerType,
    *,
    conn_string: str | None = None,
) -> InMemorySaver | SqliteSaver:
    """Create a LangGraph checkpointer based on the specified type.

    Args:
        checkpointer_type: Type of checkpointer to create ("memory" or "sqlite").
        conn_string: Connection string for SQLite checkpointer.
                     If None and type is "sqlite", uses ":memory:" (in-memory DB).

    Returns:
        An InMemorySaver for "memory" type, or SqliteSaver for "sqlite" type.

    Raises:
        UnsupportedCheckpointerTypeError: If checkpointer_type is not supported.

    Examples:
        >>> memory_checkpointer = create_checkpointer("memory")
        >>> sqlite_checkpointer = create_checkpointer("sqlite", conn_string="./checkpoints.db")
    """
    if checkpointer_type == "memory":
        return InMemorySaver()
    if checkpointer_type == "sqlite":
        conn_string = conn_string if conn_string is not None else ":memory:"
        conn = sqlite3.connect(conn_string, check_same_thread=False)
        return SqliteSaver(conn)
    raise UnsupportedCheckpointerTypeError(checkpointer_type)


__all__ = [
    "CheckpointerType",
    "UnsupportedCheckpointerTypeError",
    "create_checkpointer",
]
