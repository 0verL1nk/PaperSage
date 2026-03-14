import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.orchestration import (
    CheckpointerType,
    UnsupportedCheckpointerTypeError,
    create_checkpointer,
)


class TestCreateCheckpointer:
    def test_create_memory_checkpointer(self):
        result = create_checkpointer("memory")
        assert isinstance(result, InMemorySaver)

    def test_create_sqlite_checkpointer_with_default_conn(self):
        result = create_checkpointer("sqlite")
        assert isinstance(result, SqliteSaver)

    def test_create_sqlite_checkpointer_with_custom_conn(self):
        result = create_checkpointer("sqlite", conn_string=":memory:")
        assert isinstance(result, SqliteSaver)

    def test_create_sqlite_checkpointer_with_file_path(self):
        result = create_checkpointer("sqlite", conn_string="./test_checkpoints.db")
        assert isinstance(result, SqliteSaver)

    def test_unsupported_type_raises_error(self):
        with pytest.raises(UnsupportedCheckpointerTypeError) as exc_info:
            create_checkpointer("postgres")
        assert "Unsupported checkpointer type: 'postgres'" in str(exc_info.value)
        assert "Supported types: ['memory', 'sqlite']" in str(exc_info.value)

    def test_unsupported_type_error_has_type_attribute(self):
        with pytest.raises(UnsupportedCheckpointerTypeError) as exc_info:
            create_checkpointer("invalid")
        assert exc_info.value.checkpointer_type == "invalid"


class TestCheckpointerType:
    def test_checkpointer_type_literal_allows_memory(self):
        checkpointer_type: CheckpointerType = "memory"
        result = create_checkpointer(checkpointer_type)
        assert isinstance(result, InMemorySaver)

    def test_checkpointer_type_literal_allows_sqlite(self):
        checkpointer_type: CheckpointerType = "sqlite"
        result = create_checkpointer(checkpointer_type)
        assert isinstance(result, SqliteSaver)
