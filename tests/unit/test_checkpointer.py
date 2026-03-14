import os
import tempfile
import threading

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


class TestSqliteThreadSafety:
    def test_sqlite_checkpointer_works_in_different_thread(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            checkpointer = create_checkpointer("sqlite", conn_string=db_path)
            result_holder: dict = {"error": None}

            def access_from_thread():
                try:
                    list(checkpointer.list({"configurable": {"thread_id": "test"}}))
                except Exception as e:
                    result_holder["error"] = e

            thread = threading.Thread(target=access_from_thread)
            thread.start()
            thread.join(timeout=5)

            assert result_holder["error"] is None, (
                f"Cross-thread access failed: {result_holder['error']}"
            )
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
