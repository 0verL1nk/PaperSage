import pytest
from pydantic import ValidationError

from utils.schemas import EnqueueResult, FileRecord


def test_enqueue_result_rejects_invalid_mode() -> None:
    with pytest.raises(ValidationError):
        EnqueueResult(mode="invalid", job_id=None)


def test_file_record_normalizes_types() -> None:
    record = FileRecord(file_path=123, file_name=456, uid=789, created_at=0)
    assert record.file_path == "123"
    assert record.file_name == "456"
    assert record.uid == "789"
    assert record.created_at == "0"
