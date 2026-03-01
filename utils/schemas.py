from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class EnqueueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["queued", "sync"]
    job_id: str | None = None


class FileRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_path: str
    file_name: str
    uid: str
    created_at: str

    @field_validator("file_path", "file_name", "uid", "created_at", mode="before")
    @classmethod
    def _to_string(cls, value: object) -> str:
        return "" if value is None else str(value)
