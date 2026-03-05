from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    project_uid: str = Field(default="")
    doc_uid: str = Field(default="")
    doc_name: str = Field(default="")
    chunk_id: str
    text: str
    score: float = Field(default=0.0)
    page_no: int | None = None
    offset_start: int | None = None
    offset_end: int | None = None


class EvidencePayload(BaseModel):
    evidences: list[EvidenceItem] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
