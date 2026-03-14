from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

OpenVikingTier = Literal["L0", "L1", "L2"]
DEFAULT_OPENVIKING_SEARCH_TOP_K = 8
MetadataValue = object


@dataclass(frozen=True)
class OpenVikingSearchRequest:
    project_uid: str
    query: str
    tier: OpenVikingTier = "L2"
    top_k: int = DEFAULT_OPENVIKING_SEARCH_TOP_K


@dataclass(frozen=True)
class OpenVikingSearchHit:
    resource_id: str
    project_uid: str
    content: str
    score: float
    tier: OpenVikingTier
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenVikingReadRequest:
    resource_id: str
    tier: OpenVikingTier = "L2"


@dataclass(frozen=True)
class OpenVikingReadResult:
    resource_id: str
    project_uid: str
    content: str
    tier: OpenVikingTier
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenVikingResourceRecord:
    resource_id: str
    project_uid: str
    tier: OpenVikingTier
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


class OpenVikingAdapter(Protocol):
    def add_resource(
        self,
        *,
        project_uid: str,
        content: str,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> str: ...

    def search(self, request: OpenVikingSearchRequest) -> list[OpenVikingSearchHit]: ...

    def read(self, request: OpenVikingReadRequest) -> OpenVikingReadResult: ...

    def list_resources(self, *, project_uid: str) -> list[OpenVikingResourceRecord]: ...

    def delete_resource(self, *, resource_id: str) -> bool: ...
