from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.domain.openviking_contracts import OpenVikingReadResult, OpenVikingSearchHit


@dataclass
class _StoredResource:
    resource_id: str
    project_uid: str
    content: str
    metadata: dict[str, object]


class _FakeOpenVikingAdapter:
    def __init__(self) -> None:
        self._resources: dict[str, _StoredResource] = {}
        self._counter = 0

    def add_resource(
        self,
        *,
        project_uid: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        self._counter += 1
        resource_id = f"res-{self._counter}"
        self._resources[resource_id] = _StoredResource(
            resource_id=resource_id,
            project_uid=project_uid,
            content=content,
            metadata=dict(metadata or {}),
        )
        return resource_id

    def search(self, request) -> list[OpenVikingSearchHit]:
        query_terms = {term for term in str(request.query or "").lower().split() if term}
        hits: list[OpenVikingSearchHit] = []
        for item in self._resources.values():
            if item.project_uid != request.project_uid:
                continue
            haystack = f"{item.content}\n{item.metadata}".lower()
            score = 1.0
            if query_terms:
                score += float(sum(1 for term in query_terms if term in haystack))
            hits.append(
                OpenVikingSearchHit(
                    resource_id=item.resource_id,
                    project_uid=item.project_uid,
                    content=item.content,
                    score=score,
                    tier="L2",
                    metadata=dict(item.metadata),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(1, int(getattr(request, "top_k", 8)))]

    def read(self, request) -> OpenVikingReadResult:
        item = self._resources[str(request.resource_id)]
        return OpenVikingReadResult(
            resource_id=item.resource_id,
            project_uid=item.project_uid,
            content=item.content,
            tier="L2",
            metadata=dict(item.metadata),
        )


@pytest.fixture(autouse=True)
def _stub_skills_openviking(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeOpenVikingAdapter()
    monkeypatch.setattr("agent.skills.loader.get_openviking_adapter", lambda: adapter)
