from __future__ import annotations

import pytest

from agent.adapters.viking_adapter import (
    OpenVikingHttpAdapter,
    VikingAdapterError,
    VikingTimeoutError,
)
from agent.domain.openviking_contracts import OpenVikingReadRequest, OpenVikingSearchRequest


def test_viking_adapter_maps_successful_crud_operations() -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def _transport(
        *,
        method: str,
        url: str,
        timeout_seconds: float,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        _ = timeout_seconds
        calls.append((method, url, payload))
        if method == "POST" and url.endswith("/resources"):
            return {"resource_id": "res-1"}
        if method == "POST" and url.endswith("/search"):
            return {
                "hits": [
                    {
                        "resource_id": "res-1",
                        "project_uid": "project-1",
                        "content": "match",
                        "score": 0.9,
                        "tier": "L2",
                        "metadata": {"source": "paper"},
                    }
                ]
            }
        if method == "GET" and "/resources/res-1" in url:
            return {
                "resource_id": "res-1",
                "project_uid": "project-1",
                "content": "full text",
                "tier": "L2",
                "metadata": {"source": "paper"},
            }
        if method == "GET" and "/projects/project-1/resources" in url:
            return {
                "resources": [
                    {
                        "resource_id": "res-1",
                        "project_uid": "project-1",
                        "tier": "L2",
                        "metadata": {"source": "paper"},
                    }
                ]
            }
        if method == "DELETE" and "/resources/res-1" in url:
            return {"deleted": True}
        raise AssertionError(f"Unexpected request: {method} {url}")

    adapter = OpenVikingHttpAdapter(
        base_url="http://openviking.local",
        timeout_seconds=2.0,
        max_retries=0,
        retry_backoff_seconds=0.0,
        transport=_transport,
    )

    resource_id = adapter.add_resource(project_uid="project-1", content="doc body")
    hits = adapter.search(OpenVikingSearchRequest(project_uid="project-1", query="doc"))
    read_result = adapter.read(OpenVikingReadRequest(resource_id="res-1"))
    resources = adapter.list_resources(project_uid="project-1")
    deleted = adapter.delete_resource(resource_id="res-1")

    assert resource_id == "res-1"
    assert hits[0].score == 0.9
    assert read_result.content == "full text"
    assert resources[0].resource_id == "res-1"
    assert deleted is True
    assert calls[0][2] == {"project_uid": "project-1", "content": "doc body", "metadata": {}}


def test_viking_adapter_retries_timeout_then_succeeds() -> None:
    sleep_calls: list[float] = []
    attempts = 0

    def _transport(
        *,
        method: str,
        url: str,
        timeout_seconds: float,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        _ = (method, url, timeout_seconds, payload)
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("timeout")
        return {"resource_id": "res-2"}

    adapter = OpenVikingHttpAdapter(
        base_url="http://openviking.local",
        timeout_seconds=1.0,
        max_retries=1,
        retry_backoff_seconds=0.25,
        transport=_transport,
        sleep=sleep_calls.append,
    )

    resource_id = adapter.add_resource(project_uid="project-1", content="doc")

    assert resource_id == "res-2"
    assert attempts == 2
    assert sleep_calls == [0.25]


def test_viking_adapter_raises_timeout_after_retry_budget_exhausted() -> None:
    def _transport(
        *,
        method: str,
        url: str,
        timeout_seconds: float,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        _ = (method, url, timeout_seconds, payload)
        raise TimeoutError("timeout")

    adapter = OpenVikingHttpAdapter(
        base_url="http://openviking.local",
        timeout_seconds=0.2,
        max_retries=1,
        retry_backoff_seconds=0.0,
        transport=_transport,
    )

    with pytest.raises(VikingTimeoutError):
        _ = adapter.search(OpenVikingSearchRequest(project_uid="project-1", query="q"))


def test_viking_adapter_raises_error_on_invalid_response_shape() -> None:
    def _transport(
        *,
        method: str,
        url: str,
        timeout_seconds: float,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        _ = (method, url, timeout_seconds, payload)
        return {"resource_id": ""}

    adapter = OpenVikingHttpAdapter(
        base_url="http://openviking.local",
        timeout_seconds=1.0,
        max_retries=0,
        retry_backoff_seconds=0.0,
        transport=_transport,
    )

    with pytest.raises(VikingAdapterError):
        _ = adapter.add_resource(project_uid="project-1", content="doc")
