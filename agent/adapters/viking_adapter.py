from __future__ import annotations

import json
import time
from collections.abc import Callable
from http.client import HTTPResponse
from json import JSONDecodeError
from typing import Protocol, cast
from urllib import error, parse, request

from agent.domain.openviking_contracts import (
    OpenVikingReadRequest,
    OpenVikingReadResult,
    OpenVikingResourceRecord,
    OpenVikingSearchHit,
    OpenVikingSearchRequest,
    OpenVikingTier,
)

JsonDict = dict[str, object]


class VikingAdapterError(RuntimeError):
    pass


class VikingTimeoutError(VikingAdapterError):
    pass


class VikingTransport(Protocol):
    def __call__(
        self,
        *,
        method: str,
        url: str,
        timeout_seconds: float,
        payload: JsonDict | None,
    ) -> JsonDict: ...


def _required_str(data: JsonDict, key: str, *, context: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise VikingAdapterError(f"OpenViking {context} response missing {key}")
    return value


def _default_transport(
    *,
    method: str,
    url: str,
    timeout_seconds: float,
    payload: JsonDict | None,
) -> JsonDict:
    body: bytes | None = None
    headers: dict[str, str] = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, data=body, headers=headers, method=method)
    try:
        typed_response = cast(HTTPResponse, request.urlopen(req, timeout=timeout_seconds))  # nosec B310
        with typed_response:
            text = typed_response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise VikingAdapterError(f"OpenViking HTTP error: {exc.code}") from exc
    except TimeoutError as exc:
        raise VikingTimeoutError(f"OpenViking timeout: {exc}") from exc
    except error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise VikingTimeoutError(f"OpenViking timeout: {exc}") from exc
        raise VikingAdapterError(f"OpenViking request failed: {exc}") from exc

    if not text.strip():
        return {}
    try:
        decoded = _load_json_object(text)
    except JSONDecodeError as exc:
        raise VikingAdapterError("OpenViking returned invalid JSON response") from exc
    if not isinstance(decoded, dict):
        raise VikingAdapterError("OpenViking returned a non-object JSON response")
    return cast(JsonDict, decoded)


def _coerce_tier(value: object) -> OpenVikingTier:
    if value in {"L0", "L1", "L2"}:
        return cast(OpenVikingTier, value)
    raise VikingAdapterError(f"Invalid OpenViking tier: {value!r}")


def _coerce_score(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise VikingAdapterError(f"Invalid OpenViking score value: {value!r}") from exc
    raise VikingAdapterError(f"Invalid OpenViking score type: {type(value).__name__}")


def _load_json_object(text: str) -> object:
    return json.loads(text)  # pyright: ignore[reportAny]


class OpenVikingHttpAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        transport: VikingTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must not be empty")
        self._base_url: str = normalized_base_url
        self._timeout_seconds: float = max(0.1, timeout_seconds)
        self._max_retries: int = max(0, max_retries)
        self._retry_backoff_seconds: float = max(0.0, retry_backoff_seconds)
        self._transport: VikingTransport = transport or _default_transport
        self._sleep: Callable[[float], None] = sleep or time.sleep

    def add_resource(
        self,
        *,
        project_uid: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        response = self._request(
            method="POST",
            path="/resources",
            payload={
                "project_uid": project_uid,
                "content": content,
                "metadata": metadata or {},
            },
        )
        return _required_str(response, "resource_id", context="add_resource")

    def search(self, request: OpenVikingSearchRequest) -> list[OpenVikingSearchHit]:
        response = self._request(
            method="POST",
            path="/search",
            payload={
                "project_uid": request.project_uid,
                "query": request.query,
                "tier": request.tier,
                "top_k": request.top_k,
            },
        )
        raw_hits_obj = response.get("hits", [])
        if not isinstance(raw_hits_obj, list):
            raise VikingAdapterError("OpenViking search response hits must be a list")
        raw_hits = cast(list[object], raw_hits_obj)
        hits: list[OpenVikingSearchHit] = []
        for raw_hit_item in raw_hits:
            if not isinstance(raw_hit_item, dict):
                raise VikingAdapterError("OpenViking search hit must be an object")
            raw_hit = cast(JsonDict, raw_hit_item)
            hits.append(
                OpenVikingSearchHit(
                    resource_id=_required_str(raw_hit, "resource_id", context="search hit"),
                    project_uid=_required_str(raw_hit, "project_uid", context="search hit"),
                    content=str(raw_hit.get("content", "")),
                    score=_coerce_score(raw_hit.get("score", 0.0)),
                    tier=_coerce_tier(raw_hit.get("tier")),
                    metadata=cast(dict[str, object], raw_hit.get("metadata") or {}),
                )
            )
        return hits

    def read(self, request: OpenVikingReadRequest) -> OpenVikingReadResult:
        query = parse.urlencode({"tier": request.tier})
        response = self._request(
            method="GET",
            path=f"/resources/{request.resource_id}?{query}",
            payload=None,
        )
        return OpenVikingReadResult(
            resource_id=_required_str(response, "resource_id", context="read"),
            project_uid=_required_str(response, "project_uid", context="read"),
            content=str(response.get("content", "")),
            tier=_coerce_tier(response.get("tier")),
            metadata=cast(dict[str, object], response.get("metadata") or {}),
        )

    def list_resources(self, *, project_uid: str) -> list[OpenVikingResourceRecord]:
        response = self._request(
            method="GET",
            path=f"/projects/{project_uid}/resources",
            payload=None,
        )
        raw_resources_obj = response.get("resources", [])
        if not isinstance(raw_resources_obj, list):
            raise VikingAdapterError("OpenViking list_resources response resources must be a list")
        raw_resources = cast(list[object], raw_resources_obj)
        records: list[OpenVikingResourceRecord] = []
        for raw_item in raw_resources:
            if not isinstance(raw_item, dict):
                raise VikingAdapterError("OpenViking resource entry must be an object")
            item = cast(JsonDict, raw_item)
            records.append(
                OpenVikingResourceRecord(
                    resource_id=_required_str(item, "resource_id", context="list_resources"),
                    project_uid=_required_str(item, "project_uid", context="list_resources"),
                    tier=_coerce_tier(item.get("tier")),
                    metadata=cast(dict[str, object], item.get("metadata") or {}),
                )
            )
        return records

    def delete_resource(self, *, resource_id: str) -> bool:
        response = self._request(
            method="DELETE",
            path=f"/resources/{resource_id}",
            payload=None,
        )
        return bool(response.get("deleted", False))

    def _request(
        self,
        *,
        method: str,
        path: str,
        payload: JsonDict | None,
    ) -> JsonDict:
        last_timeout: BaseException | None = None
        url = f"{self._base_url}{path}"
        for attempt in range(self._max_retries + 1):
            try:
                return self._transport(
                    method=method,
                    url=url,
                    timeout_seconds=self._timeout_seconds,
                    payload=payload,
                )
            except (TimeoutError, VikingTimeoutError) as exc:
                last_timeout = exc
                if attempt >= self._max_retries:
                    raise VikingTimeoutError(
                        f"OpenViking request timed out after {attempt + 1} attempt(s): {method} {path}"
                    ) from exc
                if self._retry_backoff_seconds > 0.0:
                    self._sleep(self._retry_backoff_seconds)
            except VikingAdapterError:
                raise
            except Exception as exc:
                raise VikingAdapterError(
                    f"OpenViking request failed for {method} {path}: {exc}"
                ) from exc
        if last_timeout is not None:
            raise VikingTimeoutError(
                f"OpenViking request timed out after {self._max_retries + 1} attempt(s): {method} {path}"
            ) from last_timeout
        raise VikingAdapterError(f"OpenViking request failed unexpectedly: {method} {path}")
