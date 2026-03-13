from __future__ import annotations

import threading

from agent.settings import load_agent_settings

from .viking_adapter import OpenVikingHttpAdapter

_runtime_adapter: OpenVikingHttpAdapter | None = None
_RUNTIME_ADAPTER_LOCK = threading.Lock()


def get_openviking_adapter() -> OpenVikingHttpAdapter:
    global _runtime_adapter
    with _RUNTIME_ADAPTER_LOCK:
        if _runtime_adapter is None:
            settings = load_agent_settings()
            _runtime_adapter = OpenVikingHttpAdapter(
                base_url=settings.viking_base_url,
                timeout_seconds=settings.viking_timeout_seconds,
                max_retries=settings.viking_max_retries,
                retry_backoff_seconds=settings.viking_retry_backoff_seconds,
            )
        return _runtime_adapter


def reset_openviking_adapter_for_testing() -> None:
    global _runtime_adapter
    with _RUNTIME_ADAPTER_LOCK:
        _runtime_adapter = None
