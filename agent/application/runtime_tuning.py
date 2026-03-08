from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

RUNTIME_TUNING_ENV_BY_KEY: dict[str, str] = {
    "agent_policy_async_enabled": "AGENT_POLICY_ASYNC_ENABLED",
    "agent_policy_async_refresh_seconds": "AGENT_POLICY_ASYNC_REFRESH_SECONDS",
    "agent_policy_async_min_confidence": "AGENT_POLICY_ASYNC_MIN_CONFIDENCE",
    "agent_policy_async_max_staleness_seconds": "AGENT_POLICY_ASYNC_MAX_STALENESS_SECONDS",
    "rag_index_batch_size": "RAG_INDEX_BATCH_SIZE",
    "agent_document_text_cache_max_chars": "AGENT_DOCUMENT_TEXT_CACHE_MAX_CHARS",
    "local_rag_project_max_chars": "LOCAL_RAG_PROJECT_MAX_CHARS",
    "local_rag_project_max_chunks": "LOCAL_RAG_PROJECT_MAX_CHUNKS",
}


def apply_runtime_tuning_env(
    *,
    settings: Mapping[str, Any],
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    target_environ = os.environ if environ is None else environ
    applied: dict[str, str] = {}

    for key, env_name in RUNTIME_TUNING_ENV_BY_KEY.items():
        value = settings.get(key)
        if value is None:
            target_environ.pop(env_name, None)
            continue

        if isinstance(value, bool):
            normalized = "true" if value else "false"
        else:
            normalized = str(value)

        target_environ[env_name] = normalized
        applied[env_name] = normalized

    return applied
