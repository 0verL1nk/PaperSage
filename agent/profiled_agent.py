from collections.abc import Callable
from typing import Any

from .profiles import AgentProfile, resolve_agent_profile
from .session_factory import (
    AgentDependencies,
    AgentRuntimeOptions,
    AgentSession,
    create_agent_session,
)


def create_profiled_agent_session(
    *,
    profile: AgentProfile | str,
    llm: Any,
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None,
    list_documents_fn: Callable[[], list[dict[str, Any]]] | None = None,
    system_prompt: str | None = None,
    document_name: str | None = None,
    project_name: str | None = None,
    scope_summary: str | None = None,
    project_uid: str | None = None,
    session_uid: str | None = None,
    user_uuid: str | None = None,
    thread_id: str | None = None,
) -> AgentSession:
    resolved_profile = _resolve_profile(profile)
    runtime_options = AgentRuntimeOptions(
        llm=llm,
        document_name=document_name,
        project_name=project_name,
        scope_summary=scope_summary,
        system_prompt=system_prompt,
        thread_id=thread_id,
    )
    dependencies = AgentDependencies(
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        read_document_fn=read_document_fn,
        list_documents_fn=list_documents_fn,
        project_uid=project_uid,
        session_uid=session_uid,
        user_uuid=user_uuid,
    )
    return create_agent_session(
        profile=resolved_profile,
        deps=dependencies,
        options=runtime_options,
    )


def _resolve_profile(profile: AgentProfile | str) -> AgentProfile:
    if isinstance(profile, AgentProfile):
        return profile
    return resolve_agent_profile(profile)


__all__ = ["create_profiled_agent_session", "AgentSession"]
