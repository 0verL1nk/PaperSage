import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...settings import load_agent_settings

_WORKSPACE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _normalize_workspace_segment(raw: str, *, default: str) -> str:
    normalized = _WORKSPACE_SEGMENT_RE.sub("_", str(raw or "").strip()).strip("._-")
    return normalized[:80] if normalized else default


def _build_leader_workspace_root(*, project_uid: str, session_uid: str) -> str:
    base_dir = str(os.getenv("AGENT_WORKSPACE_BASE_DIR", "") or "").strip()
    base = Path(base_dir) if base_dir else (Path.cwd() / ".agent" / "workspaces")
    project_segment = _normalize_workspace_segment(project_uid, default="default_project")
    session_segment = _normalize_workspace_segment(session_uid, default="default_session")
    return str((base / project_segment / session_segment / "leader").resolve())


def ensure_agent_runtime(
    *,
    session_state: dict[str, Any],
    logger,
    project_uid: str,
    session_uid: str,
    project_name: str,
    scope_docs: list[dict],
    scope_signature: str,
    mode_leader: str,
    build_session_key_fn,
    clear_project_runtime_fn,
    normalize_evidence_items_fn,
    get_user_api_key_fn,
    get_user_model_name_fn,
    get_user_base_url_fn,
    create_chat_model_fn,
    create_project_evidence_retriever_fn,
    create_leader_session_fn,
    get_user_policy_router_model_name_fn=None,
    get_user_policy_router_base_url_fn=None,
    get_user_policy_router_api_key_fn=None,
) -> None:
    logger.debug(
        "Ensuring project sessions: project=%s docs=%s",
        project_name,
        len(scope_docs),
    )
    leader_session_key = build_session_key_fn(project_uid, session_uid, mode_leader)
    leader_sessions = session_state.get("paper_agent_sessions", {})
    evidence_retrievers = session_state.get("paper_evidence_retrievers", {})
    llm_map = session_state.get("paper_project_llms", {})
    policy_llm_map = session_state.get("paper_project_policy_llms", {})
    search_fn_map = session_state.get("paper_project_search_document_fns", {})
    signatures = session_state.get("paper_project_scope_signatures", {})

    current_leader_session = leader_sessions.get(leader_session_key)
    current_evidence_retriever = evidence_retrievers.get(project_uid)
    current_signature = signatures.get(project_uid)

    if current_signature != scope_signature and current_signature is not None:
        clear_project_runtime_fn(project_uid)
        current_leader_session = None
        current_evidence_retriever = None
        tool_specs_map = session_state.get("paper_project_tool_specs", {})
        tool_specs_map.pop(project_uid, None)
        session_state["paper_project_tool_specs"] = tool_specs_map

    if current_leader_session and current_evidence_retriever:
        logger.debug("Reusing existing leader session and evidence retriever: session=%s", session_uid)
        session_state["paper_agent"] = current_leader_session["agent"]
        session_state["paper_agent_runtime_config"] = current_leader_session["runtime_config"]
        session_state["paper_evidence_retriever"] = current_evidence_retriever
        session_state["paper_leader_llm"] = llm_map.get(project_uid)
        session_state["paper_policy_router_llm"] = (
            policy_llm_map.get(project_uid) or llm_map.get(project_uid)
        )
        session_state["paper_search_document_fn"] = search_fn_map.get(project_uid)
        session_state["agent_current_project"] = project_name
        tool_specs_map = session_state.get("paper_project_tool_specs", {})
        reused_tool_specs = current_leader_session.get("tool_specs", [])
        if isinstance(reused_tool_specs, list):
            tool_specs_map[project_uid] = reused_tool_specs
            session_state["paper_project_tool_specs"] = tool_specs_map
            session_state["paper_current_tool_specs"] = reused_tool_specs
        return

    api_key = get_user_api_key_fn()
    if not api_key:
        raise ValueError("请先在“设置中心”页面配置您的 API Key")
    user_model = get_user_model_name_fn()
    if not user_model:
        raise ValueError("请先在“设置中心”页面配置模型名称")
    user_base_url = get_user_base_url_fn()
    llm = create_chat_model_fn(
        api_key=api_key,
        model_name=user_model,
        base_url=user_base_url,
    )
    logger.debug("Built chat model for agent center: model=%s", user_model)
    llm_map[project_uid] = llm
    session_state["paper_project_llms"] = llm_map
    session_state["paper_leader_llm"] = llm

    settings = load_agent_settings()
    policy_llm = llm
    router_model_name = ""
    if callable(get_user_policy_router_model_name_fn):
        router_model_name = str(get_user_policy_router_model_name_fn() or "").strip()
    if not router_model_name:
        router_model_name = str(settings.agent_policy_router_model_name or "").strip()

    router_base_url = ""
    if callable(get_user_policy_router_base_url_fn):
        router_base_url = str(get_user_policy_router_base_url_fn() or "").strip()
    if not router_base_url:
        router_base_url = str(settings.agent_policy_router_base_url or "").strip()

    router_api_key = ""
    if callable(get_user_policy_router_api_key_fn):
        router_api_key = str(get_user_policy_router_api_key_fn() or "").strip()
    if not router_api_key:
        router_api_key = str(os.getenv("AGENT_POLICY_ROUTER_API_KEY", "") or "").strip()
    if not router_api_key:
        router_api_key = api_key
    if router_model_name and router_api_key:
        try:
            policy_llm = create_chat_model_fn(
                api_key=router_api_key,
                model_name=router_model_name,
                base_url=router_base_url or user_base_url,
                temperature=settings.agent_policy_router_temperature,
            )
            logger.debug("Built policy router model: model=%s", router_model_name)
        except Exception as exc:
            logger.warning(
                "Failed to build policy router model, fallback to leader model: %s",
                exc,
            )
            policy_llm = llm
    policy_llm_map[project_uid] = policy_llm
    session_state["paper_project_policy_llms"] = policy_llm_map
    session_state["paper_policy_router_llm"] = policy_llm

    search_document_evidence_fn = current_evidence_retriever
    if search_document_evidence_fn is None:
        logger.debug("Building project evidence retriever (cache miss)")
        documents = [
            {
                "doc_uid": str(item["uid"]),
                "doc_name": str(item["file_name"]),
                "text": str(item["text"]),
            }
            for item in scope_docs
            if isinstance(item.get("text"), str)
        ]
        search_document_evidence_fn = create_project_evidence_retriever_fn(
            documents=documents,
            project_uid=project_uid,
        )
        evidence_retrievers[project_uid] = search_document_evidence_fn
        session_state["paper_evidence_retrievers"] = evidence_retrievers

    def search_document_fn(query: str) -> str:
        payload = search_document_evidence_fn(query)
        evidence_items = normalize_evidence_items_fn(payload)
        return "\n".join(item.get("text", "") for item in evidence_items)

    search_fn_map[project_uid] = search_document_fn
    session_state["paper_project_search_document_fns"] = search_fn_map
    session_state["paper_search_document_fn"] = search_document_fn

    read_document_fn = None
    if len(scope_docs) == 1:
        only_doc = scope_docs[0]
        only_text = str(only_doc.get("text") or "")

        def _read_document_fn(offset: int = 0, limit: int = 2000) -> tuple[str, int]:
            total_len = len(only_text)
            content = only_text[offset : offset + limit]
            return content, total_len

        read_document_fn = _read_document_fn

    scope_names = [str(item.get("file_name") or "") for item in scope_docs]
    scope_preview = ", ".join(scope_names[:5])
    if len(scope_names) > 5:
        scope_preview = f"{scope_preview} ... (+{len(scope_names) - 5})"

    if not current_leader_session:
        logger.info("Creating new leader project session: session=%s", session_uid)
        workspace_root = _build_leader_workspace_root(
            project_uid=project_uid,
            session_uid=session_uid,
        )
        agent_session = create_leader_session_fn(
            llm=llm,
            search_document_fn=search_document_fn,
            search_document_evidence_fn=search_document_evidence_fn,
            read_document_fn=read_document_fn,
            document_name=scope_preview or "项目范围",
            project_name=project_name,
            scope_summary=scope_preview or "空范围",
            workspace_root=workspace_root,
        )
        leader_sessions[leader_session_key] = {
            "agent": agent_session.agent,
            "runtime_config": agent_session.runtime_config,
            "tool_specs": agent_session.tool_specs,
        }
        session_state["paper_agent_sessions"] = leader_sessions
        tool_specs_map = session_state.get("paper_project_tool_specs", {})
        tool_specs_map[project_uid] = agent_session.tool_specs
        session_state["paper_project_tool_specs"] = tool_specs_map
        session_state["paper_current_tool_specs"] = agent_session.tool_specs

    signatures[project_uid] = scope_signature
    session_state["paper_project_scope_signatures"] = signatures
    session_state["paper_agent"] = session_state["paper_agent_sessions"][leader_session_key]["agent"]
    session_state["paper_agent_runtime_config"] = session_state["paper_agent_sessions"][
        leader_session_key
    ]["runtime_config"]
    session_state["paper_evidence_retriever"] = search_document_evidence_fn
    session_state["agent_current_project"] = project_name
    logger.info("Project leader session ready")


def prepare_agent_session(
    *,
    logger,
    has_cached_session_fn,
    ensure_agent_runtime_fn,
    cached_caption_fn: Callable[[], None],
    build_captioned_fn: Callable[[Callable[[], None]], None],
    project_uid: str,
    session_uid: str,
    project_name: str,
    scope_docs: list[dict],
    scope_signature: str,
) -> None:
    has_cached_session = has_cached_session_fn(project_uid, session_uid, scope_signature)
    logger.debug("Agent session cache status: has_cached=%s", has_cached_session)
    if has_cached_session:
        ensure_agent_runtime_fn(project_uid, session_uid, project_name, scope_docs, scope_signature)
        cached_caption_fn()
        return

    build_captioned_fn(
        lambda: ensure_agent_runtime_fn(
            project_uid,
            session_uid,
            project_name,
            scope_docs,
            scope_signature,
        )
    )
