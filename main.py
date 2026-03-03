import json
import logging
import os
from uuid import uuid4

import streamlit as st

DEBUG_MODE = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}

from utils import extract_files
from agent.agent_center_runner import execute_assistant_turn
from agent.archive import save_agent_output
from agent.context_governance import (
    auto_compact_messages,
    build_context_usage_snapshot,
    extract_skill_context_texts_from_trace,
    inject_compact_summary,
    should_trigger_auto_compact,
)
from agent.logging_utils import configure_application_logging, logging_context
from agent.llm_provider import build_openai_compatible_chat_model
from agent.metrics import record_query_metrics
from agent.rag_hybrid import (
    build_local_evidence_retriever_with_settings,
    build_project_evidence_retriever_with_settings,
)
from agent.multi_agent_a2a import (
    WORKFLOW_REACT,
    create_multi_agent_a2a_session,
)
from agent.paper_agent import create_paper_agent_session
from agent.session_state import initialize_agent_center_session_state
from ui.project_workspace import (
    build_project_doc_count_map,
    inject_workspace_styles,
    render_project_context_hint,
    render_workspace_status_bar,
    select_project_sidebar,
    select_scope_documents_drawer,
)
from ui.theme import inject_global_theme
from agent.ui_helpers import (
    _get_doc_metrics,
    _infer_output_type,
    _normalize_evidence_items,
    _render_chat_history,
    _render_context_usage,
    _render_output_archive,
    _render_workflow_metrics,
)
from utils.utils import (
    detect_language,
    ensure_local_user,
    ensure_default_project_for_user,
    get_content_by_uid,
    list_project_files,
    list_projects,
    get_user_api_key,
    get_user_base_url,
    get_user_files,
    get_user_model_name,
    init_database,
    save_content_to_database,
)

configure_application_logging(debug_mode=DEBUG_MODE, default_level="INFO")
logger = logging.getLogger("llm_app.agent_center")


st.set_page_config(page_title="Agent 中心", page_icon="🤖")
st.title("🤖 Agent 中心")
inject_global_theme()
init_database("./database.sqlite")
if "uuid" not in st.session_state or not st.session_state["uuid"]:
    st.session_state["uuid"] = "local-user"
ensure_local_user(st.session_state["uuid"], db_name="./database.sqlite")
ensure_default_project_for_user(st.session_state["uuid"], db_name="./database.sqlite")

if "files" not in st.session_state:
    st.session_state["files"] = []
if "projects" not in st.session_state:
    st.session_state["projects"] = []


def _load_files_from_db() -> None:
    raw_files = get_user_files(st.session_state["uuid"])
    st.session_state["files"] = []
    for file in raw_files:
        st.session_state["files"].append(
            {
                "file_path": file["file_path"],
                "file_name": file["file_name"],
                "uid": file["uid"],
                "created_at": file["created_at"],
            }
        )
    logger.info("Loaded file list from DB: count=%s", len(st.session_state["files"]))


def _load_projects_from_db() -> None:
    st.session_state["projects"] = list_projects(
        st.session_state["uuid"], include_archived=False
    )


def _session_key(project_uid: str, mode: str) -> str:
    return f"{mode}:{project_uid}"


def _has_cached_agent_session(project_uid: str, scope_signature: str) -> bool:
    react_session_key = _session_key(project_uid, WORKFLOW_REACT)
    a2a_session_key = _session_key(project_uid, "a2a")
    react_sessions = st.session_state.get("paper_agent_sessions", {})
    multi_sessions = st.session_state.get("paper_multi_agent_sessions", {})
    signatures = st.session_state.get("paper_project_scope_signatures", {})
    current_signature = signatures.get(project_uid)
    return (
        react_session_key in react_sessions and a2a_session_key in multi_sessions
        and current_signature == scope_signature
    )


def _with_language_hint(prompt: str) -> str:
    detected = detect_language(prompt)
    if detected == "en":
        return f"{prompt}\n\n[Response language requirement: answer in English.]"
    if detected == "zh":
        return f"{prompt}\n\n[回答语言要求：请使用中文回答。]"
    return prompt


def _load_document_text(selected_uid: str, file_path: str) -> tuple[str | None, str]:
    document_text_cache = st.session_state.get("document_text_cache", {})
    document_text = document_text_cache.get(selected_uid)
    if isinstance(document_text, str):
        logger.info("Document text cache hit")
        return document_text, "session_hit"

    logger.info("Document text session cache miss: uid=%s", selected_uid)
    persisted_text: str | None = None
    try:
        cached_from_db = get_content_by_uid(selected_uid, "file_extraction")
        if isinstance(cached_from_db, str) and cached_from_db.strip():
            persisted_text = cached_from_db
            logger.info(
                "Document text restored from DB cache: uid=%s text_len=%s",
                selected_uid,
                len(persisted_text),
            )
    except Exception as exc:
        logger.warning("Failed to load persisted document extraction: %s", exc)

    if persisted_text is None:
        logger.info("Document text DB cache miss, extracting: path=%s", file_path)
        with st.spinner("正在解析文档内容..."):
            document_result = extract_files(file_path)
        if document_result["result"] != 1:
            logger.error("Document extraction failed: reason=%s", document_result["text"])
            st.error("文档加载失败：" + str(document_result["text"]))
            return None, "error"
        document_text = document_result["text"]
        if not isinstance(document_text, str):
            logger.error("Document extraction returned non-string text")
            st.error("文档内容解析失败，无法建立 RAG 索引。")
            return None, "error"
        try:
            save_content_to_database(
                uid=selected_uid,
                file_path=file_path,
                content=document_text,
                content_type="file_extraction",
            )
            logger.info(
                "Document extraction persisted to DB cache: uid=%s text_len=%s",
                selected_uid,
                len(document_text),
            )
        except Exception as exc:
            logger.warning("Failed to persist document extraction: %s", exc)
        source = "extracted"
    else:
        document_text = persisted_text
        source = "db_restore"

    document_text_cache[selected_uid] = document_text
    st.session_state.document_text_cache = document_text_cache
    logger.info("Document text cached in session: text_len=%s", len(document_text))
    return document_text, source


def _scope_signature(scope_docs: list[dict]) -> str:
    return ",".join(sorted(str(item.get("uid")) for item in scope_docs if item.get("uid")))


def _clear_project_runtime(project_uid: str) -> None:
    react_sessions = st.session_state.get("paper_agent_sessions", {})
    multi_sessions = st.session_state.get("paper_multi_agent_sessions", {})
    retrievers = st.session_state.get("paper_evidence_retrievers", {})
    for mode in (WORKFLOW_REACT, "a2a"):
        key = _session_key(project_uid, mode)
        if mode == WORKFLOW_REACT:
            react_sessions.pop(key, None)
        else:
            multi_sessions.pop(key, None)
    retrievers.pop(project_uid, None)
    st.session_state.paper_agent_sessions = react_sessions
    st.session_state.paper_multi_agent_sessions = multi_sessions
    st.session_state.paper_evidence_retrievers = retrievers


def _ensure_agent(
    project_uid: str,
    project_name: str,
    scope_docs: list[dict],
    scope_signature: str,
) -> None:
    logger.info(
        "Ensuring project sessions: project=%s docs=%s",
        project_name,
        len(scope_docs),
    )
    react_session_key = _session_key(project_uid, WORKFLOW_REACT)
    a2a_session_key = _session_key(project_uid, "a2a")
    react_sessions = st.session_state.get("paper_agent_sessions", {})
    multi_sessions = st.session_state.get("paper_multi_agent_sessions", {})
    evidence_retrievers = st.session_state.get("paper_evidence_retrievers", {})
    signatures = st.session_state.get("paper_project_scope_signatures", {})

    current_react_session = react_sessions.get(react_session_key)
    current_multi_session = multi_sessions.get(a2a_session_key)
    current_evidence_retriever = evidence_retrievers.get(project_uid)
    current_signature = signatures.get(project_uid)

    if current_signature != scope_signature and current_signature is not None:
        _clear_project_runtime(project_uid)
        current_react_session = None
        current_multi_session = None
        current_evidence_retriever = None
        messages_map = st.session_state.get("paper_project_messages", {})
        messages_map.pop(project_uid, None)
        st.session_state.paper_project_messages = messages_map
        summary_map = st.session_state.get("paper_project_compact_summaries", {})
        summary_map.pop(project_uid, None)
        st.session_state.paper_project_compact_summaries = summary_map
        tool_specs_map = st.session_state.get("paper_project_tool_specs", {})
        tool_specs_map.pop(project_uid, None)
        st.session_state.paper_project_tool_specs = tool_specs_map
        skill_texts_map = st.session_state.get("paper_project_skill_context_texts", {})
        skill_texts_map.pop(project_uid, None)
        st.session_state.paper_project_skill_context_texts = skill_texts_map

    if current_react_session and current_multi_session and current_evidence_retriever:
        logger.info("Reusing existing project sessions and evidence retriever")
        st.session_state.paper_agent = current_react_session["agent"]
        st.session_state.paper_agent_runtime_config = current_react_session["runtime_config"]
        st.session_state.paper_multi_agent = current_multi_session["coordinator"]
        st.session_state.paper_evidence_retriever = current_evidence_retriever
        st.session_state.agent_current_project = project_name
        tool_specs_map = st.session_state.get("paper_project_tool_specs", {})
        reused_tool_specs = current_react_session.get("tool_specs", [])
        if isinstance(reused_tool_specs, list):
            tool_specs_map[project_uid] = reused_tool_specs
            st.session_state.paper_project_tool_specs = tool_specs_map
        messages = st.session_state.paper_project_messages.get(project_uid, [])
        st.session_state.agent_messages = messages
        return

    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在“设置中心”页面配置您的 API Key")
    user_model = get_user_model_name()
    if not user_model:
        raise ValueError("请先在“设置中心”页面配置模型名称")
    user_base_url = get_user_base_url()
    llm = build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=user_model,
        base_url=user_base_url,
    )
    logger.info("Built chat model for agent center: model=%s", user_model)

    search_document_evidence_fn = current_evidence_retriever
    if search_document_evidence_fn is None:
        logger.info("Building project evidence retriever (cache miss)")
        documents = [
            {
                "doc_uid": str(item["uid"]),
                "doc_name": str(item["file_name"]),
                "text": str(item["text"]),
            }
            for item in scope_docs
            if isinstance(item.get("text"), str)
        ]
        if len(documents) == 1:
            only = documents[0]
            search_document_evidence_fn = build_local_evidence_retriever_with_settings(
                document_text=only["text"],
                doc_uid=only["doc_uid"],
                doc_name=only["doc_name"],
                project_uid=project_uid,
            )
        else:
            search_document_evidence_fn = build_project_evidence_retriever_with_settings(
                documents=documents,
                project_uid=project_uid,
            )
        evidence_retrievers[project_uid] = search_document_evidence_fn
        st.session_state.paper_evidence_retrievers = evidence_retrievers

    def search_document_fn(query: str) -> str:
        payload = search_document_evidence_fn(query)
        evidence_items = _normalize_evidence_items(payload)
        return "\n".join(item.get("text", "") for item in evidence_items)

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
    context_hint = f"当前项目：{project_name}\n当前检索范围：{scope_preview or '空'}"

    if not current_react_session:
        logger.info("Creating new ReAct project session")
        agent_session = create_paper_agent_session(
            llm=llm,
            search_document_fn=search_document_fn,
            search_document_evidence_fn=search_document_evidence_fn,
            read_document_fn=read_document_fn,
            document_name=scope_preview or "项目范围",
            project_name=project_name,
            scope_summary=scope_preview or "空范围",
        )
        react_sessions[react_session_key] = {
            "agent": agent_session.agent,
            "runtime_config": agent_session.runtime_config,
            "tool_specs": agent_session.tool_specs,
        }
        st.session_state.paper_agent_sessions = react_sessions
        tool_specs_map = st.session_state.get("paper_project_tool_specs", {})
        tool_specs_map[project_uid] = agent_session.tool_specs
        st.session_state.paper_project_tool_specs = tool_specs_map

    if not current_multi_session:
        logger.info("Creating new A2A multi-agent project session")
        a2a_session = create_multi_agent_a2a_session(
            llm=llm,
            search_document_fn=search_document_fn,
            search_document_evidence_fn=search_document_evidence_fn,
            context_hint=context_hint,
        )
        multi_sessions[a2a_session_key] = {"coordinator": a2a_session.coordinator}
        st.session_state.paper_multi_agent_sessions = multi_sessions

    signatures[project_uid] = scope_signature
    st.session_state.paper_project_scope_signatures = signatures
    st.session_state.paper_agent = st.session_state.paper_agent_sessions[react_session_key]["agent"]
    st.session_state.paper_agent_runtime_config = st.session_state.paper_agent_sessions[
        react_session_key
    ]["runtime_config"]
    st.session_state.paper_multi_agent = st.session_state.paper_multi_agent_sessions[
        a2a_session_key
    ]["coordinator"]
    st.session_state.paper_evidence_retriever = search_document_evidence_fn
    st.session_state.agent_current_project = project_name
    messages = st.session_state.paper_project_messages.get(project_uid)
    if not messages:
        messages = [
            {
                "role": "assistant",
                "content": (
                    f"已加载项目《{project_name}》，当前检索范围 {len(scope_docs)} 篇文档。"
                    " 工作流将按问题自动路由。"
                ),
            }
        ]
        st.session_state.paper_project_messages[project_uid] = messages
    st.session_state.agent_messages = messages
    logger.info("Project sessions ready: bootstrap_messages=%s", len(messages))


def _prepare_agent_session(
    project_uid: str,
    project_name: str,
    scope_docs: list[dict],
    scope_signature: str,
) -> None:
    has_cached_session = _has_cached_agent_session(project_uid, scope_signature)
    logger.info("Agent session cache status: has_cached=%s", has_cached_session)
    if has_cached_session:
        _ensure_agent(project_uid, project_name, scope_docs, scope_signature)
        st.caption("项目级 RAG 索引已存在，已复用。")
        return

    with st.spinner("正在构建项目级 RAG 索引（首次会自动下载模型）..."):
        _ensure_agent(project_uid, project_name, scope_docs, scope_signature)


def _update_context_usage(project_uid: str) -> None:
    compact_summaries = st.session_state.get("paper_project_compact_summaries", {})
    tool_specs_map = st.session_state.get("paper_project_tool_specs", {})
    tool_specs = tool_specs_map.get(project_uid, [])
    skill_texts_map = st.session_state.get("paper_project_skill_context_texts", {})
    skill_context_texts = skill_texts_map.get(project_uid, [])
    usage_map = st.session_state.get("paper_project_context_usage", {})
    usage_map[project_uid] = build_context_usage_snapshot(
        messages=st.session_state.get("agent_messages", []),
        compact_summary=str(compact_summaries.get(project_uid, "") or ""),
        tool_specs=tool_specs if isinstance(tool_specs, list) else [],
        skill_context_texts=(
            skill_context_texts if isinstance(skill_context_texts, list) else []
        ),
    )
    st.session_state.paper_project_context_usage = usage_map


def _apply_auto_compact(project_uid: str) -> str:
    summary_map = st.session_state.get("paper_project_compact_summaries", {})
    current_summary = str(summary_map.get(project_uid, "") or "")
    llm = None
    messages = st.session_state.get("agent_messages", [])
    if should_trigger_auto_compact(messages):
        llm = st.session_state.get("paper_compactor_llm")
        if llm is None:
            api_key = get_user_api_key()
            model_name = get_user_model_name()
            base_url = get_user_base_url()
            if api_key and model_name:
                llm = build_openai_compatible_chat_model(
                    api_key=api_key,
                    model_name=model_name,
                    base_url=base_url,
                    temperature=0,
                )
                st.session_state.paper_compactor_llm = llm

    result = auto_compact_messages(
        messages,
        current_summary=current_summary,
        llm=llm,
    )
    st.session_state.agent_messages = result.messages
    st.session_state.paper_project_messages[project_uid] = result.messages
    summary_map[project_uid] = result.summary
    st.session_state.paper_project_compact_summaries = summary_map
    if result.compacted:
        logger.info(
            "Auto compact applied: project_uid=%s source_messages=%s tokens_before=%s tokens_after=%s llm=%s anchors=%s",
            project_uid,
            result.source_message_count,
            result.source_token_estimate,
            result.compacted_token_estimate,
            result.used_llm,
            result.anchor_count,
        )
        mode = "LLM" if result.used_llm else "Heuristic"
        st.caption(
            "已执行自动压缩："
            f"{result.source_token_estimate} -> {result.compacted_token_estimate} tokens"
            f" | mode={mode} | anchors={result.anchor_count}"
        )
    return result.summary


def main():
    user_uuid = st.session_state.get("uuid", "local-user")
    turn_in_progress = bool(st.session_state.get("agent_turn_in_progress", False))
    pending_turn = st.session_state.get("agent_pending_turn")
    api_key = get_user_api_key()
    if not api_key:
        st.warning("⚠️ 请先在“设置中心”页面配置您的 API Key")
        st.info('💡 请前往页面“设置中心（2_⚙️_设置中心）”完成配置后刷新。')
        logger.warning("Agent center blocked: missing API key")
        return

    user_model = get_user_model_name()
    if not user_model:
        st.warning("⚠️ 请先在“设置中心”页面配置模型名称")
        st.info('💡 请前往页面“设置中心（2_⚙️_设置中心）”完成配置后刷新。')
        logger.warning("Agent center blocked: missing model name")
        return

    inject_workspace_styles()
    _load_projects_from_db()
    projects = st.session_state.get("projects", [])
    project_doc_count_map = build_project_doc_count_map(projects, user_uuid)
    with st.sidebar:
        st.markdown("## 项目工作台")
        st.caption("模型/API 配置统一在“设置中心”管理。")
        selected_project = select_project_sidebar(
            projects,
            project_doc_count_map,
            disabled=turn_in_progress,
        )
    if selected_project is None:
        st.write("### 暂无项目，请前往“项目中心”创建。")
        return
    selected_project_uid = str(selected_project["project_uid"])
    selected_project_name = str(selected_project["project_name"])

    scoped_files = list_project_files(
        project_uid=selected_project_uid,
        uuid=st.session_state["uuid"],
        active_only=True,
    )
    with st.sidebar:
        scope_docs = select_scope_documents_drawer(
            scoped_files,
            selected_project_uid,
            disabled=turn_in_progress,
        )
        render_workspace_status_bar(
            project_name=selected_project_name,
            total_docs=project_doc_count_map.get(selected_project_uid, len(scoped_files)),
            selected_docs=len(scope_docs),
            turn_in_progress=turn_in_progress,
        )
    if not scope_docs:
        st.warning("当前项目还没有激活文档，请在文件中心或项目中心绑定文档。")
        return

    with logging_context(uid=user_uuid, project_uid=selected_project_uid):
        logger.info(
            "Selected project: name=%s docs=%s",
            selected_project_name,
            len(scope_docs),
        )

    scope_docs_with_text: list[dict] = []
    cache_stats = {"session_hit": 0, "db_restore": 0, "extracted": 0}
    with logging_context(uid=user_uuid, project_uid=selected_project_uid):
        for scope_doc in scope_docs:
            scope_uid = str(scope_doc["uid"])
            text, source = _load_document_text(scope_uid, scope_doc["file_path"])
            if text is None:
                return
            if source in cache_stats:
                cache_stats[source] += 1
            enriched = dict(scope_doc)
            enriched["text"] = text
            scope_docs_with_text.append(enriched)

        if cache_stats["db_restore"] > 0:
            st.caption(f"文档内容已从数据库缓存恢复：{cache_stats['db_restore']} 篇。")
        elif cache_stats["session_hit"] > 0:
            st.caption(f"文档内容已命中会话缓存：{cache_stats['session_hit']} 篇。")

        scope_signature = _scope_signature(scope_docs_with_text)
        _prepare_agent_session(
            selected_project_uid,
            selected_project_name,
            scope_docs_with_text,
            scope_signature,
        )
        _update_context_usage(selected_project_uid)

    with st.sidebar:
        st.markdown("### 会话信息")
        _render_output_archive(selected_project_uid, disable_interaction=turn_in_progress)
        _render_workflow_metrics(selected_project_uid)
        _render_context_usage(selected_project_uid)
        if turn_in_progress:
            st.info("正在生成回答，已临时锁定归档与文档切换，避免中断当前对话。")

    chat_messages = st.session_state.get("agent_messages", [])
    render_project_context_hint(selected_project_name, scope_docs)
    _render_chat_history(chat_messages)

    prompt = st.chat_input("输入你的论文问题", disabled=turn_in_progress)
    if (
        turn_in_progress
        and isinstance(pending_turn, dict)
        and pending_turn.get("project_uid") == selected_project_uid
        and pending_turn.get("scope_signature") == scope_signature
        and isinstance(pending_turn.get("prompt"), str)
    ):
        prompt = pending_turn["prompt"]
    elif turn_in_progress and isinstance(pending_turn, dict):
        # 文档切换或状态异常导致 pending 与当前文档不一致，清理锁状态
        st.session_state.agent_turn_in_progress = False
        st.session_state.agent_pending_turn = None
        st.rerun()
        return
    elif not prompt:
        return
    else:
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        st.session_state.agent_pending_turn = {
            "prompt": prompt,
            "project_uid": selected_project_uid,
            "scope_signature": scope_signature,
        }
        st.session_state.agent_turn_in_progress = True
        st.rerun()
        return

    compact_summary = _apply_auto_compact(selected_project_uid)
    hinted_prompt = inject_compact_summary(_with_language_hint(prompt), compact_summary)
    run_id = f"run-{uuid4().hex[:12]}"
    coordinator = st.session_state.get("paper_multi_agent")
    session_id = getattr(coordinator, "session_id", "-")
    selected_doc_uid_for_logging = (
        str(scope_docs_with_text[0]["uid"]) if scope_docs_with_text else ""
    )
    try:
        turn_result = execute_assistant_turn(
            prompt=prompt,
            hinted_prompt=hinted_prompt,
            user_uuid=user_uuid,
            project_uid=selected_project_uid,
            selected_uid=selected_doc_uid_for_logging,
            run_id=run_id,
            coordinator=coordinator,
            session_id=session_id,
            logger=logger,
        )
    finally:
        st.session_state.agent_turn_in_progress = False
        st.session_state.agent_pending_turn = None
    answer = turn_result["answer"]
    workflow_mode = turn_result["workflow_mode"]
    workflow_reason = turn_result["workflow_reason"]
    trace_payload = turn_result["trace_payload"]
    evidence_items = turn_result["evidence_items"]
    mindmap_data = turn_result["mindmap_data"]
    method_compare_data = turn_result["method_compare_data"]
    run_latency_ms = turn_result["run_latency_ms"]
    replan_rounds = turn_result["replan_rounds"]
    phase_path = turn_result["phase_path"]
    traced_skill_texts = extract_skill_context_texts_from_trace(trace_payload)
    if traced_skill_texts:
        skill_texts_map = st.session_state.get("paper_project_skill_context_texts", {})
        existing_texts = skill_texts_map.get(selected_project_uid, [])
        if not isinstance(existing_texts, list):
            existing_texts = []
        seen_texts = {str(item) for item in existing_texts}
        for text in traced_skill_texts:
            value = str(text)
            if value in seen_texts:
                continue
            existing_texts.append(value)
            seen_texts.add(value)
        skill_texts_map[selected_project_uid] = existing_texts[-20:]
        st.session_state.paper_project_skill_context_texts = skill_texts_map

    doc_metrics = _get_doc_metrics(selected_project_uid)
    updated_metrics = record_query_metrics(
        doc_metrics,
        workflow_mode=workflow_mode,
        latency_ms=run_latency_ms,
        trace_payload=trace_payload,
    )
    paper_project_metrics = st.session_state.get("paper_project_metrics", {})
    paper_project_metrics[selected_project_uid] = updated_metrics
    st.session_state.paper_project_metrics = paper_project_metrics

    st.session_state.agent_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "acp_trace": trace_payload,
            "mindmap_data": mindmap_data,
            "method_compare_data": method_compare_data,
            "evidence_items": evidence_items,
            "workflow_mode": workflow_mode,
            "workflow_reason": workflow_reason,
            "latency_ms": run_latency_ms,
            "replan_rounds": replan_rounds,
            "phase_path": phase_path,
        }
    )
    output_type = _infer_output_type(prompt, mindmap_data)
    serialized_content = (
        json.dumps(mindmap_data, ensure_ascii=False) if mindmap_data else answer
    )
    save_agent_output(
        uuid=st.session_state.get("uuid", "local-user"),
        project_uid=selected_project_uid,
        doc_uid=(
            str(scope_docs_with_text[0]["uid"])
            if len(scope_docs_with_text) == 1
            else None
        ),
        doc_name=(
            str(scope_docs_with_text[0]["file_name"])
            if len(scope_docs_with_text) == 1
            else selected_project_name
        ),
        output_type=output_type,
        content=serialized_content,
    )
    logger.info("Archived agent output: output_type=%s content_len=%s", output_type, len(serialized_content))

    st.session_state.paper_project_messages[selected_project_uid] = (
        st.session_state.agent_messages
    )
    _update_context_usage(selected_project_uid)
    st.rerun()


initialize_agent_center_session_state()

_load_files_from_db()

if not st.session_state.files:
    st.write("### 暂无文档，请前往“文件中心”页面上传。")
else:
    main()
