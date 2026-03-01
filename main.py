import json
import logging
import os
import streamlit as st
import streamlit.components.v1 as components
from pyecharts import options as opts
from pyecharts.charts import Tree

# 配置 debug 日志
DEBUG_MODE = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}
if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("./debug.log", mode="a", encoding="utf-8"),
        ],
    )
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

from utils import extract_files, show_sidebar_api_key_setting
from agent.archive import list_agent_outputs, save_agent_output
from agent.output_cleaner import sanitize_public_answer
from agent.stream import iter_agent_response_deltas
from agent.llm_provider import build_openai_compatible_chat_model
from agent.rag_hybrid import build_local_vector_retriever_with_settings
from agent.multi_agent_a2a import (
    WORKFLOW_PLAN_ACT,
    WORKFLOW_PLAN_ACT_REPLAN,
    WORKFLOW_REACT,
    create_multi_agent_a2a_session,
)
from agent.paper_agent import create_paper_agent_session
from agent.workflow_router import WORKFLOW_LABELS, auto_select_workflow_mode
from utils.utils import extract_json_string
from utils.utils import (
    detect_language,
    ensure_local_user,
    get_user_api_key,
    get_user_base_url,
    get_user_files,
    get_user_model_name,
    init_database,
)


st.set_page_config(page_title="Agent 中心", page_icon="🤖")
st.title("🤖 Agent 中心")
init_database("./database.sqlite")
if "uuid" not in st.session_state or not st.session_state["uuid"]:
    st.session_state["uuid"] = "local-user"
ensure_local_user(st.session_state["uuid"], db_name="./database.sqlite")
show_sidebar_api_key_setting()

if "files" not in st.session_state:
    st.session_state["files"] = []


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


def _selected_document():
    document_names = [file_item["file_name"] for file_item in st.session_state["files"]]
    selected_name = st.selectbox("选择文档", document_names, key="agent_doc_selector")
    selected_index = document_names.index(selected_name)
    return st.session_state["files"][selected_index]


def _session_key(document_uid: str, mode: str) -> str:
    return f"{mode}:{document_uid}"


def _has_cached_agent_session(document_uid: str) -> bool:
    react_session_key = _session_key(document_uid, WORKFLOW_REACT)
    a2a_session_key = _session_key(document_uid, "a2a")
    react_sessions = st.session_state.get("paper_agent_sessions", {})
    multi_sessions = st.session_state.get("paper_multi_agent_sessions", {})
    return (
        react_session_key in react_sessions and a2a_session_key in multi_sessions
    )


def _render_acp_trace(trace_payload: list[dict[str, str]]) -> None:
    with st.expander("查看 A2A 协调轨迹", expanded=False):
        for idx, entry in enumerate(trace_payload, start=1):
            sender = entry.get("sender", "unknown")
            receiver = entry.get("receiver", "unknown")
            perf = entry.get("performative", "message")
            content = entry.get("content", "")
            st.markdown(f"{idx}. `{sender} -> {receiver}` | `{perf}`\n\n{content}")


def _preview_text(text: str, limit: int = 140) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def _with_language_hint(prompt: str) -> str:
    detected = detect_language(prompt)
    if detected == "en":
        return f"{prompt}\n\n[Response language requirement: answer in English.]"
    if detected == "zh":
        return f"{prompt}\n\n[回答语言要求：请使用中文回答。]"
    return prompt


def _render_text_stream(stream) -> str:
    container = st.empty()
    chunks: list[str] = []
    for delta in stream:
        if not isinstance(delta, str) or not delta:
            continue
        chunks.append(delta)
        container.markdown("".join(chunks))
    return "".join(chunks)


def _create_mindmap_chart(data: dict) -> Tree:
    return (
        Tree()
        .add(
            series_name="",
            data=[data],
            orient="LR",
            initial_tree_depth=3,
            layout="orthogonal",
            pos_left="3%",
            width="75%",
            height="420px",
            edge_fork_position="12%",
            symbol_size=7,
            label_opts=opts.LabelOpts(
                position="right",
                horizontal_align="left",
                vertical_align="middle",
                font_size=13,
                padding=[0, 0, 0, -18],
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="思维导图"),
            tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove"),
        )
    )


def _try_parse_mindmap(answer: str) -> dict | None:
    if not answer:
        return None
    try:
        json_str = extract_json_string(answer)
        parsed = json.loads(json_str)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    if "name" not in parsed:
        return None
    children = parsed.get("children")
    if children is not None and not isinstance(children, list):
        return None
    return parsed


def _render_mindmap_if_any(mindmap_data: dict | None) -> None:
    if not mindmap_data:
        return
    try:
        chart = _create_mindmap_chart(mindmap_data)
        components.html(chart.render_embed(), height=460, scrolling=True)
    except Exception:
        st.warning("思维导图 JSON 已识别，但渲染失败。")


def _infer_output_type(prompt: str, mindmap_data: dict | None) -> str:
    if mindmap_data:
        return "mindmap"
    normalized = prompt.lower()
    if "总结" in prompt or "summary" in normalized:
        return "summary"
    if "改写" in prompt or "润色" in prompt or "rewrite" in normalized:
        return "rewrite"
    return "qa"


def _render_output_archive(doc_uid: str) -> None:
    user_uuid = st.session_state.get("uuid", "local-user")
    records = list_agent_outputs(uuid=user_uuid, doc_uid=doc_uid, limit=20)
    with st.expander("查看历史产出归档", expanded=False):
        if not records:
            st.caption("暂无归档记录")
            return

        options = [
            f"{item['created_at']} | {item['output_type']} | {item.get('doc_name') or 'unknown'}"
            for item in records
        ]
        selected_label = st.selectbox("选择归档记录", options, key=f"archive_{doc_uid}")
        selected_index = options.index(selected_label)
        selected = records[selected_index]
        content = selected["content"]
        if selected["output_type"] == "mindmap":
            try:
                _render_mindmap_if_any(json.loads(content))
            except Exception:
                st.write(content)
        else:
            st.write(content)


def _ensure_agent(document_uid: str, document_name: str, document_text: str):
    react_session_key = _session_key(document_uid, WORKFLOW_REACT)
    a2a_session_key = _session_key(document_uid, "a2a")
    react_sessions = st.session_state.get("paper_agent_sessions", {})
    multi_sessions = st.session_state.get("paper_multi_agent_sessions", {})
    current_react_session = react_sessions.get(react_session_key)
    current_multi_session = multi_sessions.get(a2a_session_key)

    if current_react_session and current_multi_session:
        st.session_state.paper_agent = current_react_session["agent"]
        st.session_state.paper_agent_runtime_config = current_react_session[
            "runtime_config"
        ]
        st.session_state.paper_multi_agent = current_multi_session["coordinator"]
        st.session_state.agent_current_doc = document_name
        doc_messages = st.session_state.paper_doc_messages.get(document_uid, [])
        st.session_state.agent_messages = doc_messages
        return

    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在侧边栏设置中配置您的 API Key")

    user_model = get_user_model_name()
    if not user_model:
        raise ValueError("请先在侧边栏设置中配置模型名称")
    user_base_url = get_user_base_url()
    llm = build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=user_model,
        base_url=user_base_url,
    )
    search_document_fn = build_local_vector_retriever_with_settings(document_text)

    # 创建分块读取函数
    def read_document_fn(offset: int = 0, limit: int = 2000) -> tuple[str, int]:
        total_len = len(document_text)
        content = document_text[offset : offset + limit]
        return content, total_len

    if not current_react_session:
        agent_session = create_paper_agent_session(
            llm=llm,
            search_document_fn=search_document_fn,
            read_document_fn=read_document_fn,
            document_name=document_name,
        )
        react_sessions[react_session_key] = {
            "agent": agent_session.agent,
            "runtime_config": agent_session.runtime_config,
        }
        st.session_state.paper_agent_sessions = react_sessions

    if not current_multi_session:
        a2a_session = create_multi_agent_a2a_session(
            llm=llm,
            search_document_fn=search_document_fn,
        )
        multi_sessions[a2a_session_key] = {
            "coordinator": a2a_session.coordinator,
        }
        st.session_state.paper_multi_agent_sessions = multi_sessions

    st.session_state.paper_agent = st.session_state.paper_agent_sessions[react_session_key][
        "agent"
    ]
    st.session_state.paper_agent_runtime_config = st.session_state.paper_agent_sessions[
        react_session_key
    ]["runtime_config"]
    st.session_state.paper_multi_agent = st.session_state.paper_multi_agent_sessions[
        a2a_session_key
    ]["coordinator"]
    st.session_state.agent_current_doc = document_name
    messages = st.session_state.paper_doc_messages.get(document_uid)
    if not messages:
        messages = [
            {
                "role": "assistant",
                "content": f"已加载文档《{document_name}》。工作流将按问题自动路由。",
            }
        ]
        st.session_state.paper_doc_messages[document_uid] = messages
    st.session_state.agent_messages = messages


def main():
    api_key = get_user_api_key()
    if not api_key:
        st.warning("⚠️ 请先在侧边栏设置中配置您的 API Key")
        st.info('💡 请在左侧边栏的"设置"中完成配置后刷新页面。')
        return

    user_model = get_user_model_name()
    if not user_model:
        st.warning("⚠️ 请先在侧边栏设置中配置模型名称")
        st.info('💡 请在左侧边栏的"设置"中完成配置后刷新页面。')
        return

    selected_doc = _selected_document()
    selected_uid = selected_doc["uid"]
    selected_name = selected_doc["file_name"]

    document_text_cache = st.session_state.get("document_text_cache", {})
    document_text = document_text_cache.get(selected_uid)
    if not isinstance(document_text, str):
        with st.spinner("正在解析文档内容..."):
            document_result = extract_files(selected_doc["file_path"])
        if document_result["result"] != 1:
            st.error("文档加载失败：" + str(document_result["text"]))
            return
        document_text = document_result["text"]
        if not isinstance(document_text, str):
            st.error("文档内容解析失败，无法建立 RAG 索引。")
            return
        document_text_cache[selected_uid] = document_text
        st.session_state.document_text_cache = document_text_cache
    else:
        st.caption("文档内容已命中缓存。")

    has_cached_session = _has_cached_agent_session(selected_uid)
    if has_cached_session:
        _ensure_agent(selected_uid, selected_name, document_text)
        st.caption("RAG 索引已存在，已复用。")
    else:
        with st.spinner("正在构建本地 RAG 索引（首次会自动下载模型）..."):
            _ensure_agent(selected_uid, selected_name, document_text)
    _render_output_archive(selected_doc["uid"])

    chat_messages = st.session_state.get("agent_messages", [])
    for message in chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            workflow_mode = message.get("workflow_mode")
            if isinstance(workflow_mode, str):
                label = WORKFLOW_LABELS.get(workflow_mode, workflow_mode)
                reason = message.get("workflow_reason") or ""
                st.caption(f"自动路由：{label} {reason}")
            trace_payload = message.get("acp_trace")
            if isinstance(trace_payload, list) and trace_payload:
                _render_acp_trace(trace_payload)
            _render_mindmap_if_any(message.get("mindmap_data"))

    prompt = st.chat_input("输入你的论文问题")
    if not prompt:
        return

    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    hinted_prompt = _with_language_hint(prompt)

    trace_payload: list[dict[str, str]] = []
    with st.chat_message("assistant"):
        workflow_mode, workflow_reason = auto_select_workflow_mode(
            prompt,
            coordinator=st.session_state.paper_multi_agent,
        )
        st.caption(f"自动路由：{WORKFLOW_LABELS.get(workflow_mode, workflow_mode)}")
        if workflow_mode == WORKFLOW_REACT:
            answer_raw = _render_text_stream(
                iter_agent_response_deltas(
                    st.session_state.paper_agent,
                    [{"role": "user", "content": hinted_prompt}],
                    config=st.session_state.paper_agent_runtime_config,
                )
            )
            answer = answer_raw if isinstance(answer_raw, str) else ""
            answer = sanitize_public_answer(answer)
            if not answer:
                answer = "抱歉，我暂时没有生成有效回复。"
                st.write(answer)
        else:
            event_logs: list[dict[str, str]] = []
            with st.status(
                f"{WORKFLOW_LABELS.get(workflow_mode, workflow_mode)} 执行中...",
                expanded=True,
            ) as status:
                def _on_event(item) -> None:
                    event_logs.append(
                        {
                            "sender": item.sender,
                            "receiver": item.receiver,
                            "performative": item.performative,
                            "content": item.content,
                        }
                    )
                    status.write(
                        f"{len(event_logs)}. "
                        f"`{item.sender} -> {item.receiver}` | `{item.performative}` | "
                        f"{_preview_text(item.content)}"
                    )

                answer, trace = st.session_state.paper_multi_agent.run(
                    hinted_prompt,
                    workflow_mode=workflow_mode,
                    on_event=_on_event,
                )
                status.update(label="A2A 协调完成", state="complete", expanded=False)
            answer = sanitize_public_answer(answer)
            if not answer:
                answer = "抱歉，我暂时没有生成有效回复。"
            st.write(answer)
            trace_payload = event_logs if event_logs else [
                {
                    "sender": item.sender,
                    "receiver": item.receiver,
                    "performative": item.performative,
                    "content": item.content,
                }
                for item in trace
            ]
            if trace_payload:
                _render_acp_trace(trace_payload)
        mindmap_data = _try_parse_mindmap(answer)
        _render_mindmap_if_any(mindmap_data)

    st.session_state.agent_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "acp_trace": trace_payload,
            "mindmap_data": mindmap_data,
            "workflow_mode": workflow_mode,
            "workflow_reason": workflow_reason,
        }
    )
    output_type = _infer_output_type(prompt, mindmap_data)
    serialized_content = (
        json.dumps(mindmap_data, ensure_ascii=False) if mindmap_data else answer
    )
    save_agent_output(
        uuid=st.session_state.get("uuid", "local-user"),
        doc_uid=selected_doc["uid"],
        doc_name=selected_doc["file_name"],
        output_type=output_type,
        content=serialized_content,
    )

    st.session_state.paper_doc_messages[selected_doc["uid"]] = (
        st.session_state.agent_messages
    )


if "paper_agent" not in st.session_state:
    st.session_state.paper_agent = None
if "paper_multi_agent" not in st.session_state:
    st.session_state.paper_multi_agent = None
if "paper_agent_runtime_config" not in st.session_state:
    st.session_state.paper_agent_runtime_config = None
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "agent_current_doc" not in st.session_state:
    st.session_state.agent_current_doc = None
if "paper_agent_sessions" not in st.session_state:
    st.session_state.paper_agent_sessions = {}
if "paper_multi_agent_sessions" not in st.session_state:
    st.session_state.paper_multi_agent_sessions = {}
if "paper_doc_messages" not in st.session_state:
    st.session_state.paper_doc_messages = {}
if "document_text_cache" not in st.session_state:
    st.session_state.document_text_cache = {}
if "files" not in st.session_state:
    st.session_state.files = []

_load_files_from_db()

if not st.session_state.files:
    st.write("### 暂无文档，请前往“文件中心”页面上传。")
else:
    main()
