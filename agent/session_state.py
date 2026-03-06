import streamlit as st


def initialize_agent_center_session_state() -> None:
    defaults = {
        "paper_agent": None,
        "paper_evidence_retriever": None,
        "paper_agent_runtime_config": None,
        "paper_leader_llm": None,
        "paper_policy_router_llm": None,
        "paper_search_document_fn": None,
        "paper_compactor_llm": None,
        "agent_turn_in_progress": False,
        "agent_pending_turn": None,
        "agent_messages": [],
        "agent_current_project": None,
        "agent_current_session_uid": None,
        "agent_project_selected_sessions": {},
        "paper_project_scope_signatures": {},
        "paper_agent_sessions": {},
        "paper_evidence_retrievers": {},
        "paper_project_llms": {},
        "paper_project_policy_llms": {},
        "paper_project_search_document_fns": {},
        "paper_project_messages": {},
        "paper_project_message_paging": {},
        "paper_project_metrics": {},
        "paper_project_compact_summaries": {},
        "paper_project_context_usage": {},
        "paper_project_tool_specs": {},
        "paper_project_skill_context_texts": {},
        "agent_last_conversation_key": "",
        "agent_history_keep_position": False,
        "document_text_cache": {},
        "files": [],
        "projects": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
