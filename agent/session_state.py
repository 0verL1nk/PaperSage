import streamlit as st


def initialize_agent_center_session_state() -> None:
    defaults = {
        "paper_agent": None,
        "paper_multi_agent": None,
        "paper_evidence_retriever": None,
        "paper_agent_runtime_config": None,
        "paper_compactor_llm": None,
        "agent_turn_in_progress": False,
        "agent_pending_turn": None,
        "agent_messages": [],
        "agent_current_project": None,
        "paper_project_scope_signatures": {},
        "paper_agent_sessions": {},
        "paper_multi_agent_sessions": {},
        "paper_evidence_retrievers": {},
        "paper_project_messages": {},
        "paper_project_metrics": {},
        "paper_project_compact_summaries": {},
        "paper_project_context_usage": {},
        "paper_project_tool_specs": {},
        "paper_project_skill_context_texts": {},
        "document_text_cache": {},
        "files": [],
        "projects": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
