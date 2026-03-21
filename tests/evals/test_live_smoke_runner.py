import importlib.util
from pathlib import Path

from agent.application.evals import AgentEvalCase

MODULE_PATH = Path('/home/ling/LLM_App_Final/.worktrees/research-langchain-evals/tests/evals/run_agent_task_completion_live_smoke.py')
spec = importlib.util.spec_from_file_location('live_smoke_runner', MODULE_PATH)
live_smoke_runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(live_smoke_runner)


def test_resolve_case_documents_disables_project_docs_for_document_free_case() -> None:
    case = AgentEvalCase.from_dict(
        {
            'id': 'web_only_001',
            'category': 'web_research',
            'prompt': '请联网检索最新进展',
            'success_rubric': 'Answer should rely on web research only.',
            'document_access': 'none',
        }
    )
    documents = [
        {'doc_uid': 'arxiv:2005.11401', 'doc_name': 'RAG', 'text': 'rag'},
        {'doc_uid': 'arxiv:2310.11511', 'doc_name': 'Self-RAG', 'text': 'self-rag'},
    ]

    config = live_smoke_runner.build_case_document_context(case, documents)

    assert config['document_access'] == 'none'
    assert config['documents'] == []
    assert config['search_document_evidence_fn'] is None
    assert '项目文档' in config['scope_summary']
    assert '不提供' in config['scope_summary']



def test_resolve_case_documents_scopes_to_declared_doc_uids() -> None:
    case = AgentEvalCase.from_dict(
        {
            'id': 'project_scope_001',
            'category': 'project_research',
            'prompt': '只基于项目文档回答',
            'success_rubric': 'Answer should stay within scoped project docs.',
            'document_access': 'scoped',
            'document_scope': ['arxiv:2310.11511'],
        }
    )
    documents = [
        {'doc_uid': 'arxiv:2005.11401', 'doc_name': 'RAG', 'text': 'rag'},
        {'doc_uid': 'arxiv:2310.11511', 'doc_name': 'Self-RAG', 'text': 'self-rag'},
    ]

    config = live_smoke_runner.build_case_document_context(case, documents)

    assert config['document_access'] == 'scoped'
    assert [item['doc_uid'] for item in config['documents']] == ['arxiv:2310.11511']
    assert config['search_document_evidence_fn'] is not None
    assert config['search_document_fn'] is not None
    payload = config['search_document_evidence_fn']('self-rag')
    evidences = payload.get('evidences', []) if isinstance(payload, dict) else []
    assert evidences
    assert config['search_document_fn']('self-rag') == 'self-rag'
