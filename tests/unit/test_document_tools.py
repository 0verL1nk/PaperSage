import json

from agent.tools.document import build_search_document_tool


def test_search_document_tool_reuses_cached_evidence_payload_for_identical_query() -> None:
    calls = {"count": 0}

    def _search_evidence(query: str):
        calls["count"] += 1
        return {
            "evidences": [
                {"chunk_id": f"chunk-{calls['count']}", "text": f"hit:{query}", "page_no": 1}
            ]
        }

    tool = build_search_document_tool(
        search_document_fn=lambda query: f"fallback:{query}",
        search_document_evidence_fn=_search_evidence,
    )

    first = json.loads(tool.invoke({"query": "RAG"}))
    second = json.loads(tool.invoke({"query": "RAG"}))

    assert calls["count"] == 1
    assert first["evidences"][0]["chunk_id"] == "chunk-1"
    assert second["evidences"][0]["chunk_id"] == "chunk-1"
    assert second["meta"]["dedupe"]["reused_cached_result"] is True
    assert "refine query" in second["meta"]["dedupe"]["message"]



def test_search_document_tool_reuses_cached_text_result_for_identical_query() -> None:
    calls = {"count": 0}

    def _search_text(query: str) -> str:
        calls["count"] += 1
        return f"result:{query}:{calls['count']}"

    tool = build_search_document_tool(_search_text)

    first = tool.invoke({"query": "Self-RAG"})
    second = tool.invoke({"query": "Self-RAG"})
    third = tool.invoke({"query": "GraphRAG"})

    assert calls["count"] == 2
    assert first == "result:Self-RAG:1"
    assert second == "result:Self-RAG:1"
    assert third == "result:GraphRAG:2"


def test_search_document_tool_reuses_cached_evidence_payload_for_normalized_equivalent_query() -> None:
    calls = {"count": 0}

    def _search_evidence(query: str):
        calls["count"] += 1
        return {
            "evidences": [
                {"chunk_id": f"chunk-{calls['count']}", "text": f"hit:{query}", "page_no": 1}
            ]
        }

    tool = build_search_document_tool(
        search_document_fn=lambda query: f"fallback:{query}",
        search_document_evidence_fn=_search_evidence,
    )

    first = json.loads(tool.invoke({"query": "Self-RAG NQ 50.0"}))
    second = json.loads(tool.invoke({"query": "NQ Self-RAG 50"}))

    assert calls["count"] == 1
    assert second["evidences"][0]["chunk_id"] == "chunk-1"
    assert second["meta"]["dedupe"]["reused_cached_result"] is True
    assert second["meta"]["dedupe"]["query"] == "NQ Self-RAG 50"
    assert first["evidences"][0]["chunk_id"] == "chunk-1"


def test_search_document_tool_reuses_cached_text_result_for_normalized_equivalent_query() -> None:
    calls = {"count": 0}

    def _search_text(query: str) -> str:
        calls["count"] += 1
        return f"result:{query}:{calls['count']}"

    tool = build_search_document_tool(_search_text)

    first = tool.invoke({"query": "latency Self-RAG"})
    second = tool.invoke({"query": "Self-RAG latency"})

    assert calls["count"] == 1
    assert first == "result:latency Self-RAG:1"
    assert second == "result:latency Self-RAG:1"


def test_search_document_tool_reuses_cached_evidence_payload_for_same_query_family() -> None:
    calls = {"count": 0}

    def _search_evidence(query: str):
        calls["count"] += 1
        return {
            "evidences": [
                {"chunk_id": f"chunk-{calls['count']}", "text": f"hit:{query}", "page_no": 1}
            ]
        }

    tool = build_search_document_tool(
        search_document_fn=lambda query: f"fallback:{query}",
        search_document_evidence_fn=_search_evidence,
    )

    first = json.loads(tool.invoke({"query": "Self-RAG NQ score"}))
    second = json.loads(tool.invoke({"query": "Self-RAG NQ result"}))

    assert calls["count"] == 1
    assert first["evidences"][0]["chunk_id"] == "chunk-1"
    assert second["evidences"][0]["chunk_id"] == "chunk-1"
    assert second["meta"]["dedupe"]["reused_cached_result"] is True
    assert second["meta"]["dedupe"]["reason"] == "same_query_family"


def test_search_document_tool_reuses_cached_text_result_for_same_query_family() -> None:
    calls = {"count": 0}

    def _search_text(query: str) -> str:
        calls["count"] += 1
        return f"result:{query}:{calls['count']}"

    tool = build_search_document_tool(_search_text)

    first = tool.invoke({"query": "Self-RAG NQ TQA WQ results"})
    second = tool.invoke({"query": "Self-RAG NQ TQA WQ results table"})

    assert calls["count"] == 1
    assert first == "result:Self-RAG NQ TQA WQ results:1"
    assert second == "result:Self-RAG NQ TQA WQ results:1"
