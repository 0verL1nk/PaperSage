from types import SimpleNamespace

from agent.local_rag import _rerank_docs


class _FakeReranker:
    def __init__(self, ordered_texts):
        self._ordered_texts = ordered_texts

    def rerank(self, _request):
        return [{"text": text} for text in self._ordered_texts]


def test_rerank_docs_returns_reranked_order():
    docs = [
        SimpleNamespace(page_content="A"),
        SimpleNamespace(page_content="B"),
        SimpleNamespace(page_content="C"),
    ]
    reranker = _FakeReranker(["C", "A", "B"])

    output = _rerank_docs("q", docs, reranker=reranker, top_k=2)

    assert output == ["C", "A"]


def test_rerank_docs_fallback_when_no_reranker():
    docs = [
        SimpleNamespace(page_content="A"),
        SimpleNamespace(page_content="B"),
        SimpleNamespace(page_content="C"),
    ]

    output = _rerank_docs("q", docs, reranker=None, top_k=2)

    assert output == ["A", "B"]


def test_rerank_docs_fallback_on_rerank_error():
    class _BadReranker:
        def rerank(self, _request):
            raise RuntimeError("rerank failed")

    docs = [
        SimpleNamespace(page_content="A"),
        SimpleNamespace(page_content="B"),
    ]
    output = _rerank_docs("q", docs, reranker=_BadReranker(), top_k=1)
    assert output == ["A"]
