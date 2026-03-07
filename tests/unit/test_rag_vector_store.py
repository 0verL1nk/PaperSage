from __future__ import annotations

from typing import Any

import agent.rag.vector_store as vector_store


class _FakeEmbeddings:
    def _to_vector(self, text: str) -> list[float]:
        base = float(len(text))
        return [base, float(sum(ord(ch) for ch in text) % 997)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._to_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(text)


def test_stable_vectorstore_key_is_deterministic():
    payload: dict[str, Any] = {"a": 1, "b": "x"}
    key1 = vector_store.stable_vectorstore_key(payload)
    key2 = vector_store.stable_vectorstore_key({"b": "x", "a": 1})
    assert key1 == key2
    assert len(key1) == 40


def test_build_vectorstore_inmemory_backend(monkeypatch):
    monkeypatch.setenv("AGENT_VECTORSTORE_BACKEND", "inmemory")
    store, backend = vector_store.build_vectorstore(
        texts=["alpha", "beta"],
        embedding=_FakeEmbeddings(),
        collection_prefix="test",
        collection_key="case-inmemory",
    )
    assert backend == "inmemory"
    docs = store.similarity_search("alpha", k=1)
    assert docs
    assert "alpha" in docs[0].page_content


def test_build_vectorstore_fallback_when_chroma_unavailable(monkeypatch):
    monkeypatch.setenv("AGENT_VECTORSTORE_BACKEND", "chroma")
    monkeypatch.setattr(vector_store, "_LangchainChroma", None)
    store, backend = vector_store.build_vectorstore(
        texts=["alpha", "beta"],
        embedding=_FakeEmbeddings(),
        collection_prefix="test",
        collection_key="case-fallback",
    )
    assert backend == "inmemory"
    docs = store.similarity_search("beta", k=1)
    assert docs
    assert "beta" in docs[0].page_content
