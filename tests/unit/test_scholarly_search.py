from agent import scholarly_search


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise scholarly_search.httpx.HTTPStatusError(
                "failed",
                request=None,  # type: ignore[arg-type]
                response=type("Resp", (), {"status_code": self.status_code})(),
            )

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, *_args, **_kwargs):
        return self._response


def test_search_semantic_scholar_parses_results(monkeypatch):
    payload = {
        "data": [
            {
                "title": "Paper 1",
                "authors": [{"name": "A1"}, {"name": "A2"}],
                "year": 2024,
                "venue": "Conf",
                "externalIds": {"DOI": "10.1000/demo"},
                "url": "https://example.org/p1",
                "isOpenAccess": True,
            }
        ]
    }
    monkeypatch.setattr(
        scholarly_search.httpx,
        "Client",
        lambda timeout: _FakeClient(_FakeResponse(payload)),
    )

    results = scholarly_search.search_semantic_scholar("query", limit=3)
    assert len(results) == 1
    assert results[0]["title"] == "Paper 1"
    assert results[0]["doi"] == "10.1000/demo"


def test_format_search_papers_results_empty():
    assert scholarly_search.format_search_papers_results([]) == "No academic papers found for this query."
