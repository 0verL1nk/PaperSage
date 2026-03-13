from agent.skills.loader import build_skill_runtime_payload


def test_build_skill_runtime_payload_loads_references_from_openviking(monkeypatch):
    calls = {"search": 0, "read": 0}

    class _FakeHit:
        def __init__(self, *, resource_id, metadata):
            self.resource_id = resource_id
            self.metadata = metadata

    class _FakeRead:
        def __init__(self, *, content):
            self.content = content

    class _FakeAdapter:
        def add_resource(self, *, project_uid, content, metadata=None):
            resolved_metadata = metadata or {}
            return f"{project_uid}:{resolved_metadata.get('path', '')}"

        def search(self, request):
            calls["search"] += 1
            return [
                _FakeHit(
                    resource_id="skills:agentic_search:references/output_schema.md",
                    metadata={
                        "namespace": "skills_reference",
                        "skill_name": "agentic_search",
                        "path": "references/output_schema.md",
                    },
                )
            ]

        def read(self, request):
            calls["read"] += 1
            return _FakeRead(content=f"content::{request.resource_id}")

    monkeypatch.setattr("agent.skills.loader.get_openviking_adapter", lambda: _FakeAdapter())

    payload = build_skill_runtime_payload("agentic_search", task="output schema", max_references=1)

    assert payload is not None
    references = payload["references"]
    assert len(references) == 1
    assert references[0]["path"] == "references/output_schema.md"
    assert references[0]["content"].startswith("content::")
    assert calls["search"] == 1
    assert calls["read"] == 1
