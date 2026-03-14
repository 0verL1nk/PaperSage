from __future__ import annotations

from dataclasses import asdict

from agent.domain.openviking_contracts import (
    OpenVikingReadResult,
    OpenVikingSearchHit,
    OpenVikingSearchRequest,
)


def test_search_request_defaults_to_l2_tier():
    request = OpenVikingSearchRequest(project_uid="project-1", query="graph memory")

    assert request.tier == "L2"
    assert request.top_k == 8


def test_search_hit_and_read_result_are_code_facing_serializable_shapes():
    hit = OpenVikingSearchHit(
        resource_id="res-1",
        project_uid="project-1",
        content="summary",
        score=0.82,
        tier="L1",
        metadata={"source": "paper"},
    )
    read_result = OpenVikingReadResult(
        resource_id="res-1",
        project_uid="project-1",
        content="full content",
        tier="L2",
        metadata={"source": "paper"},
    )

    assert asdict(hit)["resource_id"] == "res-1"
    assert asdict(hit)["tier"] == "L1"
    assert asdict(read_result)["tier"] == "L2"
