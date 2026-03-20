from agent.profiles import (
    paper_leader_profile,
    paper_reviewer_profile,
    paper_worker_profile,
    resolve_agent_profile,
)


def test_worker_and_reviewer_profiles_exclude_team_capability():
    assert "team_pack" not in paper_worker_profile.capability_ids
    assert "team_pack" not in paper_reviewer_profile.capability_ids
    assert "team_pack" in paper_leader_profile.capability_ids



def test_resolve_agent_profile_supports_leader_teammate_aliases():
    assert resolve_agent_profile("leader") is paper_leader_profile
    assert resolve_agent_profile("teammate") is paper_worker_profile
    assert resolve_agent_profile("worker") is paper_worker_profile
    assert resolve_agent_profile("reviewer") is paper_reviewer_profile



def test_worker_prompt_builder_does_not_delegate_dialog_ownership():
    prompt = paper_worker_profile.prompt_builder(
        document_name="文档A",
        project_name="项目A",
        scope_summary="范围A",
    )

    assert "不接管用户对话" in prompt
    assert "不创建下级 agent" in prompt



def test_reviewer_prompt_builder_focuses_on_review():
    prompt = paper_reviewer_profile.prompt_builder(
        document_name="文档A",
        project_name="项目A",
        scope_summary="范围A",
    )

    assert "你负责评审现有产出" in prompt
    assert "重点识别风险、缺证、冲突" in prompt
