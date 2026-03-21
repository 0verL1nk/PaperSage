import json
from pathlib import Path

import yaml


def test_claude_review_workflow_allows_bot_triggers() -> None:
    workflow_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "claude_review.yml"

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    review_step = workflow["jobs"]["review"]["steps"][-1]

    assert review_step["uses"] == "anthropics/claude-code-action@v1"
    assert review_step["with"]["allowed_bots"] == "*"


def test_claude_workflows_pass_required_anthropic_env_vars_via_settings() -> None:
    workflows = (
        "claude_review.yml",
        "claude.yml",
    )
    expected_env = {
        "ANTHROPIC_AUTH_TOKEN": "${{ secrets.ANTHROPIC_AUTH_TOKEN }}",
        "ANTHROPIC_BASE_URL": "${{ secrets.ANTHROPIC_BASE_URL }}",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "${{ secrets.ANTHROPIC_DEFAULT_HAIKU_MODEL }}",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "${{ secrets.ANTHROPIC_DEFAULT_OPUS_MODEL }}",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "${{ secrets.ANTHROPIC_DEFAULT_SONNET_MODEL }}",
        "ANTHROPIC_MODEL": "${{ secrets.ANTHROPIC_MODEL }}",
    }

    for workflow_name in workflows:
        workflow_path = (
            Path(__file__).resolve().parents[2] / ".github" / "workflows" / workflow_name
        )
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        step = next(
            workflow_step
            for workflow_step in workflow["jobs"].values().__iter__().__next__()["steps"]
            if workflow_step.get("uses") == "anthropics/claude-code-action@v1"
        )

        settings = json.loads(step["with"]["settings"])
        assert settings["env"] == expected_env
        assert '--model "${{ secrets.ANTHROPIC_MODEL }}"' in step["with"]["claude_args"]
