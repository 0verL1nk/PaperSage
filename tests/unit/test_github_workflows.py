from pathlib import Path

import yaml


def test_claude_review_workflow_allows_bot_triggers() -> None:
    workflow_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "claude_review.yml"

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    review_step = workflow["jobs"]["review"]["steps"][-1]

    assert review_step["uses"] == "anthropics/claude-code-action@v1"
    assert review_step["with"]["allowed_bots"] == "*"
