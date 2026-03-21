# GitHub Actions

## Claude PR Review Bot Triggers

`/.github/workflows/claude_review.yml` enables `anthropics/claude-code-action@v1` on pull request open events.

The workflow sets `allowed_bots: "*"` so bot-authored pull requests, including Dependabot, can trigger the review action without failing the non-human actor guard.
