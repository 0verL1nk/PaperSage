## Why

PaperSage currently has fixture-based router checks, but it does not have a repeatable way to evaluate whether an agent turn actually completed the user task. That leaves regressions in answer quality, evidence usage, and multi-step execution invisible until manual testing.

The project already exposes useful runtime signals such as traces, evidence items, todos, and plan state, so this is a good point to formalize an eval capability instead of relying on ad hoc scripts.

## What Changes

- Add a first-class end-to-end eval harness for agent turns that runs dataset-driven evaluations against PaperSage's stable runtime entrypoints and output contracts.
- Define a task-completion scoring model that combines final-answer success with process signals such as evidence usage and multi-step execution completion.
- Add eval fixtures and report generation for representative task categories such as fact lookup, retrieval-grounded answers, comparison tasks, and multi-step tasks.
- Add developer documentation for running evals locally and interpreting the resulting reports.
- Keep eval assertions anchored to stable end-to-end behavior so internal refactors do not break the suite unless observable task outcomes change.
- Make eval output actionable by surfacing failure reasons and optimization directions for prompts, retrieval/tool usage, or architecture changes.
- Keep the existing lightweight router baseline, but position it as a narrow routing check rather than the primary quality signal.

## Capabilities

### New Capabilities
- `agent-evals`: Dataset-driven evaluation for PaperSage agent turns, including task completion scoring, trajectory checks, and report generation.

### Modified Capabilities
None.

## Impact

- Affected code: `tests/evals/`, agent turn execution harnesses, runtime result normalization, and eval documentation.
- Dependencies: likely adds dev-only eval dependencies such as LangSmith and/or agent eval helpers.
- Systems: local developer workflow, CI/nightly quality checks, and future regression tracking for orchestration behavior.
