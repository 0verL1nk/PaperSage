# Agent Evals

## Purpose

This eval flow measures whether PaperSage completes end-to-end tasks through stable output contracts. It is not a router-only check and it is not allowed to fail just because middleware internals or decomposition details changed.

The eval loop is:

1. Run the task-completion eval dataset.
2. Inspect completion, evidence, and feedback outputs.
3. Decide whether the next iteration belongs in prompt tuning, retrieval or tool policy, or architecture work.
4. Apply the change and rerun the same dataset.

## What The Eval Checks

Each case combines two layers:

- Final-result checks:
  Every case must define a `success_rubric`, and final-answer scoring is always `LLM-as-judge`.
- Process checks:
  Stable contract checks such as evidence count, required tool names, plan presence, todo presence, and execution completion ratio.

Final task completion is computed as:

```text
completed = final_success AND process_success
```

Private middleware event names, exact internal step counts, or specific decomposition choices are not default pass or fail conditions.

## Fixture Schema

Fixtures live in JSONL and each row must include:

- `id`
- `category`
- `prompt`
- `success_rubric`
- `document_access` (`scoped` or `none`)

Keyword-matching fields such as `expected_answer_all_of`, `expected_answer_any_of`, and `forbidden_answer_any_of` are rejected. The goal is to evaluate task completion with a rubric, not string overlap.

The default fixture is no longer a toy set. It now mixes project-only questions, project-gap and boundary checks, web-only recency/tradeoff questions, and hybrid roadmap-decision cases such as rollout, guardrails, and defer-or-adopt judgments.

The default fixture intentionally avoids `required_phase_labels`, because phase-path strings are too coupled to internal orchestration details and would make end-to-end evals brittle across implementation changes.

Optional stable process constraints:

- `requires_evidence`
- `min_evidence_count`
- `require_plan`
- `require_todos`
- `min_execution_completion_ratio`
- `required_tool_names`
- `required_phase_labels`

Optional live-scope isolation metadata:

- `document_access`: `scoped` keeps project-document prompt/tool access enabled, `none` disables project-document prompt injection and omits document tools entirely
- `document_scope`: when `document_access` is `scoped`, limits the case to an explicit list of `doc_uid` values from the local fixture corpus

Example:

```json
{
  "id": "hybrid_research_001",
  "category": "hybrid_research",
  "prompt": "请结合当前项目文档和联网检索，评估 Self-RAG 是否适合纳入本项目后续路线。",
  "success_rubric": "Answer should combine current project evidence with recent web findings, weigh adoption benefits versus engineering cost or risk, and finish with a concrete roadmap recommendation such as adopt, defer, or pilot.",
  "requires_evidence": true,
  "min_evidence_count": 2,
  "require_plan": true,
  "min_execution_completion_ratio": 1.0,
  "required_tool_names": ["search_document", "search_web"],
  "required_phase_labels": ["规划", "输出最终答案"],
  "document_access": "scoped",
  "document_scope": ["arxiv:2005.11401", "arxiv:2310.11511"]
}
```

## Running

Local baseline:

```bash
make eval-baseline
```

Override judge model settings if needed:

```bash
make eval-baseline-judge \
  JUDGE_MODEL="<judge-model-name>" \
  JUDGE_BASE_URL="<optional-openai-compatible-base-url>"
```

The baseline runner now loads `.env` by default and always builds the judge.

Small real smoke run:

```bash
make eval-live-smoke \
  EVAL_CASE_ID=hybrid_research_001 \
  EVAL_LIMIT=1
```

This path uses the real PaperSage runtime entrypoints with local paper fixtures and a live LLM. Keep it to one or two cases because it is intentionally slower and may exercise real web/tool latency.

Default variables:

- `EVAL_ENV_FILE=/home/ling/LLM_App_Final/.env`
- `EVAL_FIXTURE=tests/evals/fixtures/agent_task_eval_set_v1.jsonl`
- `EVAL_LIMIT=1`

## Judge Behavior

`build_trajectory_llm_as_judge` now passes both the case `prompt` and `success_rubric` into the judge prompt. This avoids the previous failure mode where the judge only saw the raw trajectory or output messages without the actual task-specific success criteria.

## Report Fields

Aggregate fields include:

- `completion_rate`
- `final_success_rate`
- `process_success_rate`
- `evidence_coverage_rate`
- `average_execution_completion_ratio`
- `remediation_area_counts`

Case-level fields include:

- `completed`
- `final_success`
- `process_success`
- `evidence_coverage`
- `process_checks`
- `feedback`

## Using The Feedback Loop

Use `feedback.remediation_area` to decide the next move:

- `prompt`
  The rubric judged the final answer as incomplete, incorrect, or insufficiently grounded.
- `retrieval/tooling`
  The turn did not gather enough evidence or failed to use the required stable tools.
- `architecture`
  The system exposed a deeper execution reliability issue, such as repeated failure to complete multi-step work.

Use `recommended_actions` as the starting point for the next change, not as an auto-apply mechanism.

## Relationship To Router Baseline

`tests/evals/run_phase0_router_baseline.py` remains useful for routing checks, but it does not tell you whether the user task was actually completed. Treat router baseline and task-completion evals as complementary signals.
