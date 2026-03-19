# Agent Evals

## Purpose

This eval flow measures whether PaperSage completes end-to-end tasks through stable output contracts. It is not a router-only check and it is not allowed to fail just because middleware internals or decomposition details changed.

The eval loop is:

1. Run the task-completion eval dataset.
2. Inspect completion, evidence, and feedback outputs.
3. Decide whether the next iteration belongs in prompt tuning, retrieval or tool policy, or architecture work.
4. Apply the change and rerun the same dataset.

## What The Eval Checks

Each case can combine two layers:

- Final-result checks:
  Final answer requirements or `LLM-as-judge` rubric evaluation.
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
- At least one success contract:
  - `expected_answer_all_of`
  - `expected_answer_any_of`
  - `forbidden_answer_any_of`
  - `success_rubric`

Optional stable process constraints:

- `requires_evidence`
- `min_evidence_count`
- `require_plan`
- `require_todos`
- `min_execution_completion_ratio`
- `required_tool_names`
- `required_phase_labels`

Example:

```json
{
  "id": "hybrid_research_001",
  "category": "hybrid_research",
  "prompt": "请结合当前项目文档和联网检索，评估 Self-RAG 是否适合纳入本项目后续路线。",
  "expected_answer_all_of": ["项目文档", "最新进展", "建议"],
  "requires_evidence": true,
  "min_evidence_count": 2,
  "require_plan": true,
  "min_execution_completion_ratio": 1.0,
  "required_tool_names": ["search_document", "search_web"],
  "required_phase_labels": ["规划", "输出最终答案"]
}
```

## Running

Local baseline:

```bash
make eval-baseline
```

With `LLM-as-judge`:

```bash
make eval-baseline-judge \
  JUDGE_MODEL="<judge-model-name>" \
  JUDGE_BASE_URL="<optional-openai-compatible-base-url>"
```

The judge path uses `agentevals` trajectory `LLM-as-judge`.

Small real smoke run:

```bash
make eval-live-smoke \
  EVAL_CASE_ID=hybrid_research_001 \
  EVAL_LIMIT=1
```

This path uses the real PaperSage runtime entrypoints with local paper fixtures and a live LLM. Keep it to one or two cases because it is intentionally slower and may exercise real web/tool latency.

Project-only smoke run without judge and with a tighter per-request timeout:

```bash
make eval-live-smoke-no-judge \
  EVAL_CASE_ID=project_rag_fact_001 \
  EVAL_LIMIT=1 \
  AGENT_LLM_REQUEST_TIMEOUT=45
```

Default variables:

- `EVAL_ENV_FILE=/home/ling/LLM_App_Final/.env`
- `EVAL_FIXTURE=tests/evals/fixtures/agent_task_eval_set_v1.jsonl`
- `EVAL_LIMIT=1`

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
  The answer contract was not satisfied, or planning instructions were too weak.
- `retrieval/tooling`
  The turn did not gather enough evidence or failed to use the required stable tools.
- `architecture`
  The system exposed a deeper execution reliability issue, such as repeated failure to complete multi-step work.

Use `recommended_actions` as the starting point for the next change, not as an auto-apply mechanism.

## Relationship To Router Baseline

`tests/evals/run_phase0_router_baseline.py` remains useful for routing checks, but it does not tell you whether the user task was actually completed. Treat router baseline and task-completion evals as complementary signals.
