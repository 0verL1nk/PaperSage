## Context

PaperSage already has two useful layers for evaluation work:

- The runtime entry layer built around `create_paper_agent_session(...)` and `execute_turn_core(...)`, which can execute a realistic turn and return normalized fields such as `answer`, `trace_payload`, `evidence_items`, `todos`, and `runtime_state`.
- The existing `tests/evals/` folder, which proves that the project already accepts fixture-driven baselines, but only for narrow router classification checks.

The gap is that current evals do not answer the question that matters most for this project: did the agent complete the user task with the required evidence and execution behavior. This change adds that missing eval layer without moving business logic into test code and without replacing the existing router baseline.

Constraints:

- The harness must stay aligned with existing application boundaries and use canonical execution entrypoints instead of duplicating runtime logic.
- The harness must evaluate end-to-end behavior through stable contracts and must not fail solely because middleware internals, private event names, or lower-level implementation details were refactored.
- Eval dependencies should remain dev-only.
- The first version must support both deterministic rule checks and judge-based scoring, because some tasks are better validated by structured rules while others require rubric-based grading.

Stakeholders are maintainers who need regression detection for agent behavior, and contributors who need a repeatable local command before merging orchestration changes.

## Goals / Non-Goals

**Goals:**

- Provide a dataset-driven eval harness for end-to-end agent turns.
- Define a repeatable task-completion metric that can combine final-answer quality with process-level signals.
- Reuse existing runtime outputs instead of inventing parallel state models for eval.
- Produce machine-readable reports that can be compared across runs.
- Document how developers add cases and run evals locally.
- Produce actionable feedback that helps decide whether the next change belongs in prompts, retrieval/tooling policy, or architecture.

**Non-Goals:**

- Replacing the existing router baseline.
- Turning evals into a hard gate for every local `core` quality run in the first iteration.
- Building a hosted eval UI or long-term run storage service in this change.
- Refactoring orchestration internals solely for architectural neatness if they are not needed to support the eval harness.

## Decisions

### 1. Use `execute_turn_core(...)` as the primary business-level eval entrypoint

The eval harness will run realistic turns through the same execution path used by the application layer instead of assembling bespoke test-only flows. This keeps the harness aligned with current behavior and ensures it scores the normalized turn result that PaperSage already exposes.

Alternatives considered:

- Call the raw agent directly and inspect raw LangChain messages only.
  Rejected because it would bypass the project's normalized result contract and make eval logic tightly coupled to raw runtime internals.
- Build a new eval-only service layer.
  Rejected because it adds an extra abstraction with little semantic value.

### 1a. Bind assertions to stable end-to-end contracts, not volatile internals

The eval suite will assert against stable artifacts such as prompt, normalized turn result, report metrics, evidence coverage, and task-completion outcomes. It will avoid making core pass/fail decisions depend on private middleware event names, exact internal step counts, or a specific decomposition strategy, because those are implementation details that may legitimately change during refactors.

Internal traces can still be recorded as diagnostics, but they should remain secondary and non-blocking unless a case explicitly declares a stable process contract.

Alternatives considered:

- Assert exact raw internal trajectories for most cases.
  Rejected because it would make the suite brittle and punish safe refactors instead of catching user-visible regressions.
- Ignore process information entirely.
  Rejected because some task-completion requirements, such as evidence grounding or required planning, need process-level checks.

### 2. Split scoring into final-result and process-result components

Task completion will not be a single text-only judge. The harness will compute completion from two layers:

- Final-result success: whether the answer satisfies the case rubric.
- Process-result success: whether required execution constraints were met, such as evidence usage, expected tool or trajectory behavior, and multi-step completion signals.

This allows simple fact cases to stay lightweight while complex research tasks can require stronger process validation.

Alternatives considered:

- Judge only the final answer.
  Rejected because it would miss regressions where the answer is plausible but unsupported or skips the intended multi-step flow.
- Rule-check only process signals.
  Rejected because traces alone cannot determine whether the user goal was actually fulfilled.

### 3. Model eval cases as structured JSONL fixtures

Each eval case will store prompt, category, scoring expectations, and optional process constraints in a structured fixture format that can be versioned in-repo. The schema will support requirements such as minimum evidence count, required plan usage, expected trajectory/tool hints, and a rubric or reference output for final scoring.

Alternatives considered:

- Encode eval cases directly in Python tests.
  Rejected because that makes non-trivial case updates noisy and discourages dataset growth.
- Use only a hosted external dataset.
  Rejected for the first iteration because local reproducibility matters more than remote management.

### 4. Support both rule-based checks and LangChain-compatible evaluator adapters

The implementation will add a local evaluator layer that can run deterministic checks directly on turn results and optionally delegate qualitative scoring to LangChain/LangSmith-style evaluators. This preserves the ability to run useful evals locally even when hosted services are unavailable.

Alternatives considered:

- Depend entirely on LangSmith-hosted evaluation.
  Rejected because it would make the first iteration harder to run in local and restricted environments.
- Ignore LangChain evaluator patterns and build a fully custom framework.
  Rejected because the project explicitly wants to align with the LangChain eval model.

### 5. Emit JSON reports with aggregate metrics and case-level breakdowns

Each eval run will produce a report with aggregate metrics such as completion rate, evidence coverage rate, trajectory pass rate, and average execution completion ratio, plus a per-case breakdown for debugging. This matches the current baseline style and keeps outputs easy to archive.

Alternatives considered:

- Print-only console output.
  Rejected because it is harder to compare over time or consume in automation.

### 6. Add a feedback layer that classifies likely remediation paths

Eval output should not stop at pass/fail. For failed or weak cases, the system will attach structured feedback that points maintainers toward the most likely remediation area, such as:

- prompt or rubric mismatch
- missing or weak retrieval/tool usage
- unstable task decomposition
- architecture or capability gap that cannot be fixed by prompt changes alone

The classification can begin with deterministic heuristics over stable outputs and later be upgraded with a judge-backed summarizer. The key requirement is that reports support a practical optimization loop instead of forcing humans to infer everything manually.

Alternatives considered:

- Leave interpretation entirely manual.
  Rejected because it slows iteration and weakens the value of recurring eval runs.
- Auto-apply prompt changes directly from eval output.
  Rejected because the first version should recommend changes, not mutate the system without review.

## Risks / Trade-offs

- [Judge variability] -> Mitigation: keep deterministic checks separate from qualitative scoring and store raw judge decisions in reports.
- [Runtime outputs are not yet complete for every desired signal] -> Mitigation: add a thin eval adapter around existing turn results rather than broad runtime refactors.
- [End-to-end checks drift into internal coupling over time] -> Mitigation: define a stable eval record schema and require new assertions to justify why they belong in the end-to-end contract.
- [Eval cost and latency grow with case count] -> Mitigation: start with a compact seed dataset and keep heavier runs out of the default core gate.
- [Fixture drift makes scores noisy] -> Mitigation: version fixture schema, keep category-specific rubrics explicit, and require report regeneration when cases change.
- [Feedback recommendations become noisy or misleading] -> Mitigation: start with coarse recommendation buckets and always include the raw evidence used to justify the recommendation.

## Migration Plan

1. Add dev-only dependencies and local harness modules.
2. Introduce the new eval fixture schema and seed dataset.
3. Implement report generation and documentation.
4. Keep router baseline commands intact during rollout.
5. Add CI or scheduled automation only after the local command is stable and trusted.

Rollback is low-risk: remove the new eval command, fixtures, and dev dependencies while leaving existing router baseline checks untouched.

## Open Questions

- Which subset of evals should eventually run in CI versus only on demand?
- Should judge-backed scoring default to the same model under test, or a separate evaluation model?
- Does `execute_turn_core(...)` need to expose additional raw message data in the first implementation, or can the initial trajectory checks rely on `trace_payload` and normalized outputs alone?
- Which process signals are stable enough to be treated as contract-level assertions, and which should remain diagnostics only?
- How far should the first feedback loop go: recommendation-only, or include generated patch ideas for prompts and configs?
