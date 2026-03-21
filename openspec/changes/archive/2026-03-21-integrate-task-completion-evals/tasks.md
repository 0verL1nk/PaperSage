## 1. Eval Harness Foundation

- [x] 1.1 Add dev-only eval dependencies needed for local judge-backed and rule-based evaluation flows.
- [x] 1.2 Create an eval harness module that loads structured eval fixtures and executes cases through the canonical PaperSage turn execution path.
- [x] 1.3 Add result normalization utilities that convert turn outputs into a stable eval record shape for scoring and report generation.
- [x] 1.4 Define the stable end-to-end eval contract and exclude middleware-private implementation details from default pass or fail assertions.

## 2. Scoring And Reporting

- [x] 2.1 Implement final-answer evaluators that support rubric-based or reference-based success checks per case.
- [x] 2.2 Implement process evaluators for evidence coverage, required planning or todo behavior, and execution completion ratio.
- [x] 2.3 Combine final-answer and process results into a task-completion decision and aggregate metrics such as completion rate and average execution completion ratio.
- [x] 2.4 Write JSON report generation with aggregate metrics and case-level diagnostics.
- [x] 2.5 Ensure diagnostic traces remain non-blocking by default unless a case explicitly opts into a stable process contract.
- [x] 2.6 Add structured feedback generation that classifies likely remediation areas such as prompt tuning, retrieval/tool updates, or architecture work.

## 3. Fixtures And Commands

- [x] 3.1 Define the structured eval fixture schema and add a seed dataset covering fact, retrieval, comparison, and multi-step tasks.
- [x] 3.2 Add a runnable command or script for executing the new eval flow locally without replacing the existing router baseline command.
- [x] 3.3 Add automated tests for fixture validation, scoring logic, and report structure.

## 4. Documentation And Rollout

- [x] 4.1 Document how to run the eval harness locally, how to add new cases, and how to interpret the report outputs.
- [x] 4.2 Document the intended use of the new task-completion evals versus the existing router baseline.
- [x] 4.3 Verify the end-to-end eval flow in the worktree and capture the initial run result for future comparison.
- [x] 4.4 Document the optimization loop for using eval feedback to drive prompt and architecture iteration.
