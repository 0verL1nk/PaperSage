## ADDED Requirements

### Requirement: Dataset-Driven Agent Eval Execution
The system SHALL provide a dataset-driven eval harness that executes PaperSage agent turns through the project's canonical turn execution path.

#### Scenario: Run a local eval dataset
- **WHEN** a developer runs the agent eval command against a fixture file
- **THEN** the system executes each case through the PaperSage turn execution harness
- **AND** the system produces a case-level result for every valid fixture row

#### Scenario: Reuse canonical runtime entrypoints
- **WHEN** the eval harness executes a case
- **THEN** it MUST use the same application-level turn execution entrypoints used by PaperSage runtime code
- **AND** it MUST NOT duplicate business logic in a separate eval-only execution stack

### Requirement: End-To-End Contract Stability
The system SHALL define agent eval pass or fail decisions against stable end-to-end behavior contracts rather than volatile internal implementation details.

#### Scenario: Internal refactor without behavior change
- **WHEN** lower-level middleware, orchestration internals, or decomposition strategy changes without changing observable task behavior
- **THEN** the end-to-end eval suite continues to pass
- **AND** those internal changes do not require fixture rewrites unless a stable contract changed

#### Scenario: Stable contract assertions
- **WHEN** the eval harness scores a case
- **THEN** it uses stable contracts such as final answer success, evidence coverage, report fields, and declared task-completion requirements
- **AND** it does not require exact matches on private event names or internal step counts unless the case explicitly declares such a contract

### Requirement: Task Completion Scoring
The system SHALL calculate task completion using both final-answer success and process-level success signals.

#### Scenario: Simple task completion
- **WHEN** an eval case only defines final-answer success criteria
- **THEN** the system determines completion from the final-answer evaluation result

#### Scenario: Process-constrained task completion
- **WHEN** an eval case defines process requirements such as minimum evidence, required planning, or expected multi-step completion
- **THEN** the system includes those process checks in the completion decision
- **AND** the case is not marked completed unless both final-answer and required process checks pass

#### Scenario: Execution completion ratio
- **WHEN** a turn result includes plan or todo completion data
- **THEN** the system computes an execution completion ratio for the case
- **AND** records that ratio in the case result and aggregate report

### Requirement: Structured Eval Case Schema
The system SHALL support a structured eval fixture schema for defining prompts, categories, expected outcomes, and process constraints.

#### Scenario: Valid eval case fields
- **WHEN** the system loads an eval fixture row
- **THEN** it validates that the row contains a stable case identifier, a prompt, and the fields required by the selected scoring mode

#### Scenario: Retrieval-oriented constraints
- **WHEN** an eval case requires grounded retrieval behavior
- **THEN** the fixture schema can declare constraints such as minimum evidence count or evidence-required completion

#### Scenario: Multi-step task constraints
- **WHEN** an eval case requires planning or task tracking behavior
- **THEN** the fixture schema can declare constraints such as required plan usage, todo completion expectations, or trajectory hints
- **AND** those constraints MUST be expressed in terms of stable normalized outputs rather than middleware-private implementation details

### Requirement: Eval Reports
The system SHALL generate machine-readable eval reports with both aggregate metrics and case-level diagnostics.

#### Scenario: Aggregate report generation
- **WHEN** an eval run completes
- **THEN** the system writes a report containing completion rate and other configured aggregate metrics

#### Scenario: Case-level diagnostics
- **WHEN** an eval report is generated
- **THEN** it includes per-case pass or fail data, scoring details, and enough context to identify why a case failed

#### Scenario: Existing router baseline remains separate
- **WHEN** developers run the new task-completion eval flow
- **THEN** the existing router baseline remains available as a separate routing-focused check
- **AND** the new eval report does not replace router metrics with route-only success

### Requirement: Eval Feedback Loop
The system SHALL produce actionable feedback from eval results so maintainers can improve prompts, retrieval/tooling behavior, or architecture in a repeatable loop.

#### Scenario: Failed case recommendation
- **WHEN** a case fails or receives a weak score
- **THEN** the report includes a structured feedback entry describing the likely failure reason
- **AND** the feedback maps the issue to at least one remediation area such as prompt changes, retrieval/tool usage changes, or architecture changes

#### Scenario: Recommendation grounded in stable signals
- **WHEN** the system generates optimization feedback
- **THEN** it bases that feedback on stable end-to-end outputs and scoring evidence
- **AND** it does not require maintainers to inspect middleware-private implementation details to understand the recommendation

#### Scenario: Closed-loop optimization support
- **WHEN** maintainers review an eval report
- **THEN** they can identify which cases are best addressed by prompt tuning versus deeper system changes
- **AND** the report preserves enough case context to support the next implementation iteration
