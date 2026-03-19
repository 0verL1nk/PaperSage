## ADDED Requirements

### Requirement: Leader-managed team planning
The system SHALL support a structured Leader-teammate orchestration flow that converts a team-mode request into an explicit team plan before teammate execution begins.

#### Scenario: Leader creates a structured team plan
- **WHEN** the system activates team mode for a complex request
- **THEN** the Leader produces a `TeamPlan` that includes the team goal, teammate role specs, and executable todo records
- **AND** the plan is stored in runtime state before any teammate task is dispatched

#### Scenario: Team plan remains inspectable
- **WHEN** a team plan exists for the active run
- **THEN** the system preserves the role specs, todo dependencies, and completion criteria in structured state
- **AND** downstream scheduler and review logic consume that structured state instead of reparsing free-form text

### Requirement: Leader assigns teammate work from ready todos
The system SHALL dispatch teammate work from dependency-satisfied todo records rather than from ad hoc prompt text.

#### Scenario: Ready todo is assigned to a teammate
- **WHEN** a todo has no unresolved dependencies and is not completed, blocked, failed, or canceled
- **THEN** the scheduler marks the todo ready for execution
- **AND** the Leader assigns it to a teammate with the todo instructions and relevant role contract

#### Scenario: Multiple ready todos can be dispatched independently
- **WHEN** two or more todos are simultaneously ready
- **THEN** the scheduler may dispatch them independently without violating dependency order
- **AND** their results are recorded against the originating todo records

### Requirement: Reviewer checkpoints gate final synthesis
The system SHALL support reviewer checkpoints so Leader finalization can distinguish between accepted output, revision-required output, and replan-required output.

#### Scenario: Reviewer accepts team output
- **WHEN** reviewer checks determine that required evidence and task goals are satisfied
- **THEN** the run proceeds to Leader final synthesis

#### Scenario: Reviewer requests revision or replan
- **WHEN** reviewer checks detect missing evidence, contradictions, or unmet todo completion criteria
- **THEN** the system records a revision or replan decision in structured run state
- **AND** the Leader receives that decision before producing the final answer
