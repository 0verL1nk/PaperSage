## ADDED Requirements

### Requirement: Team runs use explicit workflow states
The system SHALL track structured workflow state for each team run.

#### Scenario: Team run enters scheduled and running states
- **WHEN** a structured team plan has been accepted for execution
- **THEN** the team run transitions from draft to scheduled
- **AND** transitions to running when teammate dispatch begins

#### Scenario: Team run enters review and completion states
- **WHEN** executable todos have finished and reviewer checks begin
- **THEN** the team run transitions to reviewing
- **AND** transitions to completed only after final synthesis succeeds

### Requirement: Todo transitions are validated
The system SHALL validate todo state transitions against an explicit transition model.

#### Scenario: Pending todo becomes ready
- **WHEN** all dependencies of a pending todo are completed
- **THEN** the system may transition the todo to ready

#### Scenario: Running todo fails
- **WHEN** execution returns an unrecoverable error for an in-progress todo
- **THEN** the system transitions the todo to failed
- **AND** records the failure for scheduler and review logic

#### Scenario: Downstream todo becomes blocked
- **WHEN** a dependency todo fails or is canceled
- **THEN** each downstream todo that depends on it transitions to blocked unless the Leader explicitly replans the graph

### Requirement: Replan is an explicit workflow transition
The system SHALL represent replanning as a workflow state transition instead of an implicit prompt decision.

#### Scenario: Reviewer triggers replan
- **WHEN** reviewer output or scheduler state indicates that current execution cannot satisfy the team goal
- **THEN** the team run transitions to replanning
- **AND** the next plan version replaces or updates the prior executable todo set
