## ADDED Requirements

### Requirement: Scheduler-facing todo records
The system SHALL support todo records with scheduler-facing metadata required for team execution.

#### Scenario: Todo stores assignee and backend
- **WHEN** the Leader creates or updates team todos
- **THEN** each todo may include assignment metadata such as assignee and execution backend
- **AND** the middleware persists those fields in state without stripping them

#### Scenario: Todo stores execution output metadata
- **WHEN** a teammate finishes or fails a todo
- **THEN** the todo record may store normalized result, artifact reference, retry count, and error details

### Requirement: Dependency-aware ready and blocked semantics
The system SHALL derive scheduler-ready todo state from dependency completion and failure information.

#### Scenario: Todo becomes ready when dependencies are completed
- **WHEN** all todos listed in `depends_on` are completed
- **THEN** the scheduler can treat the todo as ready for dispatch even if it was previously pending

#### Scenario: Todo becomes blocked when a dependency cannot be satisfied
- **WHEN** any todo listed in `depends_on` fails or is canceled
- **THEN** the dependent todo is treated as blocked until replanning or manual recovery occurs

### Requirement: Rich todo state model
The system SHALL support todo execution states beyond the current pending/in-progress/completed set.

#### Scenario: Todo enters failed state
- **WHEN** todo execution ends with an unrecoverable error
- **THEN** the system records the todo state as failed

#### Scenario: Todo enters canceled state
- **WHEN** the Leader or runtime stops a todo without execution success
- **THEN** the system records the todo state as canceled

### Requirement: Tool results expose scheduler convenience hints
The system SHALL surface scheduler-facing convenience data in tool results so the Leader can make stepwise dispatch decisions.

#### Scenario: Tool result lists ready and blocked todos
- **WHEN** the Leader creates or updates todos through the todo tool path
- **THEN** the tool result includes structured `ready_todos` and `blocked_todos` hints
- **AND** the Leader may use those hints without being forced into automatic dispatch
