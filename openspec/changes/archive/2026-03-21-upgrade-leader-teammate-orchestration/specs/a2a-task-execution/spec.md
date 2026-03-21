## ADDED Requirements

### Requirement: Unified task execution backend contract
The system SHALL execute teammate todos through a backend abstraction that normalizes local and A2A-backed task execution results.

#### Scenario: Local teammate execution returns normalized output
- **WHEN** a todo is assigned to the local execution backend
- **THEN** the executor returns a normalized task result containing status, output, and error metadata

#### Scenario: A2A-backed execution returns normalized output
- **WHEN** a todo is assigned to the A2A execution backend
- **THEN** the executor returns the same normalized task result contract used by local execution
- **AND** scheduler logic does not require backend-specific branching to read the result

### Requirement: Todo backend selection is explicit
The system SHALL allow each executable todo to declare which backend executes it.

#### Scenario: Todo selects local backend
- **WHEN** the Leader or planner marks a todo for local teammate execution
- **THEN** the scheduler dispatches it through the local executor path

#### Scenario: Todo selects A2A backend
- **WHEN** the Leader or planner marks a todo for A2A-backed execution
- **THEN** the scheduler dispatches it through the A2A executor path
- **AND** a backend-specific failure only fails that todo unless the scheduler propagates the error

### Requirement: A2A execution preserves task metadata
The system SHALL preserve team run metadata and todo identity when dispatching a task through an A2A executor.

#### Scenario: A2A task carries todo identity
- **WHEN** the scheduler dispatches a todo through the A2A executor
- **THEN** the todo id, run id, and teammate identity are included in the outbound task metadata
- **AND** the returned result can be matched to the originating todo record
