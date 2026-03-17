# orchestration-tools Specification

## Purpose
TBD - created by archiving change agent-centric-orchestration. Update Purpose after archive.
## Requirements
### Requirement: Plan Mode Activation
The system SHALL provide an activate_plan_mode tool that creates execution plans.

#### Scenario: Plan creation
- **WHEN** agent calls activate_plan_mode with a goal description
- **THEN** the system generates a structured execution plan with steps

#### Scenario: Plan result format
- **WHEN** activate_plan_mode completes
- **THEN** it returns plan_text, steps list, and status="plan_activated"

### Requirement: Team Mode Activation
The system SHALL provide an activate_team_mode tool that enables multi-agent collaboration.

#### Scenario: Team execution
- **WHEN** agent calls activate_team_mode with a task description
- **THEN** the system spawns team members and executes the task collaboratively

#### Scenario: Team result format
- **WHEN** activate_team_mode completes
- **THEN** it returns summary, rounds count, and status="team_completed"

### Requirement: Tool Visibility
Both orchestration tools SHALL be marked as lazy for progressive disclosure.

#### Scenario: Initial tool list
- **WHEN** agent starts a new conversation
- **THEN** orchestration tools are not visible in the initial tool list

#### Scenario: Tool activation
- **WHEN** agent calls search_tools for "plan" or "team"
- **THEN** the corresponding orchestration tool becomes visible

