## ADDED Requirements

### Requirement: Context Analysis
The middleware SHALL analyze the conversation context to determine task complexity.

#### Scenario: Simple task detection
- **WHEN** the user prompt is a single-step question
- **THEN** the middleware does not inject guidance prompts

#### Scenario: Complex task detection
- **WHEN** the user prompt contains multiple steps or dependencies
- **THEN** the middleware identifies it as a complex task

### Requirement: Guidance Injection
The middleware SHALL inject guidance prompts when complex tasks are detected.

#### Scenario: Plan tool guidance
- **WHEN** a multi-step task is detected
- **THEN** the middleware injects a system message suggesting create_plan tool
- **AND** the guidance suggests Leader write their own plan based on context

#### Scenario: Team mode guidance
- **WHEN** a task requires multiple perspectives or parallel work
- **THEN** the middleware injects a system message suggesting activate_team_mode tool

#### Scenario: No plan generation
- **WHEN** guidance is injected
- **THEN** the middleware does NOT call planning_service.build_execution_plan
- **AND** the middleware does NOT generate plan content for Leader

### Requirement: Non-intrusive Operation
The middleware SHALL NOT force agent decisions or override agent autonomy.

#### Scenario: Agent ignores guidance
- **WHEN** the middleware suggests a mode but agent chooses not to use it
- **THEN** the system continues normal execution without errors

#### Scenario: Guidance as suggestion
- **WHEN** guidance is injected
- **THEN** it is phrased as a suggestion, not a command
