## MODIFIED Requirements

### Requirement: Guidance Injection
The middleware SHALL inject guidance prompts and structured handoff signals when complex tasks are detected.

#### Scenario: Plan tool guidance
- **WHEN** a multi-step task is detected and no team execution is required
- **THEN** the middleware injects a system message suggesting plan-oriented tooling
- **AND** the guidance preserves Leader authorship of plan content

#### Scenario: Structured team handoff guidance
- **WHEN** a task requires multiple perspectives, dependency-aware execution, or parallel work
- **THEN** the middleware injects a system message directing the Leader toward team-mode orchestration
- **AND** the middleware records a structured signal in runtime state that team orchestration is eligible for activation
- **AND** the guidance preserves the Leader as the pacing and dialogue owner

#### Scenario: No direct runtime side effects
- **WHEN** guidance or structured handoff metadata is injected
- **THEN** the middleware does NOT dispatch teammate tasks by itself
- **AND** the middleware does NOT bypass the Leader's runtime decision path

### Requirement: Non-intrusive Operation
The middleware SHALL preserve agent autonomy while enabling structured orchestration entry.

#### Scenario: Agent ignores guidance
- **WHEN** the middleware suggests a planning or team mode path but the agent chooses not to use it
- **THEN** the system continues normal execution without errors

#### Scenario: Guidance plus state signal
- **WHEN** guidance is injected for a team-eligible task
- **THEN** the middleware may add structured eligibility state for downstream orchestration components
- **AND** that state alone does not force execution without an agent/runtime handoff

### Requirement: Prompt guidance reinforces Leader-owned pacing
The middleware SHALL describe team orchestration as a Leader-controlled option rather than a fixed automatic loop.

#### Scenario: Team prompt emphasizes Leader control
- **WHEN** the middleware injects team guidance
- **THEN** the prompt explains that `team_handoff` and scheduler hints are advisory
- **AND** the prompt instructs the Leader to decide whether to dispatch, review, or replan
