## MODIFIED Requirements

### Requirement: Context Analysis
The middleware SHALL analyze current conversation context to determine task complexity, and this analysis SHALL be the canonical source of orchestration guidance in the active runtime.

#### Scenario: Simple task detection
- **WHEN** the latest user prompt is a single-step question
- **THEN** the middleware does not inject guidance prompts

#### Scenario: Complex task detection
- **WHEN** the conversation indicates multiple steps, dependencies, or multi-role work
- **THEN** the middleware identifies it as a complex task

#### Scenario: No legacy interceptor dependency
- **WHEN** middleware performs complexity analysis
- **THEN** it does not depend on a separate request-time async interceptor service to provide the decision

### Requirement: Guidance Injection
The middleware SHALL inject guidance prompts when complex tasks are detected without restoring a legacy policy-first routing layer.

#### Scenario: Plan tool guidance
- **WHEN** a multi-step task is detected
- **THEN** the middleware injects a system message suggesting the appropriate planning capability
- **AND** the guidance suggests the leader create or use a plan based on current context

#### Scenario: Team mode guidance
- **WHEN** a task requires multiple perspectives or parallel work
- **THEN** the middleware injects a system message suggesting team-oriented coordination tools

#### Scenario: No legacy policy routing restoration
- **WHEN** guidance is injected
- **THEN** the system does not restore a standalone policy-first router or async interceptor before model invocation
