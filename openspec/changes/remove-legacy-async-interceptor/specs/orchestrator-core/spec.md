## MODIFIED Requirements

### Requirement: Single-Pass Execution
The orchestrator SHALL execute agent invocation in a single pass through the current application and runtime agent path without legacy mode detection entrypoints.

#### Scenario: Direct agent invocation through current runtime
- **WHEN** the turn engine receives a user prompt
- **THEN** it invokes the leader agent through the current runtime agent and middleware stack

#### Scenario: No legacy pre-execution policy entrypoint
- **WHEN** orchestrator execution starts
- **THEN** it does not require `agent.a2a.*`, `agent.orchestration.*`, or a pre-execution `policy_engine.intercept()` call to select modes

### Requirement: Middleware Integration
The orchestrator SHALL pass runtime context through agent config so middleware remains the canonical source of complexity analysis and guidance.

#### Scenario: Event callback configuration
- **WHEN** the orchestrator invokes the agent
- **THEN** it includes `on_event` in `runtime_config.configurable`

#### Scenario: Middleware event emission
- **WHEN** middleware needs to emit events
- **THEN** it can access `on_event` from the agent config
