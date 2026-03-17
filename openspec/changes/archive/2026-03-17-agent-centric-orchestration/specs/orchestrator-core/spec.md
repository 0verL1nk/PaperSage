## ADDED Requirements

### Requirement: Single-Pass Execution
The orchestrator SHALL execute agent invocation in a single pass without mode detection loops.

#### Scenario: Direct agent invocation
- **WHEN** orchestrator receives a user prompt
- **THEN** it invokes the leader agent once with the prompt

#### Scenario: No pre-execution policy check
- **WHEN** orchestrator starts execution
- **THEN** it does not call policy_engine to decide modes

### Requirement: Tool-Based Mode Activation
The orchestrator SHALL handle mode activation through agent tool calls.

#### Scenario: Agent activates plan mode
- **WHEN** agent calls activate_plan_mode tool
- **THEN** the tool execution result is included in the agent's response

#### Scenario: Agent activates team mode
- **WHEN** agent calls activate_team_mode tool
- **THEN** the tool execution result is included in the agent's response

### Requirement: Middleware Integration
The orchestrator SHALL pass on_event callback to agent config for middleware access.

#### Scenario: Event callback configuration
- **WHEN** orchestrator invokes agent
- **THEN** it includes on_event in runtime_config.configurable

#### Scenario: Middleware event emission
- **WHEN** middleware needs to emit events
- **THEN** it can access on_event from the agent config
