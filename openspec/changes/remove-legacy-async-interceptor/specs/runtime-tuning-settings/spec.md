## ADDED Requirements

### Requirement: Runtime tuning only exposes active controls
The system SHALL expose only runtime tuning controls that are consumed by the current application and middleware execution path.

#### Scenario: Settings center hides deprecated async interceptor controls
- **WHEN** a user opens the Settings Center
- **THEN** the UI does not present controls for deprecated async policy interceptor enablement, refresh interval, confidence threshold, or staleness threshold

#### Scenario: Active runtime tuning controls remain available
- **WHEN** a user opens the Settings Center
- **THEN** the UI still presents controls for runtime options that are consumed by the current system, such as RAG indexing and document text cache limits

### Requirement: Deprecated async interceptor tuning is not applied to runtime
The system MUST NOT apply deprecated async policy interceptor tuning values to runtime environment variables or runtime configuration.

#### Scenario: Runtime tuning sync ignores deprecated async fields
- **WHEN** user-level runtime tuning settings are loaded and applied
- **THEN** the system does not emit or overwrite deprecated `AGENT_POLICY_ASYNC_*` runtime variables as active behavior controls

#### Scenario: Current runtime remains driven by active dependencies
- **WHEN** the agent center runtime is prepared for a turn
- **THEN** its behavior is determined by the current runtime dependencies and middleware chain rather than deprecated async interceptor tuning values
