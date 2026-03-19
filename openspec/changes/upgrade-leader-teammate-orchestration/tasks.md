## 1. Domain Models

- [x] 1.1 Add structured domain models for `RoleSpec`, `TeamPlan`, `TeamTodoRecord`, and team run state in the orchestration/domain layer
- [x] 1.2 Extend todo status modeling to include scheduler-facing states such as `ready`, `blocked`, `failed`, and `canceled`
- [x] 1.3 Add normalization and serialization helpers so team planning, todo records, and trace payloads use a stable contract

## 2. Todo Dependency Scheduling

- [x] 2.1 Extend `TodoGraph` with ready/block detection and dependency failure propagation helpers
- [x] 2.2 Update the todo middleware state/tool contract to persist assignee, backend, retry, result, and artifact metadata
- [x] 2.3 Implement a Leader-facing todo scheduler that selects ready todos and updates todo state transitions deterministically

## 3. Leader-Teammate Orchestration

- [ ] 3.1 Add a structured team planning path that converts team-mode activation into a `TeamPlan` plus executable todo records
- [ ] 3.2 Wire the orchestration middleware to emit structured team handoff state without bypassing Leader control
- [ ] 3.3 Update team execution flow so teammate dispatch, reviewer checkpoints, and Leader finalization consume the structured scheduler state

## 4. Execution Backends And State Machine

- [ ] 4.1 Refactor the current local `TeamRuntime` into a local execution backend with a normalized task-result contract
- [ ] 4.2 Add an A2A execution backend interface and initial implementation that preserves todo/run metadata and returns normalized results
- [ ] 4.3 Implement explicit team run and todo transition validation for scheduling, running, review, replan, completion, and failure flows

## 5. Verification And Documentation

- [ ] 5.1 Add or update unit tests for todo graph transitions, scheduler behavior, team state transitions, and executor contracts
- [ ] 5.2 Add or update integration tests for Leader-teammate execution with dependency-aware todos and backend selection
- [ ] 5.3 Update architecture-facing documentation to describe the structured Leader-teammate orchestration flow and new team runtime contracts
