## Why

The current repository has the building blocks for planning, todo dependencies, team tools, and A2A-shaped trace events, but they do not form a single execution loop. This makes it hard to claim that the system already supports Leader-driven multi-agent collaboration with plan-driven execution, dependency-aware scheduling, A2A task execution, and workflow state control.

## What Changes

- Introduce a structured Leader-teammate orchestration capability that turns team collaboration from prompt-level guidance into an explicit runtime flow.
- Add a team execution model with `RoleSpec`, `TeamPlan`, teammate task assignment, and reviewer checkpoints.
- Upgrade todo dependency handling from validation-only behavior to scheduler-ready execution state, including assignee, backend, and blocked/failed transitions.
- Add an execution backend abstraction so a todo can run through a local teammate runtime or an A2A-backed executor with a unified result contract.
- Add explicit team run and todo state transitions so planning, dispatch, execution, review, failure, and replan are observable and enforceable.

## Capabilities

### New Capabilities
- `leader-teammate-orchestration`: Leader-managed teammate planning, assignment, review, and final synthesis for team-mode execution.
- `a2a-task-execution`: Unified execution backend contract for local teammate tasks and A2A-backed task execution.
- `team-state-machine`: Explicit workflow and todo state transitions for team runs, retries, blocking, and replan decisions.

### Modified Capabilities
- `todolist-middleware`: Extend todo state and dependency handling so team scheduling can operate on structured ready/blocked/failed task records.
- `orchestration-middleware`: Change team activation from suggestion-only guidance to a structured handoff into the Leader-teammate orchestration flow.

## Impact

- Affected code: `agent/middlewares/`, `agent/domain/`, `agent/team/`, `agent/tools/`, `agent/application/`, and new orchestration/adapters modules as needed.
- Affected runtime behavior: team-mode execution, todo lifecycle, reviewer checkpoints, and trace payload shape.
- Affected protocol boundary: local team execution will gain a unified executor abstraction that can host A2A-backed task execution.
