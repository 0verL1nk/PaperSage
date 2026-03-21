## Context

The current codebase already contains most of the raw parts needed for team-mode execution: `OrchestrationMiddleware` can detect complex tasks, `TeamMiddleware` exposes team tools, `TeamRuntime` can create local sub-agents, `TodoGraph` can validate dependency graphs, and `ExecutionPlan` models already exist in `agent/domain/orchestration.py`. The problem is that these parts are not connected by a single runtime contract. Team mode is still prompt-led instead of scheduler-led, todo dependencies are validated but not executed as a graph, A2A exists only as a trace/message shape, and team lifecycle states are implicit.

The design needs to preserve the repository's current direction: thin `pages/` and `ui/`, orchestration logic outside the page layer, no new meaningless wrappers, and a preference for small, testable domain/application modules. The right upgrade is not to add more agent personas. It is to make the existing Leader-teammate model explicit, stateful, backend-agnostic, and still Leader-owned in pacing.

## Goals / Non-Goals

**Goals:**
- Turn team mode into an explicit `Leader -> teammate todos -> review -> finalize/replan` runtime.
- Reuse todo dependencies as the primary topology model instead of introducing a second parallel task-graph abstraction.
- Add structured team planning inputs (`RoleSpec`, `TeamPlan`, scheduler-facing todo records) so teammate work is inspectable and testable.
- Support a unified task execution backend contract for local teammate execution and future A2A-backed execution.
- Introduce explicit team run and todo state transitions so failures, blocking, retries, and replans are deterministic.
- Keep the Leader as the only user-facing dialogue owner and preserve its control over dispatch cadence, reviewer checkpoints, and replanning decisions.
- Surface scheduler convenience data in tool results and prompt guidance rather than hiding decisions inside an automatic coordinator loop.

**Non-Goals:**
- Do not build a free-form group chat or peer-to-peer multi-agent system.
- Do not add many new role types; the default shape remains Leader plus generic teammates, with reviewer checkpoints.
- Do not replace existing `write_plan` semantics with a fully separate global planning engine.
- Do not implement remote agent discovery, marketplace routing, or protocol negotiation in this change.

## Decisions

### 1. Adopt a Leader-teammate runtime, not a many-role swarm

The default team architecture will be:

`Leader -> TeamPlan -> dependency-aware teammate todos -> Reviewer -> Leader finalize/replan`

This is the smallest design that matches the repository's current code and the desired product claim. It preserves the existing Team middleware and runtime idea, but replaces prompt-only coordination with a structured execution loop.

Alternatives considered:
- Keep the current prompt-led team tools and rely on better prompts. Rejected because it does not create a testable runtime contract.
- Introduce many fixed roles (`researcher`, `writer`, `comparer`, etc.) as first-class runtime types. Rejected because role proliferation would increase coordination cost before the execution model is stable.

### 2. Use todo dependency records as the scheduling graph

The scheduler will treat `depends_on` on todo records as the canonical topology model. `TodoGraph` will be extended to expose `ready`, `blocked`, and failure propagation semantics, and team scheduling will operate on these records directly.

This avoids duplicating graph logic between `ExecutionPlan.steps` and team todos. The plan remains useful as an authoring and trace artifact, while executable work is normalized into todo records.

Alternatives considered:
- Build a new dedicated `TaskGraph` abstraction unrelated to todos. Rejected because the repo already has dependency-aware todo primitives and UI support.
- Schedule directly from `ExecutionPlan.steps`. Rejected because the todo layer is already the user-visible and middleware-persisted task representation.

### 3. Add structured team planning contracts

The runtime will add domain models for:
- `RoleSpec`: teammate identity, goal, allowed tools, expected output.
- `TeamPlan`: leader-authored/leader-generated team execution plan.
- `TeamTodoRecord`: scheduler-facing todo with assignment, backend, status, result, retries, and artifact metadata.

These types live in the domain/orchestration layer and become the contract between middleware guidance, scheduler execution, and UI rendering.

Alternatives considered:
- Keep `name + system_prompt` as the only dynamic role contract. Rejected because it is too weak for review, scheduling, and A2A/backend routing.

### 4. Split orchestration into planner, scheduler, executor, and state machine

This change will introduce orchestration modules with clear responsibilities:
- planner: build `TeamPlan` and initial `TeamTodoRecord`s
- scheduler: select ready todos and dispatch them
- executors: run local teammate tasks or A2A-backed tasks
- state machine: validate transitions for team runs and todo records

`OrchestrationMiddleware` remains the entry detector, but it no longer owns the full behavior. `TeamRuntime` becomes one execution backend, not the orchestration brain.

Alternatives considered:
- Keep all logic inside middleware. Rejected because middleware should not become the only place where planning, dispatch, retry, and failure handling live.

### 7. Keep policy in the Leader and convenience in tool results

The runtime should distinguish hard constraints from pacing decisions:
- hard constraints: dependency topology, valid state transitions, backend result normalization
- Leader-owned policy: whether to dispatch now, how many todos to dispatch, whether to parallelize, when to review, and when to replan

This means the main path should not treat coordinator auto-dispatch as the product contract. Instead, tool results and prompts should surface:
- `team_handoff`
- `ready_todos`
- `blocked_todos`
- `review_hint`
- `allowed_transitions`

These hints help the Leader choose the next move while preserving user-facing dialogue ownership.

### 5. Treat A2A as an execution backend, not the primary orchestration model

The repository already serializes A2A-shaped trace events, but there is no working A2A executor. This change adds a backend abstraction where a todo can declare `execution_backend = local | a2a`. The scheduler does not care which backend runs a task; it only consumes normalized execution results.

Alternatives considered:
- Make all team coordination flow through A2A message exchange. Rejected because the current repository does not have an A2A task runtime, and this would over-expand scope.

### 6. Add explicit workflow states

Todo states will be extended beyond `pending / in_progress / completed` to support `ready`, `blocked`, `failed`, and `canceled`. Team runs will gain explicit lifecycle states such as `draft`, `scheduled`, `running`, `reviewing`, `replanning`, `completed`, and `failed`.

This state machine is required for deterministic retries, failure propagation, and UI/trace consistency.

Alternatives considered:
- Infer everything from trace events. Rejected because traces are good observability data but poor control-plane state.

## Risks / Trade-offs

- [More runtime structure increases code surface] → Mitigation: reuse existing todo graph, team runtime, and plan models instead of inventing parallel concepts.
- [Old prompt-driven team behavior may conflict with scheduler-driven execution] → Mitigation: gate the new flow behind explicit team-plan state and preserve backward-compatible tool availability during migration.
- [An over-eager coordinator may become a hard-coded workflow engine] → Mitigation: keep automatic dispatch as an internal helper only; make tool-result hints and prompt guidance the primary control surface for the Leader.
- [A2A backend support may stay partially implemented if protocol-side integration lags] → Mitigation: define the executor interface now and keep local execution as the default backend.
- [State machine complexity can leak into UI/session state] → Mitigation: keep state transition logic in domain/orchestration modules and expose only normalized records to UI.
- [Todo contract changes may ripple through tests and rendering] → Mitigation: add additive fields first and keep legacy rendering paths working until migration completes.

## Migration Plan

1. Introduce new domain models for role specs, team plans, team todo records, and workflow states.
2. Extend todo graph/state handling to support scheduler-facing ready/blocked/failed transitions.
3. Introduce a Leader-teammate scheduler that dispatches ready todos through local execution first.
4. Adapt `TeamRuntime` into a local execution backend and add an A2A executor interface stub/implementation.
5. Update middleware prompts and tool results so team-mode execution exposes structured hints without auto-bypassing the Leader.
6. Update middleware and application wiring so team-mode execution enters the structured runtime when activated.
7. Update UI and traces to render new todo/team state without breaking existing conversations.
8. Roll back by disabling the structured team path and falling back to the current prompt-led team tools if regressions appear.

## Open Questions

- Should reviewer checkpoints run only at the end of the team run, or also after configurable critical todos?
- Should `write_todos` remain directly user-visible in all team flows, or should Leader-generated team todos be stored through a distinct orchestration API and mirrored into middleware state?
- For A2A execution, should the first version support only request/response task execution, or also long-running remote task polling?
