## 1. Runtime Tuning Cleanup

- [x] 1.1 Remove deprecated async interceptor controls and copy from `pages/2_settings.py`
- [x] 1.2 Stop applying deprecated `agent_policy_async_*` tuning values in the runtime tuning application layer
- [x] 1.3 Align `agent/settings.py` and any remaining runtime-facing comments with the current middleware-driven behavior
- [x] 1.4 Update `agent/domain/request_context.py` docstrings so they no longer describe `policy_engine.intercept()` as the active caller path

## 2. Test Migration

- [x] 2.1 Replace `agent.a2a.*` imports in eval and integration tests with assertions against the current application or middleware entrypoints
- [x] 2.2 Replace monkeypatch targets that reference `agent.orchestration.*` with targets in the current runtime modules
- [x] 2.3 Add or update focused tests proving deprecated async interceptor tuning no longer affects active runtime behavior
- [x] 2.4 Add or update focused tests proving orchestration guidance is produced through middleware-based execution

## 3. Documentation Sync

- [x] 3.1 Update `README.md` to describe the current canonical execution path and remove legacy async interceptor wording
- [x] 3.2 Update current-state architecture documentation that still presents `agent.a2a` or `agent.orchestration` as live runtime dependencies
- [x] 3.3 Ensure any user-facing settings descriptions match the reduced runtime tuning surface

## 4. Verification

- [x] 4.1 Run targeted unit and integration tests covering runtime tuning, settings, and migrated orchestration behavior
- [x] 4.2 Run the relevant OpenSpec validation/status check and confirm the change is apply-ready
- [x] 4.3 Record the verification commands and outcomes for implementation follow-through
