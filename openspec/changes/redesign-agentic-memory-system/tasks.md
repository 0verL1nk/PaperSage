## 1. Data Model And Episode Pipeline

- [x] 1.1 Design and migrate SQLite schema for raw memory episodes, structured memory items, and evidence links
- [x] 1.2 Add repository-layer read/write APIs for episodes, typed memory items, lifecycle state, and evidence metadata
- [x] 1.3 Replace the current synchronous turn-memory persistence entrypoint so completed turns persist raw episodes instead of canonical `Q/A` fragments
- [x] 1.4 Add stable dedup-related fields and indexes such as canonical text identity, dedup key, and active-status filtering

## 2. Async Memory Writer And Reconcile Logic

- [x] 2.1 Add a dedicated async memory writer task that reads a saved episode and bounded context from the repository layer
- [x] 2.2 Implement structured candidate extraction for user memory and knowledge memory with validated output schema
- [x] 2.3 Implement reconcile actions (`ADD`, `UPDATE`, `DELETE`, `NONE`, `SUPERSEDE`) against existing active memories
- [x] 2.4 Implement lifecycle transitions for superseded, rejected, deleted, and active memory items with evidence preservation
- [x] 2.5 Add retry-safe and idempotent write behavior for memory jobs so repeated execution does not create duplicate active memories

## 3. Retrieval And Prompt Injection Cutover

- [ ] 3.1 Replace keyword-first long-term memory lookup with typed retrieval for active user memory and semantic recall for knowledge memory
- [ ] 3.2 Refactor prompt construction so user memory is injected through policy/system context and knowledge memory through evidence-aware context
- [ ] 3.3 Add controlled fallback behavior for degraded retrieval without keeping the old keyword path as the canonical implementation
- [ ] 3.4 Ensure rejected, deleted, and superseded memories are excluded from default prompt injection

## 4. Legacy Path Retirement

- [ ] 4.1 Remove the old synchronous fragment-writing flow that stores truncated `Q/A` text as canonical long-term memory
- [ ] 4.2 Remove or demote the old keyword-scoring retrieval path so it is no longer the default long-term memory search implementation
- [ ] 4.3 Delete legacy memory wrappers and facade exports that only forward calls without preserving a necessary boundary
- [ ] 4.4 Update call sites to use the new canonical repository/service entrypoints instead of legacy memory helpers

## 5. Verification, Migration, And Documentation

- [x] 5.1 Add unit tests for episode persistence, candidate extraction normalization, reconcile actions, lifecycle transitions, and dedup behavior
- [ ] 5.2 Add integration tests covering async memory job execution, typed retrieval, prompt injection, and legacy-path retirement
- [ ] 5.3 Add migration coverage for existing memory data compatibility and rollback-safe reads during the transition window
- [ ] 5.4 Update README and architecture docs to describe the new async agentic memory pipeline and removed legacy behavior
