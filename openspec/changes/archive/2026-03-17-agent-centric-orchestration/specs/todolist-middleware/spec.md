## ADDED Requirements

### Requirement: LangChain TodoListMiddleware Integration
The system SHALL integrate LangChain's TodoListMiddleware to provide todolist management capabilities.

#### Scenario: Middleware registration
- **WHEN** agent is initialized
- **THEN** TodoListMiddleware is registered in the middleware chain

#### Scenario: Todolist tools injection
- **WHEN** agent processes a request
- **THEN** TodoListMiddleware automatically injects todolist management tools

### Requirement: Remove Custom Todo Tools
The system SHALL remove existing custom todo tool implementations.

#### Scenario: Remove write_todo tool
- **WHEN** refactoring local_ops.py
- **THEN** write_todo tool definition and related code are removed

#### Scenario: Remove edit_todo tool
- **WHEN** refactoring local_ops.py
- **THEN** edit_todo tool definition and related code are removed

#### Scenario: Remove todo helper functions
- **WHEN** refactoring local_ops.py
- **THEN** todo-related helper functions are removed (e.g., _normalize_todo_status, _load_todo_store)

### Requirement: Todolist Storage
Todolist data SHALL be managed by LangChain middleware in agent state and persisted via checkpointer.

#### Scenario: Todolist persistence via checkpointer
- **WHEN** agent creates todolist items
- **THEN** they are stored in PlanningState within agent state
- **AND** SqliteSaver checkpointer automatically persists them to database

#### Scenario: Cross-session access
- **WHEN** user returns to the same session (thread_id)
- **THEN** todolist is restored from checkpointer
- **AND** agent can continue tracking tasks

### Requirement: Tool Visibility
Todolist tools SHALL be automatically available without progressive disclosure.

#### Scenario: Tool availability
- **WHEN** agent starts processing
- **THEN** todolist tools are immediately available in the tool list

#### Scenario: No search required
- **WHEN** agent needs todolist functionality
- **THEN** tools are accessible without calling search_tools
