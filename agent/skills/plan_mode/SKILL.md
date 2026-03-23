---
name: plan_mode
description: Structured planning mode for complex multi-step tasks.
keywords: plan, todo, strategy, research, investigation
---

# Plan Mode

Complex task detected. Switch to structured planning mode.

## Mode Workflow

### Step 1: Create Plan
Use `write_plan` to define your execution strategy.

```
write_plan(goal="<task goal>", description="<strategy in steps>")
```

Example:
- goal: "Research and compare LLM frameworks"
- description: "1. Search for papers on LLM frameworks\n2. Extract key features\n3. Compare in table format\n4. Write summary"

### Step 2: Create Todo List
Use `write_todos` to track all sub-tasks with dependencies.

```
write_todos(todos=[
  Todo(id="step1", content="Search for LLM framework papers", status="pending"),
  Todo(id="step2", content="Extract key features from results", status="pending", depends_on=["step1"]),
  Todo(id="step3", content="Compare features in table", status="pending", depends_on=["step2"]),
])
```

### Step 3: Execute by Todo
System returns `scheduler_hints` with ready todo IDs. Execute in order:

1. Pick a `ready` todo
2. Use `search_document` / `search_papers` / `search_web` to gather info
3. Mark todo `completed` by re-calling `write_todos` with updated status

### Step 4: Verify Progress
Use `read_plan` to review your strategy at any time.

### Step 5: Finalize
After all todos completed:
- Compile findings into final answer
- Ensure every claim has evidence tags
- Conclude with confidence level

## Available Tools in This Mode

- `write_plan` - Create/update execution plan
- `read_plan` - Read current plan
- `write_todos` - Manage todo list with dependencies
- `search_document` - Search bound project documents
- `search_papers` - Search academic papers
- `search_web` - Web search
- `use_skill` - Apply domain skill (summary, critical_reading, etc.)

## Mode Principles

1. **Plan first, act second** - Never jump into sub-tasks without a plan
2. **Track progress** - Every sub-task should be a todo
3. **Respect dependencies** - Only execute `ready` todos
4. **Cite evidence** - Every conclusion needs evidence tag
5. **Exit when done** - Mode ends when all todos completed or user confirms done
