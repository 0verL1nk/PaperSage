---
name: team_mode
description: Multi-agent collaboration mode for complex tasks requiring diverse expertise.
keywords: team, collaboration, spawn, dispatch, coordinate
---

# Team Mode

Complex task requiring multiple roles detected. Switch to team collaboration mode.

## Mode Workflow

### Step 1: Plan Team Structure
Before spawning agents, define your team plan:

1. What sub-tasks need different expertise?
2. Who is the primary reviewer?
3. What are the dependencies between tasks?

### Step 2: Spawn Teammates
Use `spawn_agent` to create team members with specific roles.

```
spawn_agent(
  name="<descriptive_name>",
  role="teammate | reviewer",
  system_prompt="<optional role-specific guidance>"
)
```

Example:
- `spawn_agent(name="researcher", role="teammate", system_prompt="Focus on finding recent papers")`
- `spawn_agent(name="reviewer", role="reviewer", system_prompt="Critically evaluate methodology")`

### Step 3: Dispatch Tasks
Use `send_message` to assign tasks to spawned agents.

```
send_message(agent_id="<agent_id>", message="<task description>")
```

Check agent IDs with `list_agents`.

### Step 4: Collect Results
Use `get_agent_result` to retrieve teammate outputs.

```
get_agent_result(agent_id="<agent_id>")
```

- If agent is busy, wait and retry
- If agent returned result, proceed to next step

### Step 5: Coordinate and Synthesize
- Review all teammate results
- Use reviewer role to critique intermediate outputs
- Synthesize into final answer
- Close agents when done: `close_agent(agent_id="<agent_id>")`

## Available Tools in This Mode

- `spawn_agent` - Create teammate/reviewer agent
- `send_message` - Dispatch task to agent
- `get_agent_result` - Retrieve agent execution result
- `list_agents` - List all active agents
- `close_agent` - Close agent when task complete
- `write_plan` - Create team execution plan
- `write_todos` - Track team sub-tasks
- `search_document` - Search documents (all agents can use)
- `use_skill` - Apply domain skill

## Team Role Guide

| Role | When to Use | Typical Tasks |
|------|-------------|---------------|
| `teammate` | Parallel sub-tasks (research, extraction) | Gather info, run analysis |
| `reviewer` | Quality assurance | Critique methodology, validate conclusions |

## Mode Principles

1. **Plan before spawning** - Know your team structure first
2. **One task per agent** - Don't overload teammates
3. **Handle busy agents** - If agent is busy, wait with `get_agent_result` later
4. **Review before synthesizing** - Use reviewer to catch issues early
5. **Close when done** - Clean up agents to free resources
6. **You are the leader** - Final answer and user communication is your responsibility

## Exit Condition

Mode ends when:
- All teammate tasks completed and reviewed
- Final synthesis is ready
- Agents closed via `close_agent`
