---
name: method_compare
description: Compare two methods, techniques, or approaches across objectives, setup, metrics, and trade-offs.
---

# Method Compare Skill

## When to use this skill

Use this skill when:
- User asks to compare two methods or techniques
- User wants to understand differences between approaches
- User needs help choosing between multiple methods
- User asks about pros and cons of different approaches

## How to compare methods

### Step 1: Identify the methods
- Clearly define each method/approach
- Understand the context of comparison

### Step 2: Compare across dimensions

#### Objectives
- What is each method trying to achieve?
- Are the goals comparable?

#### Setup/Requirements
- What resources are needed?
- What is the computational cost?
- What expertise is required?

#### Performance Metrics
- How is success measured?
- What are the benchmark results?
- How do they compare on key metrics?

#### Trade-offs
- What are the advantages of each?
- What are the limitations?
- What are the failure modes?

### Step 3: Consider use cases
- When is each method preferred?
- What factors influence the choice?
- Are there hybrid approaches?

## Output format

Return strict JSON first (no markdown fence), with this schema:

```json
{
  "topic": "Method A vs Method B",
  "columns": ["Dimension", "Method A", "Method B"],
  "rows": [
    {"Dimension": "Objective", "Method A": "...", "Method B": "..."},
    {"Dimension": "Setup", "Method A": "...", "Method B": "..."},
    {"Dimension": "Performance", "Method A": "...", "Method B": "..."},
    {"Dimension": "Trade-offs", "Method A": "...", "Method B": "..."}
  ],
  "recommendation": "..."
}
```

Then optionally add a short plain-text recommendation summary.

If strict JSON cannot be produced, fall back to the following markdown format:

```
## Method Comparison: [Method A] vs [Method B]

### Overview
| Aspect | [Method A] | [Method B] |
|--------|------------|------------|
| Objective | ... | ... |
| Setup | ... | ... |
| Performance | ... | ... |

### Detailed Analysis

#### [Dimension 1]
**Method A:** [Description]
**Method B:** [Description]

#### [Dimension 2]
...

### When to Use Each

**[Method A]:**
- Use case 1
- Use case 2

**[Method B]:**
- Use case 1
- Use case 2

### Recommendation
[Provide guidance based on common scenarios]
```

## Tips
- Be fair and balanced
- Consider the specific context
- Don't overgeneralize
- Acknowledge when comparison is not apples-to-apples
