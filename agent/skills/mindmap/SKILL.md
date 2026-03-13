---
name: mindmap
description: Generate strict JSON mind maps from document evidence with clear hierarchy and concise node labels.
---

# Mindmap Skill

## When to use this skill

Use this skill when:
- User asks for a mind map, concept map, or knowledge structure
- User wants hierarchical decomposition of a paper
- Output must be machine-parseable JSON for visualization

## How to build the mind map

### Step 1: Ground in evidence
- Retrieve evidence from the current document before drafting nodes
- Use chapter names, section headers, or repeated key terms as candidates
- Do not invent concepts not supported by the document

### Step 2: Build hierarchy
- Root node: one concise theme for the whole paper
- First level: 3-6 major branches (problem, method, experiment, results, limitations, outlook)
- Second level: 2-4 concrete points per branch
- Keep depth between 2 and 4 levels

### Step 3: Keep labels concise
- Use short noun phrases for node names
- Avoid full sentences when possible
- Merge duplicated or overlapping branches

### Step 4: Output format
- Use **<mindmap> JSON </mindmap>** wrapper - NO markdown fences
- Output pure JSON only, no explanations before or after

## Output contract

Wrap JSON with `<mindmap>` tags (no markdown fences):

<mindmap>
{
  "name": "主题",
  "children": [
    {
      "name": "子主题",
      "children": [
        {"name": "要点1", "children": []},
        {"name": "要点2", "children": []}
      ]
    }
  ]
}
</mindmap>

## Quality checks

- Ensure branch coverage is balanced (no single branch dominates all details)
- Ensure sibling nodes are parallel in granularity
- If evidence is insufficient, include a minimal node such as "信息不足" instead of guessing
