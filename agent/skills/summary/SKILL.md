---
name: summary
description: Summarize long context into concise bullets and cite evidence from the source.
---

# Summary Skill

## When to use this skill

Use this skill when:
- User asks to summarize a paper, section, or content
- User wants a concise overview of lengthy text
- User needs key points extracted from documents

## How to summarize

### Step 1: Understand the content
- Read through the full content to get the overall structure
- Identify the main topics and key arguments

### Step 2: Extract key points
- Focus on:
  - Main thesis or central idea
  - Key findings or conclusions
  - Important methods or approaches
  - Significant data or statistics

### Step 3: Structure the summary
- Use bullet points for clarity
- Group related information together
- Keep each point concise (1-2 sentences)

### Step 4: Cite evidence
- For each key conclusion, append at least one canonical evidence tag
- Use the exact format `<evidence>chunk_id|p页码|o起止偏移</evidence>`
- Extract `chunk_id`, `page_no`, `offset_start`, and `offset_end` from `search_document` results
- Do not invent locators or fall back to bracket-style citations

## Output format

```
## Summary

### Main Points
- Point 1 <evidence>doc_a:chunk_12|p4|o100-168</evidence>
- Point 2 <evidence>doc_a:chunk_18|p6|o12-88</evidence>
- ...

### Key Findings
- Finding 1 <evidence>doc_b:chunk_7|p9|o210-320</evidence>
- ...
```

## Tips
- Keep the summary objective
- Don't add personal opinions
- Prioritize factual information
- Include relevant context but stay concise
