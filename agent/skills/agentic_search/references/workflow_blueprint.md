# Workflow Blueprint

## Phase 1: Scope

- Input: user research request
- Output:
  - objective statement
  - depth level (quick / standard / deep)
  - evidence granularity target
  - citation contract for project-document evidence: `<evidence>chunk_id|p页码|o起止偏移</evidence>`

## Phase 2: Plan

- Produce 3-8 sub-queries.
- Tag each sub-query with one primary source lane:
  - `document`
  - `scholarly`
  - `web`

## Phase 3: Retrieve (iterative)

Round loop:
1. Execute selected sub-queries.
2. Extract candidate facts and evidence anchors.
   - For project-document evidence, preserve `chunk_id`, `page_no`, `offset_start`, and `offset_end` so the final answer can emit canonical `<evidence>...</evidence>` tags.
3. Evaluate whether gaps remain.
4. If gaps remain and budget exists, generate next-round queries.

Stop conditions:
- required claims all evidence-backed
- or search budget exhausted

## Phase 4: Consolidate

- Deduplicate overlapping facts.
- Resolve contradictions using source quality rubric.
- Keep unresolved points explicit.

## Phase 5: Deliver

Return:
- findings
- evidence map
- uncertainty list
- recommended next actions

If the final answer is prose rather than JSON, each document-grounded key claim should carry at least one `<evidence>chunk_id|p页码|o起止偏移</evidence>` tag.
