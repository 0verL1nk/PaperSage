---
name: agentic_search
description: Run a local-first research workflow by planning sub-queries, iterating retrieval, validating sources, and producing traceable evidence-backed conclusions.
metadata:
  level: advanced
---

# Agentic Search Skill

## When To Use

Use this skill when the user asks for:
- multi-step research instead of a single lookup
- external verification for incomplete document evidence
- evidence-backed conclusions with explicit source traceability
- gap-filling loops (find missing facts, then re-search)

Do not use this skill for:
- direct extraction from a clearly scoped local document where one retrieval pass is enough
- purely stylistic rewriting tasks

## Workflow

1. Define research objective and output contract
- Convert user request into explicit deliverables: key question, expected depth, and evidence contract.
- For project-document evidence, the contract is `<evidence>chunk_id|p页码|o起止偏移</evidence>`.

2. Build query plan
- Split into sub-queries by intent:
  - factual lookup
  - method comparison
  - recency update
  - risk/limitation check
- Prefer local document retrieval first.

3. Execute iterative retrieval
- Run local/document search first.
- If local evidence is insufficient, expand to scholarly search and then web search.
- Track each round with: query, source type, key hit, confidence.

4. Evaluate source quality
- Score sources by credibility, relevance, recency, and cross-source consistency.
- Downgrade claims that rely on a single weak source.

5. Synthesize answer with traceability
- Produce structured findings with explicit evidence anchors.
- When citing project-document evidence in prose, use `<evidence>chunk_id|p页码|o起止偏移</evidence>` instead of bracket citations or free-form page references.
- Separate confirmed facts, probable inferences, and unresolved gaps.

## Runtime Contract

- Return conclusions that are evidence-backed and audit-friendly.
- For each key conclusion, attach at least one source anchor.
- For current-project document anchors, use the exact `<evidence>chunk_id|p页码|o起止偏移</evidence>` format.
- If evidence conflicts, explain conflict instead of forcing one conclusion.
- If evidence is still insufficient, output explicit follow-up search actions.

## Progressive References

Read additional references only when needed:
- `references/workflow_blueprint.md`: full multi-round execution blueprint
- `references/source_quality_rubric.md`: scoring and conflict-resolution rubric
- `references/output_schema.md`: final structured output schema

## Scripted Helpers

- `scripts/evidence_aggregator.py`: normalize and merge heterogeneous evidence records
- `scripts/source_score.py`: deterministic source scoring utility
