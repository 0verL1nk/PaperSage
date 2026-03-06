# Output Schema

Use a machine-friendly object in this shape:

```json
{
  "objective": "string",
  "findings": [
    {
      "claim": "string",
      "confidence": "high|medium|low",
      "evidence": [
        {
          "source_type": "document|scholarly|web",
          "source_id": "string",
          "locator": "string",
          "score": 0.0
        }
      ]
    }
  ],
  "conflicts": [
    {
      "topic": "string",
      "claim_a": "string",
      "claim_b": "string",
      "note": "string"
    }
  ],
  "open_questions": ["string"],
  "next_actions": ["string"]
}
```

If strict JSON is not requested, keep the same sections in markdown.
