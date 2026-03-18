import json
import re
from typing import Any


def extract_json_string(text: str) -> str:
    """Extract JSON from text, supporting both <mindmap> tags and raw JSON."""
    if not isinstance(text, str):
        raise ValueError("input must be string")

    # Try <mindmap> tags first (higher priority) - use greedy match for nested JSON
    mindmap_match = re.search(r"<mindmap>\s*(\{.*\})\s*</mindmap>", text, re.DOTALL)
    if mindmap_match:
        return mindmap_match.group(1)

    # Try <json> tags - use greedy match for nested JSON
    json_match = re.search(r"<json>\s*(\{.*\})\s*</json>", text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Fallback: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("json block not found")
    return text[start : end + 1]


def parse_method_compare_payload(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        payload = json.loads(extract_json_string(text))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None

    columns = payload.get("columns")
    if isinstance(columns, list) and all(isinstance(col, str) and col for col in columns):
        normalized_columns = columns
    else:
        first_row = rows[0]
        normalized_columns = [str(key) for key in first_row.keys()]
        if not normalized_columns:
            return None

    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_row: dict[str, str] = {}
        for column in normalized_columns:
            value = row.get(column, "")
            normalized_row[column] = str(value) if value is not None else ""
        normalized_rows.append(normalized_row)

    topic = payload.get("topic")
    recommendation = payload.get("recommendation")
    return {
        "topic": topic if isinstance(topic, str) else "",
        "columns": normalized_columns,
        "rows": normalized_rows,
        "recommendation": recommendation if isinstance(recommendation, str) else "",
    }
