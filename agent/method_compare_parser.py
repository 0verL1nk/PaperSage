import json
import re
from typing import Any


def _extract_first_complete_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            _candidate, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return text[idx : idx + end]
    raise ValueError("json block not found")


def _extract_json_from_tagged_block(text: str, tag_name: str) -> str | None:
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        candidate = str(match.group(1) or "").strip()
        if not candidate:
            continue
        try:
            return _extract_first_complete_json_object(candidate)
        except ValueError:
            continue
    return None


def extract_json_string(text: str) -> str:
    """Extract JSON from text, supporting both <mindmap> tags and raw JSON."""
    if not isinstance(text, str):
        raise ValueError("input must be string")

    mindmap_payload = _extract_json_from_tagged_block(text, "mindmap")
    if mindmap_payload is not None:
        return mindmap_payload

    json_payload = _extract_json_from_tagged_block(text, "json")
    if json_payload is not None:
        return json_payload

    return _extract_first_complete_json_object(text)


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
