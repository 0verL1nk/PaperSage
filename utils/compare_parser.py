import csv
import io
import json
from typing import Any

from .utils import extract_json_string


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


def method_compare_to_csv(payload: dict[str, Any]) -> str:
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[str(col) for col in columns])
    writer.writeheader()
    for row in rows:
        if not isinstance(row, dict):
            continue
        writer.writerow({col: row.get(col, "") for col in columns})
    return output.getvalue()
