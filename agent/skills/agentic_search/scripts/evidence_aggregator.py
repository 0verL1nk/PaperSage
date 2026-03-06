#!/usr/bin/env python3
"""Normalize heterogeneous evidence payloads into a flat list."""

from __future__ import annotations

from typing import Any


def aggregate_evidence(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        source_type = str(payload.get("source_type") or "unknown")
        items = payload.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            record = {
                "source_type": source_type,
                "source_id": str(item.get("source_id") or ""),
                "text": str(item.get("text") or ""),
                "locator": str(item.get("locator") or ""),
                "score": float(item.get("score", 0.0)) if isinstance(item.get("score"), (int, float)) else 0.0,
            }
            if not record["text"].strip():
                continue
            records.append(record)
    return records


if __name__ == "__main__":
    # minimal self-check
    sample = [{"source_type": "document", "items": [{"source_id": "d1", "text": "x", "score": 0.8}]}]
    print(aggregate_evidence(sample))
