#!/usr/bin/env python3
"""Deterministic source quality score helper."""

from __future__ import annotations


def compute_score(
    credibility: float,
    relevance: float,
    recency: float,
    consistency: float,
) -> float:
    for value in (credibility, relevance, recency, consistency):
        if value < 0 or value > 1:
            raise ValueError("all inputs must be within [0, 1]")
    score = 0.35 * credibility + 0.35 * relevance + 0.15 * recency + 0.15 * consistency
    return round(score, 4)


if __name__ == "__main__":
    print(compute_score(0.9, 0.8, 0.7, 0.9))
