from __future__ import annotations


def detect_language(text: str) -> str:
    """Return coarse language label: zh/en/other."""
    chinese_chars = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
    english_chars = len([c for c in text if c.isascii() and c.isalpha()])

    total_chars = len(text.strip())
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    english_ratio = english_chars / total_chars if total_chars > 0 else 0

    if chinese_ratio > 0.3:
        return "zh"
    if english_ratio > 0.5:
        return "en"
    return "other"
