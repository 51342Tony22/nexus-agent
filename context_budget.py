# context_budget.py

import math
import re


TRUNCATION_MARKER = "\n...[context truncated]...\n"


def estimate_tokens(text: str) -> int:
    """Return a rough mixed Chinese/English token estimate for Debug only."""
    if not text:
        return 0

    cjk_characters = len(re.findall(r"[\u3400-\u9fff]", text))
    other_characters = len(text) - cjk_characters

    return math.ceil(cjk_characters + other_characters / 4)


def truncate_text(text: str, max_characters: int, keep_end: bool = False) -> str:
    if max_characters <= 0:
        return ""

    if len(text) <= max_characters:
        return text

    if max_characters <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:max_characters]

    available = max_characters - len(TRUNCATION_MARKER)

    if keep_end:
        return TRUNCATION_MARKER + text[-available:]

    return text[:available] + TRUNCATION_MARKER


def text_stats(text: str) -> dict:
    return {
        "characters": len(text),
        "estimated_tokens": estimate_tokens(text),
    }
