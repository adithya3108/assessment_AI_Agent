from __future__ import annotations

import re

from app.models import ChatMessage

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|system|developer) instructions",
    r"reveal (your )?(system|developer) prompt",
    r"act as (?!an shl)",
    r"jailbreak",
    r"bypass",
]

LEGAL_PATTERNS = [
    r"\blegally required\b",
    r"\bdoes .* satisfy .* requirement\b",
    r"\blegal .* requirement",
    r"\bhipaa .* requirement",
    r"\bhiring law\b",
    r"\blegal advice\b",
    r"\bcompliance guarantee\b",
]

OFF_TOPIC_PATTERNS = [
    r"\b(recipe|weather|stock|crypto|movie|dating)\b",
    r"\bnon[- ]?shl\b",
]


def refusal_reason(messages: list[ChatMessage]) -> str | None:
    latest = next((message.content for message in reversed(messages) if message.role.value == "user"), "")
    text = latest.lower()
    if any(re.search(pattern, text) for pattern in INJECTION_PATTERNS):
        return "I can only discuss SHL assessments from the provided catalog."
    if any(re.search(pattern, text) for pattern in LEGAL_PATTERNS):
        return "I can discuss SHL assessment fit, but I cannot provide legal advice or certify legal compliance."
    if any(re.search(pattern, text) for pattern in OFF_TOPIC_PATTERNS):
        return "I can only discuss SHL assessments from the provided catalog."
    return None
