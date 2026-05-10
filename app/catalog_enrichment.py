from __future__ import annotations

import re

from app.models import AssessmentDocument


CATEGORY_RULES: dict[str, list[str]] = {
    "technical": [
        "java",
        "spring",
        "sql",
        "aws",
        "docker",
        "linux",
        "network",
        "programming",
        "coding",
        "excel",
        "word",
        "hipaa",
        "accounting",
        "statistics",
    ],
    "personality": [
        "opq",
        "personality",
        "behavior",
        "behaviour",
        "work style",
        "dependability",
        "safety instrument",  # DSI / Safety & Dependability — not generic "safety"
        "motivation",
        "mq",
    ],
    "cognitive": [
        "verify",
        "g+",
        "cognitive",
        "ability",
        "aptitude",
        "reasoning",
        "numerical",
        "deductive",
        "inductive",
    ],
    "communication": [
        "communication",
        "spoken",
        "language",
        "contact center",
        "contact centre",
        "call",
        "customer service",
        "phone",
        "stakeholder",
    ],
    "teamwork": [
        "teamwork",
        "collaboration",
        "collaborative",
        "competenc",
        "workplace",
        "scenarios",
        "situational",
    ],
    "leadership": [
        "leadership",
        "leader",
        "manager",
        "executive",
        "director",
        "cxo",
        "influencing",
        "strategic",
        "hipo",
        "high potential",
        "360",
        "multi-rater",
        "mfs",
    ],
}

ENTRY_TERMS = {"entry", "beginner", "basic", "fundamental", "graduate", "early career"}


def enrich_document(doc: AssessmentDocument) -> AssessmentDocument:
    categories = set(doc.categories)
    text = doc.searchable_text
    for category, terms in CATEGORY_RULES.items():
        if any(term in text for term in terms):
            categories.add(category)
    if "P" in doc.test_type:
        categories.add("personality")
    if "A" in doc.test_type:
        categories.add("cognitive")
    if "S" in doc.test_type:
        categories.add("communication")
    if "B" in doc.test_type or "C" in doc.test_type:
        categories.add("teamwork")
    if "K" in doc.test_type:
        categories.add("technical")
    doc.categories = sorted(categories)
    return doc


def doc_category(doc: AssessmentDocument, category: str) -> bool:
    return category in set(doc.categories)


def duplicate_key(doc: AssessmentDocument) -> str:
    name = doc.name.lower()
    if "java" in name:
        return "java"
    if "opq" in name:
        return "opq"
    if "excel" in name:
        return "excel"
    if "word" in name:
        return "word"
    if "contact center" in name or "customer service phone" in name:
        return "contact-center-simulation"
    cleaned = re.sub(r"\b(new|advanced level|entry level|general|essentials|365)\b", "", name)
    return re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")[:48]


def is_entry_level(doc: AssessmentDocument) -> bool:
    text = f"{doc.name} {' '.join(doc.job_levels)} {doc.description}".lower()
    return any(term in text for term in ENTRY_TERMS)
