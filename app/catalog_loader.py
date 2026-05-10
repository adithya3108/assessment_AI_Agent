from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.catalog_enrichment import enrich_document
from app.models import AssessmentDocument


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT / "app" / "catalog" / "seed_catalog.json"
NORMALIZED_CATALOG_PATH = ROOT / "app" / "catalog" / "normalized_catalog.json"


def normalize_catalog_item(item: dict[str, Any]) -> AssessmentDocument:
    raw_item = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    merged = raw_item | item
    name = first_present(merged, "name", "Name", "title", "assessment_name", "productName")
    url = first_present(merged, "url", "URL", "productUrl", "product_url", "link")
    description = first_present(merged, "description", "Description", "shortDescription", "summary")
    test_type = first_present(merged, "test_type", "testType", "test_type_code", "type", "assessmentType")
    duration = first_present(merged, "duration", "Duration", "time", "assessmentLength")
    keys = list_value(first_present(merged, "keys", "Keys", "assessmentKeys"))
    skills = list_value(first_present(merged, "skills", "Skills", "competencies", "keywords")) + keys
    levels = list_value(first_present(merged, "job_levels", "jobLevels", "levels", "jobLevel"))
    languages = list_value(first_present(merged, "languages", "Languages", "language"))
    categories = list_value(first_present(merged, "categories", "Categories", "category")) + keys
    test_type = str(test_type or derive_test_type(keys)).strip()

    return enrich_document(AssessmentDocument(
        name=str(name or "").strip(),
        url=str(url or "").strip(),
        description=str(description or "").strip(),
        skills=[str(skill).strip().lower() for skill in skills if str(skill).strip()],
        test_type=str(test_type or "").strip(),
        duration=str(duration or "").strip(),
        job_levels=[str(level).strip().lower() for level in levels if str(level).strip()],
        languages=[str(language).strip() for language in languages if str(language).strip()],
        categories=[str(category).strip().lower() for category in categories if str(category).strip()],
        raw=raw_item or item,
    ))


def load_catalog(path: str | Path | None = None) -> list[AssessmentDocument]:
    selected_path = Path(path or os.getenv("SHL_CATALOG_PATH") or NORMALIZED_CATALOG_PATH)
    if not selected_path.exists():
        selected_path = DEFAULT_CATALOG_PATH
    data = json.loads(selected_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("products") or data.get("assessments") or data.get("data") or []
    docs = [normalize_catalog_item(item) for item in data if isinstance(item, dict)]
    return [doc for doc in docs if doc.name and doc.url]


def write_normalized_catalog(raw_items: list[dict[str, Any]], path: str | Path = NORMALIZED_CATALOG_PATH) -> list[AssessmentDocument]:
    docs = [normalize_catalog_item(item) for item in raw_items if isinstance(item, dict)]
    payload = [doc.model_dump(exclude={"raw"}) | {"raw": doc.raw} for doc in docs if doc.name and doc.url]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return [AssessmentDocument.model_validate(item) for item in payload]


def first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.replace("|", ",").split(",") if part.strip()]
    return [value]


def derive_test_type(keys: list[Any]) -> str:
    mapping = {
        "Ability & Aptitude": "A",
        "Assessment Exercises": "E",
        "Biodata & Situational Judgment": "B",
        "Competencies": "C",
        "Development & 360": "D",
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Simulations": "S",
    }
    codes = [mapping[str(key).strip()] for key in keys if str(key).strip() in mapping]
    return ",".join(dict.fromkeys(codes))
