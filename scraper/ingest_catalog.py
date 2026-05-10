from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_environment
from app.catalog_loader import write_normalized_catalog

DEFAULT_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"


def fetch_catalog(url: str) -> list[dict]:
    with urlopen(url, timeout=20) as response:
        raw = response.read().decode("utf-8", errors="replace")
        data = parse_catalog_json(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("products", "assessments", "data", "catalog"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Could not find a catalog item list in the downloaded JSON.")


def parse_catalog_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = strip_invalid_json_control_chars(raw)
        return json.loads(cleaned, strict=False)


def strip_invalid_json_control_chars(raw: str) -> str:
    # JSON permits tab, LF, and CR as whitespace. Other ASCII control chars
    # sometimes appear in scraped catalog descriptions and must be removed.
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description="Download and normalize SHL catalog JSON.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=str(ROOT / "app" / "catalog" / "normalized_catalog.json"))
    args = parser.parse_args()
    docs = write_normalized_catalog(fetch_catalog(args.url), args.output)
    print(f"Wrote {len(docs)} normalized catalog items to {args.output}")


if __name__ == "__main__":
    main()
