from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_environment
from app.index_builder import build_faiss_index


if __name__ == "__main__":
    load_environment()
    count = build_faiss_index()
    print(f"Built configured FAISS embedding index for {count} catalog items.")
