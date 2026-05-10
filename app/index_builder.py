from __future__ import annotations

import json
from pathlib import Path

from app.catalog_loader import load_catalog
from app.embeddings import get_embeddings
from app.models import AssessmentDocument

DEFAULT_INDEX_DIR = Path("vectorstore/faiss_index")


def doc_to_text(doc: AssessmentDocument) -> str:
    # Include raw.keys (e.g. "Personality & Behavior", "Ability & Aptitude") so the
    # embedding model can distinguish assessment families even when test_type is empty.
    raw_keys = doc.raw.get("keys", []) if doc.raw else []
    parts = [
        f"Name: {doc.name}",
        f"Description: {doc.description}",
        f"Assessment Family: {', '.join(raw_keys)}",
        f"Skills: {', '.join(doc.skills)}",
        f"Test Type: {doc.test_type}",
        f"Duration: {doc.duration}",
        f"Job Levels: {', '.join(doc.job_levels)}",
        f"Languages: {', '.join(doc.languages)}",
        f"Categories: {', '.join(doc.categories)}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


def build_faiss_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> int:
    """Build a persisted LangChain FAISS index using configured embeddings.

    Default provider is OpenRouter with openai/text-embedding-3-small. Set
    EMBEDDING_PROVIDER=openai to use direct OpenAI instead.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    docs = load_catalog()
    lc_docs = [
        Document(
            page_content=doc_to_text(doc),
            metadata={
                "name": doc.name,
                "url": doc.url,
                "description": doc.description,
                "skills": doc.skills,
                "test_type": doc.test_type,
                "duration": doc.duration,
                "job_levels": doc.job_levels,
                "languages": doc.languages,
                "categories": doc.categories,
            },
        )
        for doc in docs
    ]
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    target = Path(index_dir)
    target.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target))
    (target / "catalog_metadata.json").write_text(
        json.dumps([doc.model_dump() for doc in docs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(docs)


def build_openai_faiss_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> int:
    return build_faiss_index(index_dir)


if __name__ == "__main__":
    count = build_faiss_index()
    print(f"Built FAISS index for {count} SHL catalog documents.")
