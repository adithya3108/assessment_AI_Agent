from __future__ import annotations

import math
import logging
import re
from collections import Counter
from pathlib import Path

from app.catalog_enrichment import doc_category, enrich_document
from app.catalog_loader import load_catalog
from app.embeddings import get_embeddings
from app.index_builder import DEFAULT_INDEX_DIR, doc_to_text
from app.models import AssessmentDocument, HiringState
from app.observability import traceable

TOKEN_RE = re.compile(r"[a-zA-Z0-9+#.]+")
logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class HybridRetriever:
    def __init__(self, catalog: list[AssessmentDocument] | None = None, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
        self.catalog = catalog or load_catalog()
        self.index_dir = Path(index_dir)
        self._doc_tokens = [tokenize(doc.searchable_text) for doc in self.catalog]
        self._doc_freq = Counter(token for tokens in self._doc_tokens for token in set(tokens))
        self._avg_len = sum(len(tokens) for tokens in self._doc_tokens) / max(len(self._doc_tokens), 1)
        self._vectorstore = self._load_vectorstore()
        self._bm25_retriever = self._build_langchain_bm25()

    @traceable(name="hybrid_retrieval")
    def retrieve(self, state: HiringState, top_k: int = 20) -> list[AssessmentDocument]:
        query = state.retrieval_query() or " ".join(message for message in state.skills)
        if state.intent.value == "compare" and state.compared_items:
            query = " ".join(state.compared_items + [query])
        logger.info("retrieval_query=%s", query)

        candidates: dict[str, tuple[AssessmentDocument, float]] = {}

        vector_results = self._vector(query, top_k=top_k)
        logger.info("vector_results=%s", [(doc.name, round(score, 3)) for doc, score in vector_results[:10]])
        for doc, score in vector_results:
            candidates[doc.url] = (doc, candidates.get(doc.url, (doc, 0.0))[1] + score)

        bm25_results = self._langchain_bm25(query, top_k=top_k)
        logger.info("bm25_results=%s", [(doc.name, round(score, 3)) for doc, score in bm25_results[:10]])
        for doc, score in bm25_results:
            candidates[doc.url] = (doc, candidates.get(doc.url, (doc, 0.0))[1] + score)

        if not candidates:
            for doc, score in self._bm25(query, top_k=top_k):
                candidates[doc.url] = (doc, candidates.get(doc.url, (doc, 0.0))[1] + score)

        for doc, score in self._lexical_similarity(query, top_k=top_k):
            candidates[doc.url] = (doc, candidates.get(doc.url, (doc, 0.0))[1] + score)

        for doc, score in self._skill_backfill(state):
            candidates[doc.url] = (doc, candidates.get(doc.url, (doc, 0.0))[1] + score)

        for doc, score in self._category_backfill(state):
            candidates[doc.url] = (doc, candidates.get(doc.url, (doc, 0.0))[1] + score)

        docs = [doc for doc, _ in sorted(candidates.values(), key=lambda item: item[1], reverse=True)]
        docs = self._apply_refinement_filters(docs, state)
        logger.info("merged_retrieval_results=%s", [doc.name for doc in docs[:top_k]])
        return docs[:top_k]

    def _load_vectorstore(self):
        if not (self.index_dir / "index.faiss").exists():
            return None
        try:
            from langchain_community.vectorstores import FAISS

            embeddings = get_embeddings()
            return FAISS.load_local(str(self.index_dir), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None

    def _build_langchain_bm25(self):
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain_core.documents import Document

            docs = [
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
                for doc in self.catalog
            ]
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = 20
            return retriever
        except Exception:
            return None

    def _vector(self, query: str, top_k: int) -> list[tuple[AssessmentDocument, float]]:
        if self._vectorstore is None:
            return []
        try:
            results = self._vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
            return [(self._doc_from_metadata(doc.metadata), float(score) * 10.0) for doc, score in results]
        except Exception:
            return []

    def _langchain_bm25(self, query: str, top_k: int) -> list[tuple[AssessmentDocument, float]]:
        if self._bm25_retriever is None:
            return []
        try:
            self._bm25_retriever.k = top_k
            docs = self._bm25_retriever.invoke(query)
            return [(self._doc_from_metadata(doc.metadata), float(top_k - index)) for index, doc in enumerate(docs)]
        except Exception:
            return []

    def _bm25(self, query: str, top_k: int) -> list[tuple[AssessmentDocument, float]]:
        query_tokens = tokenize(query)
        scores: list[tuple[AssessmentDocument, float]] = []
        total_docs = max(len(self.catalog), 1)
        k1 = 1.5
        b = 0.75
        for doc, tokens in zip(self.catalog, self._doc_tokens):
            token_counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                freq = token_counts[token]
                if not freq:
                    continue
                idf = math.log(1 + (total_docs - self._doc_freq[token] + 0.5) / (self._doc_freq[token] + 0.5))
                denom = freq + k1 * (1 - b + b * len(tokens) / max(self._avg_len, 1))
                score += idf * (freq * (k1 + 1)) / denom
            if score:
                scores.append((doc, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

    def _lexical_similarity(self, query: str, top_k: int) -> list[tuple[AssessmentDocument, float]]:
        query_tokens = set(tokenize(query))
        scores = []
        for doc in self.catalog:
            doc_tokens = set(tokenize(doc.searchable_text))
            overlap = query_tokens & doc_tokens
            name_bonus = sum(2 for token in query_tokens if token in tokenize(doc.name))
            score = len(overlap) + name_bonus
            if score:
                scores.append((doc, float(score)))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

    def _doc_from_metadata(self, metadata: dict) -> AssessmentDocument:
        return enrich_document(AssessmentDocument(
            name=metadata.get("name", ""),
            url=metadata.get("url", ""),
            description=metadata.get("description", ""),
            skills=metadata.get("skills", []) or [],
            test_type=metadata.get("test_type", ""),
            duration=metadata.get("duration", ""),
            job_levels=metadata.get("job_levels", []) or [],
            languages=metadata.get("languages", []) or [],
            categories=metadata.get("categories", []) or [],
            raw=metadata,
        ))

    def _category_backfill(self, state: HiringState) -> list[tuple[AssessmentDocument, float]]:
        # Scores are calibrated so explicit requests beat skill_backfill (base 18.0)
        # and also beat noisy BM25/vector candidates.
        wanted: list[tuple[str, float]] = []
        if state.personality_required or state.stakeholder_interaction or state.teamwork_required:
            # Bump to 26.0 when explicitly requested so OPQ32r beats retrieval noise.
            base = 26.0 if state.personality_required else 14.0
            wanted.append(("personality", base))
        elif state.personality_required is None and state.seniority in {"senior"}:
            # For senior IC hires where personality has NOT been explicitly declined,
            # include OPQ32r-family docs as a low-priority default.
            wanted.append(("personality", 10.0))
        if state.cognitive_required:
            # Bump to 24.0 when explicitly requested.
            wanted.append(("cognitive", 24.0))
        if state.communication_required:
            wanted.append(("communication", 14.0))
        if state.situational_required:
            wanted.append(("teamwork", 15.0))

        results: list[tuple[AssessmentDocument, float]] = []
        for category, base_score in wanted:
            matches = [doc for doc in self.catalog if doc_category(doc, category)]
            # Use top 10 to give the reranker more personality/cognitive options.
            for index, doc in enumerate(matches[:10]):
                results.append((doc, base_score - index))

        # Full-battery boost: when personality + cognitive + situational are ALL
        # explicitly requested, pin OPQ32r-family and Verify G+ at the very top
        # so they survive even when noisy BM25 items score high.
        full_battery = (
            state.personality_required and state.cognitive_required and state.situational_required
        )
        if full_battery:
            opq_docs = [doc for doc in self.catalog if "opq" in doc.name.lower()]
            verify_docs = [doc for doc in self.catalog if "verify" in doc.name.lower() and ("g+" in doc.name.lower() or "interactive g+" in doc.name.lower())]
            grad_scenario_docs = [doc for doc in self.catalog if "graduate scenarios" in doc.name.lower()]
            for i, doc in enumerate(opq_docs[:3]):
                results.append((doc, 38.0 - i))
            for i, doc in enumerate(verify_docs[:3]):
                results.append((doc, 36.0 - i))
            for i, doc in enumerate(grad_scenario_docs[:3]):
                results.append((doc, 34.0 - i))
            logger.info("full_battery_boost applied")

        if results:
            logger.info("category_backfill=%s", [(doc.name, round(score, 2)) for doc, score in results])
        return results

    def _skill_backfill(self, state: HiringState) -> list[tuple[AssessmentDocument, float]]:
        ignored = {"communication", "teamwork", "leadership", "customer service", "safety", "sales"}
        skill_terms = [skill for skill in state.skills if skill not in ignored]
        mid_or_senior = state.seniority in {"mid", "senior"}
        results: list[tuple[AssessmentDocument, float]] = []
        for skill in skill_terms:
            matches = [
                doc
                for doc in self.catalog
                if skill in doc.searchable_text and (doc_category(doc, "technical") or "K" in doc.test_type)
            ]
            # Sort descending: prefer non-entry, then advanced variants, then skill in name.
            matches.sort(
                key=lambda doc: (
                    0 if (mid_or_senior and is_entry_name(doc.name)) else 1,
                    1 if (mid_or_senior and "advanced" in doc.name.lower()) else 0,
                    1 if skill in doc.name.lower() else 0,
                ),
                reverse=True,
            )
            for index, doc in enumerate(matches[:8]):
                results.append((doc, 18.0 - index))
        if results:
            logger.info("skill_backfill=%s", [(doc.name, round(score, 2)) for doc, score in results])
        return results

    def _apply_refinement_filters(self, docs: list[AssessmentDocument], state: HiringState) -> list[AssessmentDocument]:
        excluded = " ".join(state.exclude_terms).lower()
        if excluded:
            docs = [doc for doc in docs if not any(term in doc.searchable_text for term in tokenize(excluded))]
        if state.previous_recommendations and any(term in " ".join(state.include_terms).lower() for term in ["keep", "same", "shortlist"]):
            previous = {name.lower() for name in state.previous_recommendations}
            ordered = [doc for doc in self.catalog if doc.name.lower() in previous]
            docs = ordered + [doc for doc in docs if doc.name.lower() not in previous]

        # Domain-noise suppression: demote retail/sales simulations when the
        # context is clearly non-sales — triggered by technical skills OR by
        # an explicit non-sales intent (personality/cognitive/situational without
        # any sales signal).
        technical_skills = set(state.skills) - {"communication", "teamwork", "leadership", "customer service", "safety", "sales"}
        sales_role = "sales" in state.skills or any(
            term in (state.role or "").lower() for term in ["sales", "retail", "customer service"]
        )
        suppress_noise = bool(technical_skills) or (
            not sales_role and (state.personality_required or state.cognitive_required or state.situational_required)
        )
        if suppress_noise:
            _noise_names = {
                "retail sales and service simulation",
                "sales & service phone simulation",
                "sales & service phone solution",
                "retail sales and service solution",
                "entry level sales solution",
                "entry level cashier solution",
            }
            core = [doc for doc in docs if doc.name.lower() not in _noise_names]
            noise = [doc for doc in docs if doc.name.lower() in _noise_names]
            docs = core + noise  # push noise to end, don't remove entirely

        return docs


def is_entry_name(name: str) -> bool:
    lowered = name.lower()
    return "entry" in lowered or "basic" in lowered or "fundamental" in lowered


def build_faiss_index() -> int:
    from app.index_builder import build_faiss_index as build_index

    return build_index()
