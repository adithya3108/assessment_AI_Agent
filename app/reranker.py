from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from app.catalog_enrichment import doc_category, duplicate_key, is_entry_level
from app.models import AssessmentDocument, HiringState
from app.observability import traceable
from app.retrieval import tokenize

logger = logging.getLogger(__name__)


@dataclass
class ScoredDocument:
    doc: AssessmentDocument
    score: float
    rationale: list[str]


class Reranker:
    """BAAI/bge-reranker-base reranker with a deterministic fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self._model = self._load_model()

    @traceable(name="diversity_aware_rerank")
    def rerank(self, state: HiringState, docs: list[AssessmentDocument], top_k: int = 5) -> list[AssessmentDocument]:
        if not docs:
            return []
        semantic_scores = self._semantic_scores(state, docs)
        scored = [self._score_doc(state, doc, index, semantic_scores.get(doc.url, 0.0)) for index, doc in enumerate(docs)]
        scored.sort(key=lambda item: item.score, reverse=True)
        logger.info(
            "reranker_scores=%s",
            [(item.doc.name, round(item.score, 3), item.rationale) for item in scored[:12]],
        )
        selected = self._compose_balanced_shortlist(state, scored, top_k)
        logger.info("final_recommendations=%s", [doc.name for doc in selected])
        return selected

    def _load_model(self):
        if os.getenv("DISABLE_BGE_RERANKER", "").lower() == "true":
            return None
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder(self.model_name)
        except Exception:
            return None

    def _semantic_scores(self, state: HiringState, docs: list[AssessmentDocument]) -> dict[str, float]:
        if self._model is None:
            return {}
        query = state.retrieval_query()
        pairs = [(query, self._doc_summary(doc)) for doc in docs]
        try:
            scores = self._model.predict(pairs)
            return {doc.url: float(score) for doc, score in zip(docs, scores)}
        except Exception:
            return {}

    def _doc_summary(self, doc: AssessmentDocument) -> str:
        return " ".join(
            part
            for part in [
                doc.name,
                doc.description,
                " ".join(doc.skills),
                doc.test_type,
                doc.duration,
                " ".join(doc.job_levels),
            ]
            if part
        )

    def _score_doc(self, state: HiringState, doc: AssessmentDocument, index: int, semantic_score: float) -> ScoredDocument:
        query_tokens = set(tokenize(state.retrieval_query()))
        include_tokens = set(tokenize(" ".join(state.include_terms)))
        skill_tokens = set(tokenize(" ".join(state.skills)))
        compared = [item.lower() for item in state.compared_items]
        technical_skills = {
            "java",
            "spring",
            "sql",
            "aws",
            "docker",
            "rest",
            "angular",
            "rust",
            "linux",
            "networking",
            "excel",
            "word",
            "hipaa",
            "finance",
            "statistics",
        }
        technical_query = bool(skill_tokens & technical_skills)
        doc_text = doc.searchable_text
        doc_tokens = set(tokenize(doc_text))
        doc_skill_tokens = set(tokenize(" ".join(doc.skills)))
        rationale: list[str] = []
        score = semantic_score * 2.0
        if semantic_score:
            rationale.append(f"semantic={semantic_score:.2f}")
        score += 1.0 / (index + 1)
        overlap = len(query_tokens & doc_tokens)
        if overlap:
            score += 1.5 * overlap
            rationale.append(f"query_overlap={overlap}")
        skill_overlap = len(skill_tokens & doc_skill_tokens)
        if skill_overlap:
            score += 7.0 * skill_overlap
            rationale.append(f"skill_overlap={skill_overlap}")
        include_overlap = len(include_tokens & doc_tokens)
        if include_overlap:
            score += 3.0 * include_overlap
            rationale.append(f"include_overlap={include_overlap}")
        if technical_query and doc_category(doc, "technical"):
            score += 4.0
            rationale.append("technical_fit")
        # Simulations (job-sample tests) can be broadly categorized as personality
        # or cognitive due to keyword overlap, but they should not receive these
        # category bonuses — they are communication/teamwork tools, not instruments.
        is_simulation = "S" in doc.test_type and not doc_category(doc, "personality") or any(
            term in doc.name.lower() for term in ["simulation", "phone solution", "phone call", "call simulation"]
        )
        if state.personality_required and doc_category(doc, "personality") and not is_simulation:
            score += 12.0
            rationale.append("personality_required")
        if state.cognitive_required and doc_category(doc, "cognitive") and not is_simulation:
            score += 10.0
            rationale.append("cognitive_required")
        if state.situational_required and not is_simulation and (doc_category(doc, "teamwork") or "B" in doc.test_type):
            score += 8.0
            rationale.append("situational_required")
        if state.stakeholder_interaction and (doc_category(doc, "communication") or doc_category(doc, "personality") or doc_category(doc, "leadership")):
            score += 7.0
            rationale.append("stakeholder_fit")
        if state.communication_required and doc_category(doc, "communication"):
            score += 7.0
            rationale.append("communication_fit")
        if state.teamwork_required and (doc_category(doc, "teamwork") or doc_category(doc, "personality")):
            score += 6.0
            rationale.append("teamwork_fit")
        if technical_query and doc_category(doc, "personality") and not state.personality_required and not state.stakeholder_interaction:
            score -= 2.0
            rationale.append("personality_not_requested")

        # Prefer "Advanced Level" variants when hiring mid or senior.
        if state.seniority in {"mid", "senior"} and "advanced" in doc.name.lower():
            score += 5.0
            rationale.append("advanced_level_bonus")

        # Penalize entry-level docs for mid/senior hires.
        if state.seniority in {"mid", "senior"} and is_entry_level(doc):
            score -= 8.0
            rationale.append("seniority_penalty")

        if state.seniority == "entry" and any(level in " ".join(doc.job_levels) for level in ["senior", "executive"]):
            score -= 4.0
            rationale.append("entry_role_penalty")

        # Penalize leadership / HiPo / executive docs when there is no leadership
        # signal in the hiring context. Seniority alone (e.g. senior engineer) is
        # NOT a leadership signal — we require an explicit role title or skill.
        # Use word-boundary regex so "leads design" (verb) doesn't match "lead" (title).
        _LEADERSHIP_TITLE_RE = re.compile(
            r"\b(team lead|tech lead|engineering lead|manager|director|cxo|"
            r"executive|vp|management|leadership)\b",
            re.IGNORECASE,
        )
        leadership_signal = (
            state.stakeholder_interaction
            or "leadership" in state.skills
            or bool(_LEADERSHIP_TITLE_RE.search(state.role or ""))
        )
        if doc_category(doc, "leadership") and not leadership_signal:
            # Stronger penalty for technical roles — they produce the most noise.
            score -= 12.0 if technical_query else 7.0
            rationale.append("leadership_not_relevant")

        if compared and any(item in doc.name.lower() or item in doc_text for item in compared):
            score += 12.0
            rationale.append("comparison_target")
        return ScoredDocument(doc=doc, score=score, rationale=rationale)

    def _compose_balanced_shortlist(self, state: HiringState, scored: list[ScoredDocument], top_k: int) -> list[AssessmentDocument]:
        selected: list[ScoredDocument] = []
        used_duplicates: set[str] = set()

        def add_best(category: str, count: int = 1) -> None:
            for _ in range(count):
                candidate = self._best_candidate(state, scored, selected, used_duplicates, category)
                if candidate:
                    selected.append(candidate)
                    used_duplicates.add(duplicate_key(candidate.doc))

        if state.intent.value == "compare":
            for candidate in scored:
                if len(selected) >= top_k:
                    break
                selected.append(candidate)
            return [item.doc for item in selected[:top_k]]

        if state.skills:
            add_best("technical", 2)
        if state.personality_required or state.stakeholder_interaction or state.teamwork_required:
            add_best("personality", 2 if state.personality_required else 1)
        if state.cognitive_required:
            add_best("cognitive", 1)
        if state.communication_required:
            add_best("communication", 1)
        if state.situational_required:
            add_best("teamwork", 1)

        for candidate in scored:
            if len(selected) >= top_k:
                break
            if candidate in selected:
                continue
            key = duplicate_key(candidate.doc)
            if key in used_duplicates and not self._allow_second_from_group(candidate.doc, selected):
                logger.info("diversity_penalty_skipped=%s duplicate_key=%s", candidate.doc.name, key)
                continue
            selected.append(candidate)
            used_duplicates.add(key)

        return [item.doc for item in selected[:top_k]]

    def _best_candidate(
        self,
        state: HiringState,
        scored: list[ScoredDocument],
        selected: list[ScoredDocument],
        used_duplicates: set[str],
        category: str,
    ) -> ScoredDocument | None:
        technical_state_skills = [
            skill
            for skill in state.skills
            if skill
            not in {
                "communication",
                "teamwork",
                "leadership",
                "customer service",
                "safety",
                "sales",
            }
        ]
        skill_tokens = set(tokenize(" ".join(technical_state_skills)))
        strict_skill_match = category == "technical" and bool(skill_tokens)
        fallback: ScoredDocument | None = None
        for candidate in scored:
            if candidate in selected:
                continue
            key = duplicate_key(candidate.doc)
            if key in used_duplicates and not self._allow_second_from_group(candidate.doc, selected):
                continue
            if doc_category(candidate.doc, category):
                if strict_skill_match:
                    doc_tokens = set(tokenize(candidate.doc.searchable_text))
                    if skill_tokens & doc_tokens:
                        return candidate
                    fallback = fallback or candidate
                    continue
                return candidate
        return fallback if not strict_skill_match else None

    def _allow_second_from_group(self, doc: AssessmentDocument, selected: list[ScoredDocument]) -> bool:
        key = duplicate_key(doc)
        if key == "java":
            return not any(duplicate_key(item.doc) == key for item in selected)
        if key == "opq":
            return len([item for item in selected if duplicate_key(item.doc) == key]) < 2
        return False
