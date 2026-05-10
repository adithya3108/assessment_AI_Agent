from __future__ import annotations

import json
import logging
import os

from app.clarification import clarification_question
from app.models import AssessmentDocument, ChatResponse, HiringState, Recommendation
from app.prompts import GENERATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ResponseGenerator:
    def generate(self, state: HiringState, docs: list[AssessmentDocument]) -> ChatResponse:
        if state.intent.value == "clarify" or not state.has_minimum_signal:
            return ChatResponse(
                reply=clarification_question(state),
                recommendations=[],
                end_of_conversation=False,
            )

        if state.intent.value == "close":
            return ChatResponse(
                reply="Confirmed. This shortlist stays grounded in the SHL catalog items we discussed.",
                recommendations=self._recommendations(docs),
                end_of_conversation=True,
            )

        if os.getenv("OPENROUTER_API_KEY") and os.getenv("USE_LLM_GENERATION", "").lower() == "true":
            generated = self._generate_with_llm(state, docs)
            if generated:
                return ChatResponse(
                    reply=generated,
                    recommendations=self._recommendations(docs),
                    end_of_conversation=False,
                )

        reply = self._generate_with_template(state, docs)
        return ChatResponse(reply=reply, recommendations=self._recommendations(docs), end_of_conversation=False)

    def refusal(self, reason: str) -> ChatResponse:
        return ChatResponse(reply=reason, recommendations=[], end_of_conversation=False)

    def _generate_with_llm(self, state: HiringState, docs: list[AssessmentDocument]) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            catalog_context = [
                {
                    "name": doc.name,
                    "test_type": doc.test_type,
                    "description": doc.description,
                    "skills": doc.skills[:8],
                    "categories": doc.categories,
                    "duration": doc.duration,
                }
                for doc in docs
            ]
            logger.info("generation_state=%s", state.model_dump(mode="json"))
            logger.info("generation_catalog_context=%s", catalog_context)
            response = client.chat.completions.create(
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"hiring_state": state.model_dump(mode="json"), "catalog_context": catalog_context},
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0.2,
                timeout=12,
            )
            return response.choices[0].message.content
        except Exception as exc:
            exc_str = str(exc)
            if "402" in exc_str or "payment" in exc_str.lower() or "billing" in exc_str.lower():
                logger.warning("generation_llm_unavailable: OpenRouter billing issue (402). Falling back to template response.")
            else:
                logger.warning("generation_llm_failed: %s. Falling back to template response.", exc_str[:200])
            return None

    def _generate_with_template(self, state: HiringState, docs: list[AssessmentDocument]) -> str:
        if not docs:
            if state.intent.value == "refine":
                return (
                    "The SHL catalog does not contain a shorter alternative with equivalent coverage — "
                    "the current shortlist represents the closest grounded fit available."
                )
            return "I could not find a grounded SHL catalog match for that request."
        names = ", ".join(doc.name for doc in docs[:5])
        if state.intent.value == "compare":
            return f"Grounded comparison from the retrieved SHL catalog: {names}. Use the test type and descriptions in the returned recommendations to compare scope and fit."
        if state.intent.value == "refine":
            return f"Updated shortlist based on your refinement: {names}."
        if state.personality_required or state.cognitive_required or state.stakeholder_interaction:
            return f"Here is a balanced SHL shortlist for the hiring context: {names}."
        return f"Here are SHL catalog assessments that best match the hiring context: {names}."

    def _recommendations(self, docs: list[AssessmentDocument]) -> list[Recommendation]:
        return [Recommendation(name=doc.name, url=doc.url, test_type=doc.test_type) for doc in docs]
