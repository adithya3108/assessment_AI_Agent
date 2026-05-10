from __future__ import annotations

import re

from app.models import ChatMessage, HiringState, Intent, Role

# Keywords that count as an *explicit* seniority mention in the raw text.
_EXPLICIT_SENIORITY_RE = re.compile(
    r"\b(senior|junior|mid[- ]?level|intermediate|entry[- ]?level|graduate|"
    r"lead|principal|staff|architect|director|executive|cxo|\d+\+?\s*(?:years|yrs))\b",
    re.IGNORECASE,
)

# Keywords that explicitly name an assessment type preference.
_ASSESSMENT_TYPE_RE = re.compile(
    r"\b(personality|cognitive|situational|behavioural|behavioral|opq|aptitude|"
    r"reasoning|numerical|technical\s+only|no\s+personality|work[- ]?style)\b",
    re.IGNORECASE,
)


def needs_clarification(state: HiringState, messages: list[ChatMessage]) -> bool:
    if state.intent in {Intent.refine, Intent.compare, Intent.close, Intent.refuse}:
        return False
    if not state.has_minimum_signal:
        state.intent = Intent.clarify
        state.clarification_confidence = 1.0
        state.clarification_reason = "missing role, job description, or core skills"
        return True
    if _already_asked_clarification(messages):
        return False

    signal_score = _signal_score(state)
    state.clarification_confidence = max(0.0, min(1.0, (4 - signal_score) / 4))

    if _is_detailed_context(state):
        return False

    # Use the raw first user message to detect vague requests, regardless of
    # what the LLM extractor may have inferred from the role title.
    if _is_vague_first_request(state, messages):
        state.intent = Intent.clarify
        state.clarification_reason = "role/skill present but seniority and assessment mix are unclear from the request"
        return True

    if state.clarification_confidence >= 0.65:
        state.intent = Intent.clarify
        state.clarification_reason = "low confidence hiring context"
        return True

    return False


def clarification_question(state: HiringState) -> str:
    role_phrase = f" for the {state.role}" if state.role else ""
    return (
        f"To narrow this down{role_phrase}: "
        "what seniority level are you targeting (entry / mid / senior), "
        "and should the battery be purely technical or also include "
        "cognitive reasoning and personality / work-style assessments?"
    )


def _signal_score(state: HiringState) -> int:
    score = 0
    score += 1 if state.role or state.job_description else 0
    score += 1 if state.skills else 0
    score += 1 if state.seniority else 0
    score += 1 if state.personality_required is not None or state.cognitive_required is not None else 0
    score += 1 if state.stakeholder_interaction or state.communication_required or state.teamwork_required else 0
    score += 1 if state.job_description and len(state.job_description) > 160 else 0
    return score


def _is_detailed_context(state: HiringState) -> bool:
    return bool(
        (state.job_description and len(state.job_description) > 160)
        or (
            state.seniority
            and (
                state.personality_required is not None
                or state.cognitive_required is not None
                or state.stakeholder_interaction
                or state.communication_required
                or state.teamwork_required
            )
        )
    )


# Skills that map 1-to-1 to SHL catalog tests — no clarification needed when
# one of these is explicitly present, because the test selection is unambiguous.
_ATOMIC_SKILL_RE = re.compile(
    r"\b(excel|word|sql|java|spring|docker|aws|linux|hipaa|"
    r"medical terminology|networking|angular|rust|finance|statistics)\b",
    re.IGNORECASE,
)


def _is_vague_first_request(state: HiringState, messages: list[ChatMessage]) -> bool:
    user_turns = [message for message in messages if message.role == Role.user]
    if len(user_turns) != 1:
        return False
    text = user_turns[0].content

    # Requests up to 15 words are considered short enough to warrant clarification.
    short_request = len(text.split()) <= 15

    # If the user names an atomic skill (excel, java, SQL …) it implies a
    # specific test category — skip clarification for assessment type.
    has_atomic_skill = bool(_ATOMIC_SKILL_RE.search(text))

    # Check the *raw* text for explicit seniority keywords (not extracted state,
    # which may infer seniority from job titles).
    has_explicit_seniority = bool(_EXPLICIT_SENIORITY_RE.search(text))
    has_explicit_assessment_pref = bool(_ASSESSMENT_TYPE_RE.search(text))

    # Seniority alone (e.g. "senior leadership") is not enough — we need at least
    # one *concrete* (non-soft) skill. Soft skills like leadership/communication/teamwork
    # don't imply a specific test and therefore don't reduce ambiguity.
    _SOFT_SKILLS = {"communication", "teamwork", "leadership", "customer service", "safety", "sales"}
    has_concrete_skills = bool(set(state.skills) - _SOFT_SKILLS)
    missing_enrichment = not has_explicit_assessment_pref and not has_atomic_skill and (
        not has_explicit_seniority or not has_concrete_skills
    )
    return short_request and missing_enrichment


def _already_asked_clarification(messages: list[ChatMessage]) -> bool:
    for message in messages:
        if message.role == Role.assistant:
            text = message.content.lower()
            if "seniority" in text and ("personality" in text or "cognitive" in text or "assessment" in text):
                return True
    return False
