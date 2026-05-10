from __future__ import annotations

import json
import os
import re

from app.models import ChatMessage, HiringState, Intent, Role
from app.prompts import STATE_EXTRACTION_PROMPT


SKILL_ALIASES = {
    "java": ["java", "core java"],
    "spring": ["spring"],
    "sql": ["sql", "database", "relational"],
    "aws": ["aws", "amazon web services", "cloud"],
    "docker": ["docker", "container"],
    "rest": ["rest", "api"],
    "angular": ["angular"],
    "rust": ["rust"],
    "linux": ["linux"],
    "networking": ["networking", "network"],
    "excel": ["excel"],
    "word": ["word"],
    "hipaa": ["hipaa"],
    "medical terminology": ["medical terminology"],
    "finance": ["finance", "financial", "accounting"],
    "statistics": ["statistics", "stats"],
    "customer service": ["customer service", "contact centre", "contact center", "inbound calls"],
    "safety": ["safety", "dependability", "procedure compliance", "plant operator", "industrial", "chemical"],
    "sales": ["sales", "reskill", "talent audit"],
    "leadership": ["leadership", "cxo", "director", "executive"],
    "communication": ["communication", "communicate", "written", "verbal", "stakeholder"],
    "teamwork": ["teamwork", "collaboration", "collaborate", "mentor", "team"],
}


class StateExtractor:
    def extract(self, messages: list[ChatMessage]) -> HiringState:
        if os.getenv("OPENROUTER_API_KEY") and os.getenv("USE_LLM_EXTRACTION", "").lower() == "true":
            llm_state = self._extract_with_llm(messages)
            if llm_state:
                return llm_state
        return self._extract_with_rules(messages)

    def _extract_with_llm(self, messages: list[ChatMessage]) -> HiringState | None:
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            compact_history = "\n".join(f"{m.role.value}: {m.content}" for m in messages[-8:])
            response = client.chat.completions.create(
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
                messages=[
                    {"role": "system", "content": STATE_EXTRACTION_PROMPT},
                    {"role": "user", "content": compact_history},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                timeout=12,
            )
            content = response.choices[0].message.content or "{}"
            # Strip markdown code fences if the model wraps JSON in ```json ... ```
            content = re.sub(r"^```(?:json)?\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content.strip())
            return self._augment_state(HiringState.model_validate(json.loads(content)), messages)
        except Exception as exc:
            exc_str = str(exc)
            if "402" in exc_str or "payment" in exc_str.lower() or "billing" in exc_str.lower():
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "state_extraction_llm_unavailable: OpenRouter billing issue (402). Falling back to rule-based extraction."
                )
            return None

    def _extract_with_rules(self, messages: list[ChatMessage]) -> HiringState:
        user_messages = [message.content for message in messages if message.role == Role.user]
        assistant_messages = [message.content for message in messages if message.role == Role.assistant]
        latest = user_messages[-1] if user_messages else ""
        full_user_text = "\n".join(user_messages).lower()
        latest_lower = latest.lower()

        intent = Intent.recommend
        # Priority order: close > compare > refine > recommend.
        # Using elif so an explicit close like "Keep Verify G+. Locking it in."
        # is not overridden by the "keep" refine token.
        _close_tokens = [
            "thanks", "thank you", "perfect", "confirmed", "locking it in",
            "that works", "that's good", "that covers it", "that's what we need",
            "clear.", "good two-stage", "good choice",
            "keep the shortlist", "keep the five", "understood. keep",
            "final list", "final battery", "finalise", "finalize",
        ]
        # "good." only closes when the whole message is short (≤ 5 words) so
        # "Good. Can you also add..." does not prematurely end the conversation.
        _short_close_tokens = ["good.", "good!", "okay.", "ok."]
        _is_short_message = len(latest_lower.split()) <= 5
        if any(token in latest_lower for token in _close_tokens) or (
            _is_short_message and any(token in latest_lower for token in _short_close_tokens)
        ):
            intent = Intent.close
        # Compare signals — questions about the difference between assessments.
        elif any(token in latest_lower for token in ["compare", "difference between", "different from", "versus", " vs ", "what's the difference"]):
            intent = Intent.compare
        # Refine signals — add/drop/replace specific items.
        elif any(token in latest_lower for token in ["add", "drop", "remove", "replace", "make it", "keep", "update", "swap"]):
            intent = Intent.refine

        skills = []
        for skill, aliases in SKILL_ALIASES.items():
            if any(alias in full_user_text for alias in aliases):
                skills.append(skill)

        role = self._extract_role(full_user_text)
        seniority = self._extract_seniority(full_user_text)

        state = HiringState(
            intent=intent,
            role=role,
            job_description=latest if len(latest) > 120 or "jd" in latest_lower else None,
            skills=skills,
            seniority=seniority,
            personality_required=self._bool_from_text(
                full_user_text,
                ["personality", "opq", "behavior", "behaviour", "work style", "fit", "culture"],
            ),
            cognitive_required=self._bool_from_text(full_user_text, ["cognitive", "g+", "reasoning", "numerical", "problem-solving", "problem solving"]),
            situational_required=self._bool_from_text(full_user_text, ["situational", "scenario", "judgement", "judgment", "simulation"]),
            stakeholder_interaction=self._bool_from_text(full_user_text, ["stakeholder", "client facing", "customer facing", "cross-functional", "cross functional"]),
            communication_required=self._bool_from_text(full_user_text, ["communication", "communicate", "presentation", "spoken"]),
            teamwork_required=self._bool_from_text(full_user_text, ["teamwork", "collaboration", "collaborate", "team"]),
            language=self._extract_language(full_user_text),
            region="US" if re.search(r"\bus\b|usa|south texas|english", full_user_text) else None,
            include_terms=self._include_terms(latest_lower),
            exclude_terms=self._exclude_terms(latest_lower),
            compared_items=self._compared_items(latest),
            previous_recommendations=self._previous_recommendations(assistant_messages),
        )

        if not state.has_minimum_signal and intent not in {Intent.compare, Intent.close}:
            state.intent = Intent.clarify
        return self._augment_state(state, messages)

    def _augment_state(self, state: HiringState, messages: list[ChatMessage]) -> HiringState:
        full_user_text = "\n".join(message.content for message in messages if message.role == Role.user).lower()
        if state.seniority is None:
            state.seniority = self._extract_seniority(full_user_text)
        if state.stakeholder_interaction is None:
            state.stakeholder_interaction = self._bool_from_text(full_user_text, ["stakeholder", "client facing", "customer facing", "cross-functional", "cross functional"])
        if state.communication_required is None:
            state.communication_required = self._bool_from_text(full_user_text, ["communication", "communicate", "presentation", "spoken"])
        if state.teamwork_required is None:
            state.teamwork_required = self._bool_from_text(full_user_text, ["teamwork", "collaboration", "collaborate", "team"])
        if state.personality_required is None:
            state.personality_required = self._bool_from_text(full_user_text, ["personality", "opq", "behavior", "behaviour", "work style", "culture fit"])
        if state.cognitive_required is None:
            state.cognitive_required = self._bool_from_text(full_user_text, ["cognitive", "g+", "reasoning", "numerical", "problem-solving", "problem solving"])
        if not state.skills:
            state.skills = [skill for skill, aliases in SKILL_ALIASES.items() if any(alias in full_user_text for alias in aliases)]
        if not state.has_minimum_signal and state.intent not in {Intent.compare, Intent.close}:
            state.intent = Intent.clarify
        return state

    def _extract_seniority(self, text: str) -> str | None:
        years = [int(match) for match in re.findall(r"(\d+)\+?\s*(?:years|yrs)", text)]
        if any(year >= 7 for year in years):
            seniority = "senior"
        elif any(3 <= year < 7 for year in years):
            seniority = "mid"
        elif any(term in text for term in ["senior", "cxo", "director", "executive", "tech lead", "lead engineer", "architect"]):
            seniority = "senior"
        elif any(term in text for term in ["graduate", "entry-level", "entry level", "final-year", "no work experience", "junior"]):
            seniority = "entry"
        elif any(term in text for term in ["mid", "intermediate", "experienced"]):
            seniority = "mid"
        else:
            seniority = None
        return seniority

    def _extract_role(self, text: str) -> str | None:
        patterns = [
            r"hiring (?:a |an |for )?([^.\n?]+)",
            r"need (?:a |an )?(?:solution|battery|assessment|assessments)? ?(?:for )?([^.\n?]+)",
            r"screen(?:ing)? [0-9]* ?([^.\n?]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                role = match.group(1).strip(" -")
                return role[:120]
        return None

    def _extract_language(self, text: str) -> str | None:
        if "spanish" in text:
            return "Spanish"
        if "english" in text:
            return "English"
        return None

    def _bool_from_text(self, text: str, needles: list[str]) -> bool | None:
        if any(f"no {needle}" in text or f"drop {needle}" in text or f"remove {needle}" in text for needle in needles):
            return False
        if any(needle in text for needle in needles):
            return True
        return None

    def _include_terms(self, latest: str) -> list[str]:
        terms = []
        for token in ["aws", "docker", "personality", "cognitive", "simulation", "situational", "senior", "stakeholder", "communication", "teamwork", "shorter"]:
            if token in latest:
                terms.append(token)
        return terms

    def _exclude_terms(self, latest: str) -> list[str]:
        terms = []
        match = re.search(r"(?:drop|remove|without)\s+([a-zA-Z0-9+ .-]+)", latest)
        if match:
            terms.extend(match.group(1).replace("and", ",").split(","))
        if "drop rest" in latest:
            terms.append("rest")
        if "drop the opq" in latest or "remove the opq" in latest:
            terms.extend(["opq", "opq32r"])
        return [term.strip().lower() for term in terms if term.strip()]

    def _compared_items(self, latest: str) -> list[str]:
        known = ["OPQ", "OPQ32r", "GSA", "Global Skills Assessment", "DSI", "Safety", "Contact Center Call Simulation", "Customer Service Phone Simulation", "Verify G+"]
        found = [item for item in known if item.lower() in latest.lower()]
        if "gsa" in latest.lower() and "Global Skills Assessment" not in found:
            found.append("Global Skills Assessment")
        return found

    def _previous_recommendations(self, assistant_messages: list[str]) -> list[str]:
        names: list[str] = []
        for message in assistant_messages:
            found_table = False
            # LLM-generated responses: pipe-delimited markdown table rows.
            for line in message.splitlines():
                if not line.startswith("|") or " | " not in line:
                    continue
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if cells and cells[0].isdigit() and len(cells) > 1:
                    names.append(cells[1])
                    found_table = True
            # Template responses: "hiring context: Name1, Name2." or "refinement: Name1, Name2."
            # Skip compare responses — they list alternatives, not the active shortlist.
            if not found_table and "grounded comparison" not in message.lower():
                m = re.search(
                    r"(?:hiring context|refinement):\s*(.+?)(?:\.|$)",
                    message,
                    re.IGNORECASE,
                )
                if m:
                    template_names = [n.strip().rstrip(".") for n in m.group(1).split(",") if n.strip()]
                    names.extend(template_names)
        return names[-10:]
