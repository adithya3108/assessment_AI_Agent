from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=16)

    @field_validator("messages")
    @classmethod
    def require_user_message(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not any(message.role == Role.user for message in messages):
            raise ValueError("At least one user message is required.")
        return messages


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class AssessmentDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    url: str
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    test_type: str = ""
    duration: str = ""
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        parts = [
            self.name,
            self.description,
            self.test_type,
            self.duration,
            " ".join(self.skills),
            " ".join(self.job_levels),
            " ".join(self.languages),
            " ".join(self.categories),
        ]
        return " ".join(part for part in parts if part).lower()


class Intent(str, Enum):
    recommend = "recommend"
    refine = "refine"
    compare = "compare"
    clarify = "clarify"
    refuse = "refuse"
    close = "close"


class HiringState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent = Intent.recommend
    role: str | None = None
    job_description: str | None = None
    skills: list[str] = Field(default_factory=list)
    seniority: str | None = None
    personality_required: bool | None = None
    cognitive_required: bool | None = None
    situational_required: bool | None = None
    stakeholder_interaction: bool | None = None
    communication_required: bool | None = None
    teamwork_required: bool | None = None
    language: str | None = None
    region: str | None = None
    include_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    compared_items: list[str] = Field(default_factory=list)
    previous_recommendations: list[str] = Field(default_factory=list)
    clarification_confidence: float = 0.0
    clarification_reason: str | None = None

    @property
    def has_minimum_signal(self) -> bool:
        return bool(self.role or self.job_description or self.skills)

    def query(self) -> str:
        return self.retrieval_query()

    def retrieval_query(self) -> str:
        parts = [
            self.role or "",
            self.job_description or "",
            " ".join(self.skills),
            self.seniority or "",
            "personality workplace behavior work style" if self.personality_required else "",
            "cognitive ability reasoning" if self.cognitive_required else "",
            "situational judgement scenarios" if self.situational_required else "",
            "stakeholder communication collaboration influencing" if self.stakeholder_interaction else "",
            "communication verbal written customer stakeholder" if self.communication_required else "",
            "teamwork collaboration leadership" if self.teamwork_required else "",
            " ".join(self.include_terms),
            " ".join(self.compared_items),
        ]
        return " ".join(part for part in parts if part).strip()


class GraphState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage]
    hiring_state: HiringState | None = None
    retrieved_docs: list[AssessmentDocument] = Field(default_factory=list)
    response: ChatResponse | None = None
