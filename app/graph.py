from __future__ import annotations

import logging
import time
from typing import TypedDict

from app.clarification import needs_clarification
from app.models import ChatMessage, ChatResponse, Intent, Recommendation, Role
from app.observability import traceable
from app.response_generator import ResponseGenerator
from app.retrieval import HybridRetriever
from app.reranker import Reranker
from app.safety import refusal_reason
from app.state_extractor import StateExtractor

logger = logging.getLogger(__name__)

MAX_USER_TURNS = 8


class WorkflowState(TypedDict, total=False):
    messages: list[ChatMessage]
    hiring_state: object
    retrieved_docs: list[object]
    response: ChatResponse
    refusal_reason: str
    user_turn_count: int


class RecommenderGraph:
    """
    Deterministic bounded LangGraph orchestration.

    Flow:
      START → safety → [refuse | extract_state]
            → [clarify | retrieval → rerank → grounded_response]
            → END

    Hard limits:
      - Max 8 user turns per conversation; turn 8 forces end_of_conversation=True.
      - Requests beyond turn 8 are refused.
    """

    def __init__(
        self,
        extractor: StateExtractor | None = None,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        generator: ResponseGenerator | None = None,
    ) -> None:
        self.extractor = extractor or StateExtractor()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.generator = generator or ResponseGenerator()
        self.workflow = self._build_workflow()

    @traceable(name="shl_recommender_graph")
    def invoke(self, messages: list[ChatMessage]) -> ChatResponse:
        t0 = time.perf_counter()
        result = self.workflow.invoke({"messages": messages})
        elapsed = time.perf_counter() - t0
        logger.info("total_graph_time=%.3fs", elapsed)
        response = result.get("response")
        if response is None:
            raise RuntimeError("LangGraph completed without a response.")
        return response

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_workflow(self):
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(WorkflowState)
        workflow.add_node("safety", self._safety_node)
        workflow.add_node("extract_state", self._extract_state_node)
        workflow.add_node("clarification_response", self._clarification_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("grounded_response", self._grounded_response_node)
        workflow.add_node("refusal_response", self._refusal_node)

        workflow.set_entry_point("safety")
        workflow.add_conditional_edges(
            "safety",
            self._after_safety,
            {"refuse": "refusal_response", "continue": "extract_state"},
        )
        workflow.add_conditional_edges(
            "extract_state",
            self._after_extract,
            {"clarify": "clarification_response", "retrieve": "retrieval"},
        )
        workflow.add_edge("retrieval", "rerank")
        workflow.add_edge("rerank", "grounded_response")
        workflow.add_edge("refusal_response", END)
        workflow.add_edge("clarification_response", END)
        workflow.add_edge("grounded_response", END)
        return workflow.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _safety_node(self, state: WorkflowState) -> WorkflowState:
        messages: list[ChatMessage] = state["messages"]
        user_turns = [m for m in messages if m.role == Role.user]
        state["user_turn_count"] = len(user_turns)

        # Hard cap: refuse anything beyond turn 8.
        if state["user_turn_count"] > MAX_USER_TURNS:
            state["refusal_reason"] = (
                "This conversation has reached the 8-turn limit. "
                "Please start a new session to continue."
            )
            logger.info("turn_limit_exceeded turn=%d", state["user_turn_count"])
            return state

        reason = refusal_reason(messages)
        if reason:
            state["refusal_reason"] = reason
            logger.info("refusal_reason=%s", reason)
        return state

    def _after_safety(self, state: WorkflowState) -> str:
        return "refuse" if state.get("refusal_reason") else "continue"

    def _refusal_node(self, state: WorkflowState) -> WorkflowState:
        state["response"] = self.generator.refusal(str(state["refusal_reason"]))
        return state

    @traceable(name="extract_state_node")
    def _extract_state_node(self, state: WorkflowState) -> WorkflowState:
        t0 = time.perf_counter()
        state["hiring_state"] = self.extractor.extract(state["messages"])
        logger.info(
            "extract_state_node done=%.3fs state=%s",
            time.perf_counter() - t0,
            state["hiring_state"].model_dump(mode="json"),
        )
        return state

    def _after_extract(self, state: WorkflowState) -> str:
        hiring_state = state["hiring_state"]
        if needs_clarification(hiring_state, state["messages"]):
            logger.info(
                "clarification_needed confidence=%.2f reason=%s",
                hiring_state.clarification_confidence,
                hiring_state.clarification_reason,
            )
            return "clarify"
        logger.info("clarification_not_needed proceeding_to_retrieval")
        return "retrieve"

    def _clarification_node(self, state: WorkflowState) -> WorkflowState:
        state["response"] = self.generator.generate(state["hiring_state"], [])
        return state

    @traceable(name="retrieval_node")
    def _retrieval_node(self, state: WorkflowState) -> WorkflowState:
        t0 = time.perf_counter()
        docs = self.retriever.retrieve(state["hiring_state"], top_k=20)
        elapsed = time.perf_counter() - t0
        logger.info(
            "retrieval_node done=%.3fs count=%d names=%s",
            elapsed,
            len(docs),
            [doc.name for doc in docs],
        )
        state["retrieved_docs"] = docs
        return state

    @traceable(name="rerank_node")
    def _rerank_node(self, state: WorkflowState) -> WorkflowState:
        t0 = time.perf_counter()
        docs = self.reranker.rerank(state["hiring_state"], state["retrieved_docs"], top_k=5)
        elapsed = time.perf_counter() - t0
        logger.info(
            "rerank_node done=%.3fs count=%d names=%s",
            elapsed,
            len(docs),
            [doc.name for doc in docs],
        )
        state["retrieved_docs"] = docs
        return state

    @traceable(name="grounded_response_node")
    def _grounded_response_node(self, state: WorkflowState) -> WorkflowState:
        t0 = time.perf_counter()
        hiring_state = state["hiring_state"]

        # On close intent, surface the previously recommended docs rather than
        # whatever fresh retrieval happened to return. This prevents the confirmed
        # shortlist from echoing unrelated catalog items.
        if hiring_state.intent.value == "close" and hiring_state.previous_recommendations:
            prev_names = {name.lower() for name in hiring_state.previous_recommendations}
            prev_docs = [doc for doc in self.retriever.catalog if doc.name.lower() in prev_names]
            if prev_docs:
                state["retrieved_docs"] = prev_docs

        response = self.generator.generate(state["hiring_state"], state["retrieved_docs"])

        # Turn 8 is the last allowed turn — force close.
        if state.get("user_turn_count", 0) >= MAX_USER_TURNS:
            response = ChatResponse(
                reply=response.reply + " (Conversation limit reached — this is the final response.)",
                recommendations=response.recommendations,
                end_of_conversation=True,
            )
            logger.info("turn_limit_reached forcing_end_of_conversation")

        logger.info(
            "grounded_response_node done=%.3fs end_of_conversation=%s",
            time.perf_counter() - t0,
            response.end_of_conversation,
        )
        state["response"] = response
        return state
