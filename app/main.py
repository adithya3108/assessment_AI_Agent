from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_environment
from app.conversation_logger import log_conversation
from app.graph import RecommenderGraph
from app.models import ChatRequest, ChatResponse, HealthResponse

load_environment()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

graph: RecommenderGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    configure_langsmith()
    graph = RecommenderGraph()
    yield


app = FastAPI(title="SHL Conversational Assessment Recommender", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=False)
def chat(request: ChatRequest) -> ChatResponse:
    active_graph = graph or RecommenderGraph()
    response = active_graph.invoke(request.messages)
    if response.end_of_conversation:
        try:
            log_conversation(request.messages, response)
        except Exception:
            pass  # never let logging break the API response
    return response


def configure_langsmith() -> None:
    """Enable LangSmith tracing when LANGSMITH_* variables are present."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "true"))
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "shl-conversational-recommender"))
