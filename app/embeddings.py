from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class OpenRouterEmbeddings(Embeddings):
    """LangChain embeddings wrapper for OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.batch_size = batch_size
        # Prefer dedicated embedding credentials so the LLM provider (e.g. Groq)
        # can differ from the embedding provider (OpenRouter).
        self.client = OpenAI(
            api_key=api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"),
            default_headers=self._headers(),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([item.embedding for item in response.data])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        referer = os.getenv("OPENROUTER_SITE_URL")
        title = os.getenv("OPENROUTER_APP_NAME")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers


def get_embeddings() -> Embeddings:
    provider = os.getenv("EMBEDDING_PROVIDER", "openrouter").lower()
    model = os.getenv("EMBEDDING_MODEL")
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model or "text-embedding-3-small")
    if provider == "openrouter":
        return OpenRouterEmbeddings(model=model or "openai/text-embedding-3-small")
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
