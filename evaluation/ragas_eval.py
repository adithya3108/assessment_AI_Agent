from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_environment
from app.graph import RecommenderGraph
from app.models import ChatMessage
from app.retrieval import HybridRetriever
from app.reranker import Reranker
from app.state_extractor import StateExtractor


def load_sample_conversations(root: str = "GenAI_SampleConversations") -> list[dict[str, str]]:
    conversations = []
    for path in sorted(Path(root).glob("C*.md")):
        conversations.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
    return conversations


def build_ragas_dataset() -> list[dict[str, str]]:
    rows = []
    extractor = StateExtractor()
    retriever = HybridRetriever()
    reranker = Reranker()
    graph = RecommenderGraph(extractor=extractor, retriever=retriever, reranker=reranker)

    for item in load_sample_conversations():
        messages = extract_user_messages(item["text"])
        if not messages:
            continue
        state = extractor.extract(messages)
        docs = reranker.rerank(state, retriever.retrieve(state, top_k=20), top_k=5) if state.has_minimum_signal else []
        response = graph.invoke(messages)
        rows.append(
            {
                "question": "\n".join(message.content for message in messages),
                "answer": response.reply,
                "contexts": [doc.searchable_text for doc in docs],
                "ground_truth": item["text"],
            }
        )
    return rows


def extract_user_messages(markdown: str) -> list[ChatMessage]:
    blocks = re.findall(r"\*\*User\*\*\s*\n\n((?:>.*(?:\n|$))+)", markdown)
    messages: list[ChatMessage] = []
    for block in blocks:
        content = "\n".join(line.lstrip("> ").strip() for line in block.splitlines() if line.startswith(">")).strip()
        if content:
            messages.append(ChatMessage(role="user", content=content))
    return messages


def _configure_ragas_llm_and_embeddings():
    """Point RAGAS at OpenRouter free-tier LLaMA for metric scoring."""
    import os
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="google/gemma-4-26b-a4b-it:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_retries=3,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "SHL Recommender"),
            },
        )
    )
    return llm, None


def run_ragas(output_path: str | None = None) -> Any:
    """Run RAGAS metrics sequentially to avoid Groq rate limits."""
    import asyncio
    import time
    load_environment()
    from ragas.metrics import context_precision, context_recall, faithfulness
    from ragas.metrics.base import EvaluationMode

    groq_llm, _embed = _configure_ragas_llm_and_embeddings()
    metrics = [context_precision, context_recall, faithfulness]
    for metric in metrics:
        metric.llm = groq_llm

    dataset_cache = Path("evaluation/ragas_dataset.json")
    if dataset_cache.exists():
        rows = json.loads(dataset_cache.read_text(encoding="utf-8"))
        print(f"Loaded {len(rows)} rows from cached dataset.")
    else:
        rows = build_ragas_dataset()
    records = []
    for i, row in enumerate(rows):
        rec: dict[str, Any] = {
            "question": row["question"],
            "answer": row["answer"],
        }
        for metric in metrics:
            try:
                score = asyncio.run(metric.ascore(row))
                rec[metric.name] = round(float(score), 4) if score is not None else None
            except Exception as exc:
                print(f"  [{metric.name}] row {i} failed: {exc}")
                rec[metric.name] = None
            time.sleep(1)
        records.append(rec)
        print(f"Row {i+1}/{len(rows)}: {rec}")

    if output_path:
        Path(output_path).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Saved to {output_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on SHL sample conversations.")
    parser.add_argument("--output", default="evaluation/ragas_results.json")
    parser.add_argument("--dry-run", action="store_true", help="Build the dataset without calling RAGAS metrics.")
    args = parser.parse_args()
    if args.dry_run:
        dataset = build_ragas_dataset()
        Path(args.output).write_text(json.dumps(dataset, indent=2), encoding="utf-8")
        print(f"Wrote {len(dataset)} RAGAS rows to {args.output}")
    else:
        print(run_ragas(args.output))
