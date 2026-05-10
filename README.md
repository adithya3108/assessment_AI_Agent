# SHL Conversational Assessment Recommender

Stateless FastAPI RAG service for recommending SHL assessments from catalog data.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

## API

`GET /health`

```json
{"status":"ok"}
```

`POST /chat`

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a senior Java backend developer"}
  ]
}
```

Response is strictly:

```json
{
  "reply": "...",
  "recommendations": [
    {"name": "...", "url": "...", "test_type": "..."}
  ],
  "end_of_conversation": false
}
```

## Stateless Design

The backend stores no conversation memory. The client sends the full message history on every request. Each request extracts an ephemeral `HiringState`, retrieves catalog documents, reranks them, and returns a validated Pydantic response.

## Catalog And Vector Index

The app ships with a seeded catalog from the sample conversations, but the production path uses the official SHL JSON catalog plus OpenRouter embeddings in FAISS.

Ingest the official catalog:

```powershell
python scraper/ingest_catalog.py
```

Build the persisted FAISS index using the configured embedding provider.
By default this uses OpenRouter with `openai/text-embedding-3-small`:

```powershell
python scraper/build_index.py
```

At API startup, `app.retrieval.HybridRetriever` loads `vectorstore/faiss_index` with LangChain FAISS and merges vector hits with LangChain BM25 hits. It retrieves top 20 candidates, reranks them with `BAAI/bge-reranker-base`, and keeps the top 5.

## Keys

Add keys to `.env` later:

- `OPENROUTER_API_KEY` for extraction, generation, and default embeddings
- `OPENAI_API_KEY` only if you switch `EMBEDDING_PROVIDER=openai`
- `LANGSMITH_API_KEY` for tracing

LLM calls are disabled by default until keys are added:

```text
USE_LLM_EXTRACTION=false
USE_LLM_GENERATION=false
```

Set both to `true` after adding `OPENROUTER_API_KEY`.

Local config loading reads `.env.example` first and then applies any non-empty
values from `.env`. For sharing or deployment, keep real secrets in `.env` and
leave `.env.example` as placeholders.

## RAGAS

Run a dry dataset build:

```powershell
python evaluation/ragas_eval.py --dry-run
```

Run full RAGAS metrics once keys are available:

```powershell
python evaluation/ragas_eval.py
```
