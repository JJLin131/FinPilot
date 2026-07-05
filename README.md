# Agent Harness

Python finance-agent service built with `FastAPI + LangGraph + OpenTelemetry + Langfuse`.

The currently supported business path is finance knowledge QA:

```text
POST /api/finance/chat
  -> FastAPI
  -> FinanceAgentService
  -> LangGraph
  -> context load
  -> intent classify
  -> embedding score
  -> route decide
  -> QueryAgent
  -> search_finance_knowledge
  -> query rewrite
  -> Chroma vector retrieval + BM25 retrieval
  -> RRF fusion
  -> remote reranker
  -> answer compose
  -> MySQL audit + optional Langfuse score
```

`TransferAgent` is intentionally kept as an unsupported placeholder for now.

## Requirements

- Python 3.12+
- Docker Desktop
- MySQL database, default `work_memory`
- DeepSeek API key when `AI_PROVIDER=deepseek`
- Remote or profile-enabled local AI services:
  - Ollama on `11434`
  - Chroma on `8000`
  - reranker on `8081`

Copy `.env.example` to `.env` and fill in `DEEPSEEK_API_KEY` before Docker startup.

## Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m FinanceAgent.main
```

Health check:

```powershell
curl http://localhost:8099/healthz
```

## Docker Startup

```powershell
docker compose up --build
```

Default ports:

- Agent API: `http://localhost:8099`
- Langfuse UI: `http://localhost:3000`
- OTel HTTP receiver: `http://localhost:4318`
- MinIO API: `http://localhost:9090`

Optional local AI lab profile:

```powershell
docker compose --profile ai-lab up --build
```

## Knowledge Ingest

Single document ingest:

```powershell
curl -X POST http://localhost:8099/api/knowledge/documents `
  -H "Content-Type: application/json" `
  -d '{
    "document_id": "finance-rule-001",
    "domain": "FINANCE",
    "tenant_id": "tenant-a",
    "title": "基金赎回规则",
    "source": "manual",
    "content": "基金赎回通常 T+1 到账。",
    "tags": ["基金", "赎回"]
  }'
```

Bootstrap bundled Markdown resources:

```powershell
curl -X POST http://localhost:8099/api/knowledge/bootstrap/resources
```

The ingest path writes the same chunk set to:

- MySQL `knowledge_document`, `knowledge_document_chunk`, `knowledge_chunk_content`
- BM25 index under `data/bm25`
- Chroma document info and chunk vectors when vector services are reachable

## Chat API

```powershell
curl -X POST http://localhost:8099/api/finance/chat `
  -H "Content-Type: application/json" `
  -H "X-Debug-Trace: true" `
  -d '{
    "tenant_id": "tenant-a",
    "user_id": "user-1",
    "chat_id": "chat-1",
    "content": "工资发放审批规则是什么？"
  }'
```

Without `X-Debug-Trace: true`, route debug, retrieval debug, and tool calls are hidden from the response.

## Database

Runtime code creates the default database when the configured MySQL user has permission, then creates the tables it directly depends on:

- audit tables
- chat memory and user profile tables
- knowledge document and chunk tables

The reference DDL remains in `src/main/resources/schema.sql`.

## Evaluation Status

Evaluation runner code is present under `FinanceAgent/evals`, but datasets, pytest coverage, promptfoo config, and helper scripts are intentionally left as TODO work. See `TODO.md`.
