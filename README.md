# FinPilot

FinPilot is a finance agent service and CLI built with `FastAPI + LangGraph + RAG + memory + audit + observability`.

The main business path is finance knowledge QA:

```text
User
  -> finpilot CLI or POST /api/finance/chat
  -> FinPilotService
  -> LangGraph
  -> user memory load
  -> intent classify
  -> embedding score
  -> route decide
  -> QueryAgent
  -> shared finance knowledge retrieval
  -> answer compose
  -> user-scoped memory + audit
```

Knowledge is shared across all users and managed by administrators. User isolation applies to chat memory, profile memory, semantic memory, and audit attribution through `user_id + chat_id`.

## Requirements

- Python 3.12+
- Docker Desktop
- MySQL database, default `work_memory`
- DeepSeek API key when `AI_PROVIDER=deepseek`
- Remote or profile-enabled local AI services:
  - Ollama on `11434`
  - Chroma on `8000`
  - reranker on `8081`

Copy `.env.example` to `.env` and fill in `DEEPSEEK_API_KEY`.

## Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\finpilot.exe doctor
.\.venv\Scripts\finpilot.exe chat
```

Run the API service:

```powershell
.\.venv\Scripts\finpilot.exe serve
```

Health check:

```powershell
curl http://localhost:8099/healthz
```

## CLI

Ask one question:

```powershell
finpilot ask "工资发放审批规则是什么？" --user-id user-1
```

Start an interactive session:

```powershell
finpilot chat --user-id user-1
```

Useful REPL commands:

```text
/help
/new [chat-id]
/debug on
/context
/clear
/exit
```

Bootstrap shared knowledge resources:

```powershell
finpilot knowledge bootstrap
```

## Docker Startup

```powershell
docker compose up --build
```

Default ports:

- FinPilot API: `http://localhost:8099`
- Langfuse UI: `http://localhost:3000`
- OTel HTTP receiver: `http://localhost:4318`
- MinIO API: `http://localhost:9090`

Optional local AI lab profile:

```powershell
docker compose --profile ai-lab up --build
```

## Shared Knowledge Ingest

Single document ingest:

```powershell
curl -X POST http://localhost:8099/api/knowledge/documents `
  -H "Content-Type: application/json" `
  -d '{
    "document_id": "finance-rule-001",
    "domain": "FINANCE",
    "title": "工资发放审批规则",
    "source": "manual",
    "content": "工资发放通常需要提交审批并完成财务复核。",
    "tags": ["工资", "审批"]
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
- shared knowledge document and chunk tables

Existing databases with older nullable `tenant_id` columns remain compatible; FinPilot no longer writes business tenant data.

## Evaluation Status

Evaluation runner code is present under `finpilot/evals`, but datasets, pytest coverage, promptfoo config, and helper scripts are intentionally left as TODO work. See `TODO.md`.
