# E-Commerce AI Agent Service

Python finance-agent service built with `FastAPI + LangGraph + OpenTelemetry + Langfuse`, using your host MySQL for audits and a remote Ollama/reranker machine over Tailscale.

## What Runs Now

- Python service is the primary runtime.
- MySQL stays on your host machine.
- Docker Compose starts:
  - `agent-service`
  - `otel-collector`
  - self-hosted `Langfuse` stack
- Optional local AI lab containers still exist behind the `ai-lab` profile, but your default setup uses a remote Tailscale AI box instead.

## Architecture

```text
POST /api/finance/chat
  -> FastAPI
  -> FinanceAgentService
  -> LangGraph pipeline
  -> intent classify
  -> embedding score
  -> route decide
  -> sub-agent
  -> tool invoke / RAG lookup
  -> remote Ollama query rewrite
  -> remote reranker
  -> answer compose
  -> MySQL audit + Langfuse trace/score
```

Current Python package:

```text
python_agent_service/
  api.py
  service.py
  graph.py
  router.py
  subagents.py
  tools.py
  rag.py
  llm.py
  reranker.py
  audit.py
  tracing.py
  langfuse_support.py
  ragas_support.py
  evals.py
```

## Prerequisites

- Docker Desktop
- Python 3.13+ for local development
- A local MySQL instance on your host machine
- A remote machine reachable by Tailscale that exposes:
  - Ollama on `11434`
  - reranker on `8081`

Expected host MySQL defaults:

- host: `localhost`
- port: `3306`
- database: `work_memory`
- user: `root`
- password: `123456`

If your MySQL differs, update:

- [.env](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\.env:1)
- [compose.yaml](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\compose.yaml:1)

## Remote Ollama And Reranker via Tailscale

Current defaults:

- `DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`
- `AI_PROVIDER=deepseek`
- `AI_MODEL_NAME=deepseek-v4-pro`
- `ROUTING_PROVIDER=deepseek`
- `ROUTING_MODEL_NAME=deepseek-v4-pro`
- `OLLAMA_BASE_URL=http://100.92.110.54:11434`
- `QUERY_REWRITER_BASE_URL=http://100.92.110.54:11434`
- `RERANKER_BASE_URL=http://100.92.110.54:8081`

These are used for:

- route classification via DeepSeek chat completions
- knowledge answer synthesis via DeepSeek chat completions
- query rewriting via remote Ollama
- reranking via remote cross-encoder service

Check remote connectivity before startup:

```powershell
.\.venv\Scripts\python.exe scripts\check_remote_ai.py
```

If it fails:

- verify both machines are online in Tailscale
- verify the remote machine listens on `11434` and `8081`
- verify the remote machine firewall allows inbound traffic
- verify this host can route to the Tailscale IP

## One-Click Docker Startup

Start the full Python + observability stack:

```powershell
docker compose up --build
```

This starts:

- Agent API: `http://localhost:8099`
- Langfuse UI: `http://localhost:3000`
- MinIO API: `http://localhost:9090`
- OTel Collector HTTP: `http://localhost:4318`

Default Langfuse bootstrap credentials:

- email: `admin@local.dev`
- password: `local-admin-password`

Default Langfuse project keys:

- public key: `pk-lf-local`
- secret key: `sk-lf-local`

Optional local AI containers:

```powershell
docker compose --profile ai-lab up --build
```

## MySQL Setup

This project does not containerize MySQL. It reuses your host MySQL.

Verify connectivity:

```powershell
.\.venv\Scripts\python.exe scripts\check_mysql.py
```

Create the Python-side extra tables:

```powershell
.\.venv\Scripts\python.exe scripts\init_python_agent_tables.py
```

Existing audit tables expected from the earlier Java service:

- `agent_tool_audit`
- `unknown_intent_audit`

Additional Python-side tables:

- `agent_eval_run`
- `agent_trace_feedback`

These DDLs are also present in [src/main/resources/schema.sql](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\src\main\resources\schema.sql:1).

## Local Python Startup

Create or reuse the project-local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the API:

```powershell
.\.venv\Scripts\python.exe -m python_agent_service.main
```

Health check:

```powershell
curl http://localhost:8099/healthz
```

## API Usage

```bash
curl -X POST http://localhost:8099/api/finance/chat \
  -H "Content-Type: application/json" \
  -H "X-Debug-Trace: true" \
  -d '{
    "tenant_id": "tenant-a",
    "user_id": "user-1",
    "chat_id": "chat-1",
    "content": "查一下账户余额"
  }'
```

Response fields:

- `request_id`
- `trace_id`
- `domain`
- `status`
- `answer`
- `evidence`
- `route`
- `route_debug` when `X-Debug-Trace: true`
- `retrieval_debug` when `X-Debug-Trace: true`
- `tool_calls` when `X-Debug-Trace: true`

## Evaluation

Dataset files live in [evals/datasets](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\evals\datasets).

Current suites:

- `routing`
- `tool-use`
- `rag-retrieval`
- `grounded-answer`
- `safety`

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

Run an eval suite through the API:

```powershell
curl -X POST http://localhost:8099/internal/evals/run/routing
```

Promptfoo config is in [promptfoo.yaml](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\promptfoo.yaml:1).

## Langfuse Integration

The project now uses Langfuse for:

- trace ingestion via OpenTelemetry Collector
- dataset synchronization from local `jsonl` eval files
- trace-level scoring for:
  - intent match
  - tool match
  - retrieval hit
  - safety blocking
  - runtime tool success
  - runtime unknown rate
  - faithfulness proxy
  - heuristic or Ragas-backed RAG scores

Relevant files:

- [python_agent_service/langfuse_support.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\python_agent_service\langfuse_support.py:1)
- [python_agent_service/evals.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\python_agent_service\evals.py:1)
- [observability/otel-collector.yaml](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\observability\otel-collector.yaml:1)

Manual dataset sync:

```powershell
.\.venv\Scripts\python.exe scripts\sync_langfuse_datasets.py
```

## Ragas Integration

The Docker image installs the `eval` extra, which includes `ragas`.

Local desktop development may still run without `ragas`. In that case:

- evals still run
- the service falls back to heuristic RAG scores
- the output includes `ragas_available`

Adapter file:

- [python_agent_service/ragas_support.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\python_agent_service\ragas_support.py:1)

## Useful Scripts

- MySQL check: [scripts/check_mysql.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\scripts\check_mysql.py:1)
- MySQL init: [scripts/init_python_agent_tables.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\scripts\init_python_agent_tables.py:1)
- Remote AI check: [scripts/check_remote_ai.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\scripts\check_remote_ai.py:1)
- Langfuse dataset sync: [scripts/sync_langfuse_datasets.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\scripts\sync_langfuse_datasets.py:1)

## Key Files

- Python app: [python_agent_service/main.py](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\python_agent_service\main.py:1)
- Docker stack: [compose.yaml](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\compose.yaml:1)
- Container image: [Dockerfile](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\Dockerfile:1)
- OTel collector: [observability/otel-collector.yaml](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\observability\otel-collector.yaml:1)
- Env defaults: [.env](D:\IntelliJ_IDEA_U\Projects\ecommerce-ai-agent-service\.env:1)

## Notes

- Dockerized agent reaches host MySQL via `host.docker.internal`.
- Dockerized agent reaches remote Ollama and reranker directly through the configured Tailscale IP.
- If Langfuse is down, the API still runs; traces and scores just will not export.
- If `ragas` is unavailable locally, heuristic scores are used instead of failing the eval run.
