from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException

from python_agent_service.evals import EvalRunner
from python_agent_service.models import AgentChatResponse, FinanceChatRequest
from python_agent_service.service import FinanceAgentService


def create_app() -> FastAPI:
    app = FastAPI(title="ecommerce-ai-agent-service-py", version="0.1.0")
    service = FinanceAgentService()
    eval_runner = EvalRunner(service, service.audit_store)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/api/finance/chat", response_model=AgentChatResponse)
    def finance_chat(request: FinanceChatRequest, x_debug_trace: str | None = Header(default=None)):
        response = service.chat(request.tenant_id, request.user_id, request.chat_id, request.content)
        debug_enabled = (x_debug_trace or "").lower() == "true"
        if not debug_enabled:
            response.route_debug = None
            response.retrieval_debug = None
            response.tool_calls = None
        return response

    @app.post("/internal/evals/run/{suite}")
    def run_eval_suite(suite: str):
        try:
            return eval_runner.run_suite(suite)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()

