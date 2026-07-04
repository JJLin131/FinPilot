from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException

from FinanceAgent.agent.service import FinanceAgentService
from FinanceAgent.evals import EvalRunner
from FinanceAgent.models import AgentChatResponse, FinanceChatRequest
from FinanceAgent.rag.lifecycle import KnowledgeLifecycleService
from FinanceAgent.rag.models import KnowledgeDocumentRequest, KnowledgeDocumentResult


def create_app() -> FastAPI:
    app = FastAPI(title="ecommerce-ai-agent-service-py", version="0.1.0")
    service = FinanceAgentService()
    eval_runner = EvalRunner(service, service.audit_store)
    lifecycle_service = KnowledgeLifecycleService(service.rag_service)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.on_event("shutdown")
    def shutdown_service():
        service.shutdown()

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

    @app.post("/api/knowledge/documents", response_model=KnowledgeDocumentResult)
    def ingest_knowledge_document(request: KnowledgeDocumentRequest):
        return service.rag_service.ingest(request)

    @app.post("/api/knowledge/bootstrap/resources", response_model=list[KnowledgeDocumentResult])
    def bootstrap_knowledge_resources():
        return service.rag_service.bootstrap_resources()

    @app.delete("/api/knowledge/expired")
    def delete_expired_knowledge_documents():
        return {"deletedDocuments": lifecycle_service.remove_expired_documents()}

    return app


app = create_app()
