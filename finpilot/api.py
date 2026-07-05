from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException

from finpilot.agent.service import FinPilotService
from finpilot.evals import EvalRunner
from finpilot.models import AgentChatResponse, FinPilotChatRequest
from finpilot.rag.lifecycle import KnowledgeLifecycleService
from finpilot.rag.models import KnowledgeDocumentRequest, KnowledgeDocumentResult


def create_app() -> FastAPI:
    app = FastAPI(title="FinPilot", version="0.1.0")
    service = FinPilotService()
    eval_runner = EvalRunner(service, service.audit_store)
    lifecycle_service = KnowledgeLifecycleService(service.rag_service)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.on_event("shutdown")
    def shutdown_service():
        service.shutdown()

    @app.post("/api/finance/chat", response_model=AgentChatResponse)
    def finance_chat(request: FinPilotChatRequest, x_debug_trace: str | None = Header(default=None)):
        response = service.chat(request.user_id, request.chat_id, request.content)
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

