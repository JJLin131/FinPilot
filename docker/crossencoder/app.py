import os
import logging
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("CROSSENCODER_DEVICE", "cpu")
MAX_LENGTH = int(os.getenv("CROSSENCODER_MAX_LENGTH", "512"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crossencoder")

model: CrossEncoder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    real_device = DEVICE if DEVICE == "cpu" or torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model: {MODEL_NAME} on device={real_device} max_length={MAX_LENGTH}")

    model = CrossEncoder(
        MODEL_NAME,
        device=real_device,
        max_length=MAX_LENGTH,
    )

    logger.info("Model loaded successfully")
    yield

    if model:
        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Model unloaded")


app = FastAPI(title="Cross-Encoder Reranker", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


class RerankItem(BaseModel):
    index: int
    score: float
    text: str


class RerankResponse(BaseModel):
    results: List[RerankItem]
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


@app.get("/health", response_model=HealthResponse)
def health():
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        device=str(model.model.device)
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.documents:
        raise HTTPException(status_code=400, detail="documents cannot be empty")

    pairs = [(req.query, doc) for doc in req.documents]
    scores = model.predict(pairs, show_progress_bar=False)

    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    results = [
        RerankItem(index=i, score=float(score), text=req.documents[i])
        for i, score in enumerate(scores)
    ]

    results.sort(key=lambda x: x.score, reverse=True)

    return RerankResponse(results=results, model=MODEL_NAME)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("CROSSENCODER_PORT", "8765"))
    )