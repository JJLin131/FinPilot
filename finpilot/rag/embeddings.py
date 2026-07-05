from __future__ import annotations

import logging
from typing import Iterable

import httpx

from finpilot.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.model_name = model_name or settings.embedding_model_name
        self.timeout_seconds = timeout_seconds or settings.embedding_timeout_seconds

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_many([text])
        return vectors[0] if vectors else []

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        values = [text for text in texts]
        if not values:
            return []
        try:
            return self._embed_batch(values)
        except Exception as exc:
            logger.warning("Ollama batch embedding failed, falling back to single embedding calls: %s", exc)
        return [self._embed_one(text) for text in values]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
            )
            response.raise_for_status()
            payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Ollama /api/embed returned no embeddings.")
        return [self._coerce_vector(item) for item in embeddings]

    def _embed_one(self, text: str) -> list[float]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
            )
            response.raise_for_status()
            payload = response.json()
        return self._coerce_vector(payload.get("embedding"))

    def _coerce_vector(self, value) -> list[float]:
        if not isinstance(value, list) or not value:
            raise ValueError("Embedding response did not contain a vector.")
        return [float(item) for item in value]

