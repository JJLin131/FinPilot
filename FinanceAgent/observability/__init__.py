from __future__ import annotations

import logging

from FinanceAgent.config import settings
from FinanceAgent.observability.langfuse_support import get_langfuse_client

logger = logging.getLogger(__name__)


def setup_langfuse() -> None:
    client = get_langfuse_client()
    if settings.langfuse_enabled and client is None:
        logger.warning("Langfuse is enabled but client initialization failed.")


__all__ = ["setup_langfuse"]
