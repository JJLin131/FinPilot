from __future__ import annotations

import logging

from python_agent_service.config import settings
from python_agent_service.langfuse_support import get_langfuse_client

logger = logging.getLogger(__name__)


def setup_langfuse() -> None:
    client = get_langfuse_client()
    if settings.langfuse_enabled and client is None:
        logger.warning("Langfuse is enabled but client initialization failed.")
