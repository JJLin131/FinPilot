from __future__ import annotations

import uvicorn

from python_agent_service.api import app
from python_agent_service.config import settings


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=settings.app_port)


if __name__ == "__main__":
    main()

