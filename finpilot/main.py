from __future__ import annotations

import uvicorn

from finpilot.api import app
from finpilot.config import settings


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=settings.app_port)


if __name__ == "__main__":
    main()


