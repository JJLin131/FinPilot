from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class UsageCapture:
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    cost: float = 0.0


_CURRENT_USAGE: ContextVar[UsageCapture | None] = ContextVar("finpilot_current_usage", default=None)


@contextmanager
def capture_usage():
    captured = UsageCapture()
    token = _CURRENT_USAGE.set(captured)
    try:
        yield captured
    finally:
        _CURRENT_USAGE.reset(token)


def record_usage(*, prompt_tokens: int, completion_tokens: int, cost: float = 0.0) -> None:
    captured = _CURRENT_USAGE.get()
    if captured is None:
        return
    captured.token_usage["prompt_tokens"] += int(prompt_tokens)
    captured.token_usage["completion_tokens"] += int(completion_tokens)
    captured.token_usage["total_tokens"] += int(prompt_tokens) + int(completion_tokens)
    captured.cost = round(captured.cost + float(cost), 8)
