from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReleaseGate:
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)

    @classmethod
    def from_path(cls, path: Path) -> "ReleaseGate":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def evaluate(self, *, suite_results: list[Any], case_results: list[dict[str, Any]]) -> dict[str, Any]:
        reasons: list[str] = []
        blocking = set(self.config.get("blocking_statuses") or [])
        for item in case_results:
            status = str(item.get("status") or "")
            case_id = str(item.get("case_id") or "unknown")
            if status in blocking:
                reasons.append(f"{case_id}: blocking status {status}")
            elif self.config.get("critical_requires_all_passed") and item.get("severity") == "critical" and not item.get("passed"):
                reasons.append(f"{case_id}: critical case failed")
        thresholds = self.config.get("metric_thresholds") or {}
        for suite in suite_results:
            suite_name = str(getattr(suite, "suite", ""))
            metrics = getattr(suite, "metrics", {})
            for metric_name, limits in (thresholds.get(suite_name) or {}).items():
                value = metrics.get(metric_name)
                if not isinstance(value, (int, float)):
                    reasons.append(f"{suite_name}.{metric_name}: metric missing")
                    continue
                if "min" in limits and value < limits["min"]:
                    reasons.append(f"{suite_name}.{metric_name}: {value} is below minimum {limits['min']}")
                if "max" in limits and value > limits["max"]:
                    reasons.append(f"{suite_name}.{metric_name}: {value} exceeds maximum {limits['max']}")
        return {"status": "BLOCKED" if reasons else "PASSED", "reasons": reasons}
