from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from finpilot.safety.models import SafetyFinding, SafetyReviewResult


class ArgumentReviewer:
    name = "argument"

    def review(self, spec, parameters: dict[str, Any]) -> SafetyReviewResult:
        args_model = getattr(spec, "args_model", None)
        if args_model is None:
            return SafetyReviewResult.allow(parameters)
        try:
            validated = args_model.model_validate(parameters)
        except ValidationError as exc:
            return SafetyReviewResult(
                action="BLOCK",
                findings=[
                    SafetyFinding(
                        code="TOOL_ARGUMENT_VALIDATION_FAILED",
                        reviewer=self.name,
                        action="BLOCK",
                        message="Tool arguments failed schema validation.",
                        detail={"errors": exc.errors()},
                    )
                ],
            )
        return SafetyReviewResult.allow(validated.model_dump())
