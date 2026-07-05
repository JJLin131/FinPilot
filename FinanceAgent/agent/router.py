from __future__ import annotations

import json
from difflib import SequenceMatcher

from FinanceAgent.intents import INTENT_KEYWORDS, INTENT_ORDER
from FinanceAgent.llm import IntentClassificationService
from FinanceAgent.models import RouteDecision

ROUTE_THRESHOLD = 0.78
REVIEW_THRESHOLD = 0.60
MIN_SIMILARITY = 0.35


class IntentRouter:
    def __init__(self, classifier: IntentClassificationService | None = None):
        self.classifier = classifier or IntentClassificationService()

    def classify(self, user_message: str) -> tuple[str, str]:
        try:
            return self.classifier.classify(user_message)
        except Exception:
            pass
        return self._heuristic_classify(user_message)

    def _heuristic_classify(self, user_message: str) -> tuple[str, str]:
        lowered = user_message.lower()
        for intent in INTENT_ORDER:
            keywords = [entry.lower() for entry in INTENT_KEYWORDS.get(intent, [])]
            if any(keyword in lowered for keyword in keywords):
                return intent, f"Matched intent keywords for {intent}"
        return "UNKNOWN", "No supported intent keywords matched"

    def embedding_score(self, user_message: str) -> tuple[str, str | None, float, float]:
        lowered = user_message.lower()
        scored: list[tuple[str, float]] = []
        for intent in INTENT_ORDER:
            keywords = INTENT_KEYWORDS.get(intent, [])
            best = 0.0
            for keyword in keywords:
                normalized = keyword.lower()
                if normalized in lowered:
                    best = max(best, 1.0)
                else:
                    best = max(best, SequenceMatcher(a=lowered, b=normalized).ratio())
            scored.append((intent, best))
        scored.sort(key=lambda item: item[1], reverse=True)
        top1, top1_score = scored[0]
        top2, top2_score = scored[1] if len(scored) > 1 else ("UNKNOWN", 0.0)
        semantic_score = top1_score
        margin_score = max(top1_score - top2_score, 0.0)
        return top1, top2, semantic_score, margin_score

    def route(self, user_message: str) -> RouteDecision:
        classifier_intent, reason = self.classify(user_message)
        top1, top2, semantic_score, margin_score = self.embedding_score(user_message)
        return self.route_from_scores(classifier_intent, reason, top1, top2, semantic_score, margin_score)

    def route_from_scores(
        self,
        classifier_intent: str,
        reason: str,
        top1: str | None,
        top2: str | None,
        semantic_score: float,
        margin_score: float,
    ) -> RouteDecision:
        raw_intent_json = json.dumps({"intent": classifier_intent, "reason": reason}, ensure_ascii=False)
        if classifier_intent == "UNKNOWN":
            return self._unknown(raw_intent_json, classifier_intent, reason, "MODEL_RETURNED_UNKNOWN", True, top1, top2)
        if top1 == "UNKNOWN":
            return self._unknown(raw_intent_json, classifier_intent, reason, "LOW_CONFIDENCE", True, top1, top2)

        agreement_score = 1.0 if classifier_intent == top1 else 0.0
        if semantic_score <= 0.0:
            return self._known(
                raw_intent_json,
                classifier_intent,
                reason,
                top1,
                top2,
                REVIEW_THRESHOLD,
                semantic_score,
                margin_score,
                agreement_score,
            )
        if semantic_score < MIN_SIMILARITY and classifier_intent != top1:
            return self._unknown(raw_intent_json, classifier_intent, reason, "LOW_SEMANTIC_SCORE", True, top1, top2)
        confidence = 0.70 * semantic_score + 0.15 * margin_score + 0.15 * agreement_score

        if semantic_score < MIN_SIMILARITY and classifier_intent == top1:
            return self._known(
                raw_intent_json,
                classifier_intent,
                reason,
                top1,
                top2,
                max(confidence, REVIEW_THRESHOLD),
                semantic_score,
                margin_score,
                agreement_score,
            )

        if confidence < REVIEW_THRESHOLD:
            cause = "MODEL_EMBEDDING_DISAGREEMENT" if classifier_intent != top1 else "LOW_CONFIDENCE"
            return self._unknown(
                raw_intent_json,
                classifier_intent,
                reason,
                cause,
                True,
                top1,
                top2,
                confidence,
                semantic_score,
                margin_score,
                agreement_score,
            )
        if confidence < ROUTE_THRESHOLD and classifier_intent != top1:
            return self._unknown(
                raw_intent_json,
                classifier_intent,
                reason,
                "MODEL_EMBEDDING_DISAGREEMENT",
                True,
                top1,
                top2,
                confidence,
                semantic_score,
                margin_score,
                agreement_score,
            )

        return self._known(
            raw_intent_json,
            classifier_intent,
            reason,
            top1,
            top2,
            round(max(confidence, ROUTE_THRESHOLD), 4),
            semantic_score,
            margin_score,
            agreement_score,
        )

    def _known(
        self,
        raw_json: str,
        classifier_intent: str,
        reason: str,
        top1: str | None,
        top2: str | None,
        confidence: float,
        semantic_score: float,
        margin_score: float,
        agreement_score: float,
    ) -> RouteDecision:
        return RouteDecision(
            raw_intent_json=raw_json,
            normalized_intent=classifier_intent,
            reason=reason,
            confidence=round(confidence, 4),
            valid=True,
            target_agent="QueryAgent",
            classifier_intent=classifier_intent,
            embedding_top1_intent=top1,
            embedding_top2_intent=top2,
            fallback_cause="NONE",
            semantic_score=round(semantic_score, 4),
            margin_score=round(margin_score, 4),
            agreement_score=agreement_score,
        )

    def _unknown(
        self,
        raw_json: str,
        classifier_intent: str,
        reason: str,
        fallback_cause: str,
        valid: bool,
        top1: str | None,
        top2: str | None,
        confidence: float = 0.0,
        semantic_score: float = 0.0,
        margin_score: float = 0.0,
        agreement_score: float = 0.0,
    ) -> RouteDecision:
        return RouteDecision(
            raw_intent_json=raw_json,
            normalized_intent="UNKNOWN",
            reason=reason,
            confidence=round(confidence, 4),
            valid=valid,
            target_agent="UNSUPPORTED",
            classifier_intent=classifier_intent,
            embedding_top1_intent=top1,
            embedding_top2_intent=top2,
            fallback_cause=fallback_cause,
            semantic_score=round(semantic_score, 4),
            margin_score=round(margin_score, 4),
            agreement_score=agreement_score,
        )
