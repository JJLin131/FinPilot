from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ContextSegment(BaseModel):
    name: str
    value: Any
    priority: int = 50


class CompressionRule(BaseModel):
    name: str
    path: str
    method: Literal["deterministic", "llm"]
    target_tokens: int
    priority: int = 50
    preserve_last: int = 0

    @model_validator(mode="after")
    def validate_rule(self):
        supported_paths = {
            "loop.old_step_history",
            "loop.tool_outputs",
            "session.recent_messages",
            "session.semantic_memory",
            "evidence",
            "subagent_results",
        }
        if self.path not in supported_paths:
            raise ValueError(f"unsupported compression path: {self.path}")
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.preserve_last < 0:
            raise ValueError("preserve_last must not be negative")
        return self


class ContextPolicy(BaseModel):
    token_budget: int
    trigger_ratio: float = 0.85
    reserved_output_tokens: int = 4096
    protected_paths: list[str] = Field(default_factory=list)
    segment_limits: dict[str, int] = Field(default_factory=dict)
    compression_rules: list[CompressionRule] = Field(default_factory=list)
    summary_chunk_token_budget: int = 8000

    @model_validator(mode="after")
    def validate_policy(self):
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not 0 < self.trigger_ratio <= 1:
            raise ValueError("trigger_ratio must be between 0 and 1")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must not be negative")
        if self.summary_chunk_token_budget <= 0:
            raise ValueError("summary_chunk_token_budget must be positive")
        names = [rule.name for rule in self.compression_rules]
        if len(names) != len(set(names)):
            raise ValueError("compression rule names must be unique")
        return self


class ContextUsage(BaseModel):
    requested_policy: str
    effective_policy: str
    stage: Literal["global", "decision", "answer"]
    token_budget: int
    effective_input_budget: int
    reserved_output_tokens: int
    trigger_tokens: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    estimated_prompt_tokens: int
    compressed: bool
    within_budget: bool
    model_window_source: str
    token_counter: str = "heuristic"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class PromptContextBundle(BaseModel):
    status: Literal["ready", "over_budget"] = "ready"
    payload: dict[str, Any]
    usage: ContextUsage
    compression_events: list[dict[str, Any]]
    over_budget_paths: list[str] = Field(default_factory=list)


_AGENT_COMPRESSION_RULES = [
    CompressionRule(
        name="old_step_history",
        path="loop.old_step_history",
        method="deterministic",
        target_tokens=4_000,
        priority=10,
    ),
    CompressionRule(
        name="tool_outputs",
        path="loop.tool_outputs",
        method="llm",
        target_tokens=8_000,
        priority=20,
    ),
    CompressionRule(
        name="recent_messages",
        path="session.recent_messages",
        method="llm",
        target_tokens=4_000,
        priority=30,
        preserve_last=2,
    ),
    CompressionRule(
        name="semantic_memory",
        path="session.semantic_memory",
        method="llm",
        target_tokens=2_000,
        priority=40,
    ),
    CompressionRule(
        name="evidence",
        path="evidence",
        method="llm",
        target_tokens=18_000,
        priority=50,
    ),
]

_AGENT_PROTECTED_PATHS = [
    "session.user_message",
    "sub_agent.agent_name",
    "sub_agent.goal",
    "sub_agent.allowed_tools[*].name",
    "sub_agent.allowed_tools[*].arguments",
    "loop.step_history[-1].decision",
    "loop.step_history[-1].observation.tool_name",
    "loop.step_history[-1].observation.status",
    "loop.step_history[-1].observation.summary",
]

DEFAULT_CONTEXT_POLICIES: dict[str, ContextPolicy] = {
    "global_default": ContextPolicy(
        token_budget=64_000,
        protected_paths=["session.user_message", "route", "subagent_results[*].agent_name"],
        segment_limits={"session": 32_000, "subagent_results": 24_000},
        compression_rules=[
            CompressionRule(
                name="recent_messages",
                path="session.recent_messages",
                method="llm",
                target_tokens=12_000,
                priority=20,
                preserve_last=2,
            ),
            CompressionRule(
                name="semantic_memory",
                path="session.semantic_memory",
                method="llm",
                target_tokens=8_000,
                priority=30,
            ),
            CompressionRule(
                name="completed_subagents",
                path="subagent_results",
                method="deterministic",
                target_tokens=24_000,
                priority=40,
            ),
        ],
    ),
    "agent_default": ContextPolicy(
        token_budget=32_000,
        protected_paths=_AGENT_PROTECTED_PATHS,
        segment_limits={"session": 8_000, "sub_agent": 4_000, "loop": 16_000, "evidence": 18_000},
        compression_rules=_AGENT_COMPRESSION_RULES,
    ),
    "finance_qa_agent": ContextPolicy(
        token_budget=32_000,
        protected_paths=_AGENT_PROTECTED_PATHS,
        segment_limits={"session": 8_000, "sub_agent": 4_000, "loop": 16_000, "evidence": 18_000},
        compression_rules=_AGENT_COMPRESSION_RULES,
    ),
    "treasury_data_agent": ContextPolicy(
        token_budget=32_000,
        protected_paths=_AGENT_PROTECTED_PATHS,
        segment_limits={"session": 8_000, "sub_agent": 4_000, "loop": 16_000, "evidence": 18_000},
        compression_rules=_AGENT_COMPRESSION_RULES,
    ),
    "treasury_operation_agent": ContextPolicy(
        token_budget=32_000,
        protected_paths=_AGENT_PROTECTED_PATHS,
        segment_limits={"session": 8_000, "sub_agent": 4_000, "loop": 16_000, "evidence": 18_000},
        compression_rules=_AGENT_COMPRESSION_RULES,
    ),
}


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else dump_payload(value)
    cjk_chars = sum(1 for char in text if _is_cjk(char))
    non_cjk_chars = len(text) - cjk_chars
    return cjk_chars + math.ceil(non_cjk_chars / 4)


def dump_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class ContextBuilder:
    def __init__(
        self,
        policies: dict[str, ContextPolicy | dict[str, Any]] | None = None,
        summarizer: Any | None = None,
        model_context_windows: dict[str, int] | None = None,
        model_label: str | None = None,
    ):
        from finpilot.config import settings

        if summarizer is None:
            from finpilot.context.summarization import ContextSummarizationService

            summarizer = ContextSummarizationService()
        self.summarizer = summarizer
        self.model_context_windows = dict(
            settings.model_context_windows if model_context_windows is None else model_context_windows
        )
        self.model_label = model_label or f"{settings.ai_provider}:{settings.ai_model_name}"
        for name, window in self.model_context_windows.items():
            if int(window) <= 0:
                raise ValueError(f"model context window must be positive: {name}")
        self.policies = {name: policy.model_copy(deep=True) for name, policy in DEFAULT_CONTEXT_POLICIES.items()}
        for name, override in _configured_policy_overrides().items():
            self._register_policy(name, override)
        for name, override in (policies or {}).items():
            self._register_policy(name, override)

    def build(
        self,
        policy_name: str,
        segments: list[ContextSegment],
        *,
        stage: Literal["global", "decision", "answer"] = "decision",
        query: str = "",
        prompt_renderer: Callable[[dict[str, Any]], str] | None = None,
        summary_cache: dict[str, Any] | None = None,
    ) -> PromptContextBundle:
        if policy_name not in self.policies:
            raise ValueError(f"unknown context policy: {policy_name}")
        policy = self.policies[policy_name]
        payload = {
            segment.name: _to_jsonable(segment.value)
            for segment in sorted(segments, key=lambda item: item.priority)
        }
        runtime_policy = policy.model_copy(
            update={"protected_paths": _resolve_protected_paths(payload, policy.protected_paths)}
        )
        render = prompt_renderer or dump_payload
        payload_tokens_before = estimate_tokens(payload)
        prompt_tokens_before = estimate_tokens(render(payload))
        effective_input_budget, model_window_source = self._effective_input_budget(policy)
        trigger_tokens = int(effective_input_budget * policy.trigger_ratio)
        prompt_overhead_tokens = estimate_tokens(render({}))
        payload_trigger_tokens = max(trigger_tokens - prompt_overhead_tokens, 1)
        payload_hard_budget = max(effective_input_budget - prompt_overhead_tokens, 1)
        events: list[dict[str, Any]] = []

        if prompt_tokens_before > trigger_tokens:
            payload = copy.deepcopy(payload)
            self._compress(
                payload,
                runtime_policy,
                events,
                payload_trigger_tokens,
                payload_hard_budget,
                stage=stage,
                query=query,
                summary_cache=summary_cache if summary_cache is not None else {},
            )

        for event in events:
            event.setdefault("stage", stage)
            event.setdefault("policy", policy_name)
            event.setdefault(
                "method",
                "llm" if event.get("action") in {"llm_summarized", "summarization_failed"} else "deterministic",
            )
            event.setdefault("step_index", None)
            event.setdefault("fallback_reason", None)

        after_tokens = estimate_tokens(payload)
        estimated_prompt_tokens = estimate_tokens(render(payload))
        within_budget = estimated_prompt_tokens <= effective_input_budget
        usage = ContextUsage(
            requested_policy=policy_name,
            effective_policy=policy_name,
            stage=stage,
            token_budget=policy.token_budget,
            effective_input_budget=effective_input_budget,
            reserved_output_tokens=policy.reserved_output_tokens,
            trigger_tokens=trigger_tokens,
            estimated_tokens_before=payload_tokens_before,
            estimated_tokens_after=after_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            compressed=after_tokens < payload_tokens_before,
            within_budget=within_budget,
            model_window_source=model_window_source,
        )
        return PromptContextBundle(
            status="ready" if within_budget else "over_budget",
            payload=payload,
            usage=usage,
            compression_events=events,
            over_budget_paths=_protected_over_budget_paths(
                payload,
                runtime_policy.protected_paths,
                effective_input_budget,
            ),
        )

    def _register_policy(self, name: str, override: ContextPolicy | dict[str, Any]) -> None:
        if isinstance(override, ContextPolicy):
            self.policies[name] = override.model_copy(deep=True)
            return
        base = self.policies.get(name)
        if base is None:
            self.policies[name] = ContextPolicy.model_validate(override)
            return
        self.policies[name] = _merge_policy(base, override)

    def _effective_input_budget(self, policy: ContextPolicy) -> tuple[int, str]:
        model_name = self.model_label.split(":", 1)[-1]
        raw_window = self.model_context_windows.get(self.model_label) or self.model_context_windows.get(model_name)
        if raw_window is None:
            return policy.token_budget, "policy_fallback"
        window = int(raw_window)
        available = window - policy.reserved_output_tokens
        if available <= 0:
            raise ValueError(
                f"reserved_output_tokens must be smaller than model context window: {self.model_label}"
            )
        return min(policy.token_budget, available), "model_context_windows"

    def _compress(
        self,
        payload: dict[str, Any],
        policy: ContextPolicy,
        events: list[dict[str, Any]],
        trigger_tokens: int,
        hard_budget_tokens: int,
        *,
        stage: Literal["global", "decision", "answer"],
        query: str,
        summary_cache: dict[str, Any],
    ) -> None:
        for rule in sorted(policy.compression_rules, key=lambda item: item.priority):
            changed = self._apply_rule(
                payload,
                policy,
                rule,
                events,
                stage=stage,
                query=query,
                summary_cache=summary_cache,
            )
            if changed and estimate_tokens(payload) <= trigger_tokens:
                return

        self._enforce_segment_limits(payload, policy, events)
        if estimate_tokens(payload) > hard_budget_tokens:
            self._truncate_strings(payload, policy, events)

    def _apply_rule(
        self,
        payload: dict[str, Any],
        policy: ContextPolicy,
        rule: CompressionRule,
        events: list[dict[str, Any]],
        *,
        stage: Literal["global", "decision", "answer"],
        query: str,
        summary_cache: dict[str, Any],
    ) -> bool:
        if rule.method == "llm" and rule.path == "loop.tool_outputs":
            return self._apply_llm_tool_output_rule(
                payload,
                policy,
                rule,
                events,
                stage=stage,
                query=query,
                summary_cache=summary_cache,
            )
        current_value = _rule_value(payload, rule.path)
        before_tokens = estimate_tokens(current_value) if current_value is not None else 0
        if rule.method == "llm":
            try:
                return self._apply_llm_rule(
                    payload,
                    policy,
                    rule,
                    events,
                    stage=stage,
                    query=query,
                    summary_cache=summary_cache,
                )
            except Exception as exc:
                events.append(
                    {
                        "stage": stage,
                        "path": rule.path,
                        "action": "summarization_failed",
                        "method": "llm",
                        "before_tokens": before_tokens,
                        "after_tokens": estimate_tokens(_rule_value(payload, rule.path)) if current_value is not None else 0,
                        "fallback_reason": str(exc),
                    }
                )
        return self._apply_deterministic_rule(payload, rule, events)

    def _apply_llm_tool_output_rule(
        self,
        payload: dict[str, Any],
        policy: ContextPolicy,
        rule: CompressionRule,
        events: list[dict[str, Any]],
        *,
        stage: Literal["global", "decision", "answer"],
        query: str,
        summary_cache: dict[str, Any],
    ) -> bool:
        targets = _tool_document_outputs(payload)
        if not targets:
            return False
        target_tokens = max(rule.target_tokens // len(targets), 64)
        changed = False
        for index, output, documents in targets:
            if estimate_tokens(documents) <= target_tokens:
                continue
            path = f"loop.step_history[{index}].observation.output.documents"
            before = estimate_tokens(documents)
            try:
                summary = self.summarizer.summarize(
                    query=query,
                    path=path,
                    value=documents,
                    target_tokens=target_tokens,
                    chunk_token_budget=policy.summary_chunk_token_budget,
                    cache=summary_cache,
                )
                summary_payload = {"compressed": True, **summary.model_dump(mode="json")}
                summary_tokens = estimate_tokens(summary_payload)
                if summary_tokens > target_tokens:
                    raise ValueError(f"LLM summary exceeds target tokens: {summary_tokens} > {target_tokens}")
                output.pop("documents", None)
                output["documents_summary"] = [summary_payload]
                output["document_count"] = len(documents)
                after = estimate_tokens(output["documents_summary"])
                events.append(
                    {
                        "stage": stage,
                        "path": path,
                        "action": "llm_summarized",
                        "method": "llm",
                        "before_tokens": before,
                        "after_tokens": after,
                    }
                )
                changed = True
            except Exception as exc:
                fallback_changed = _summarize_document_output(output)
                after = estimate_tokens(output.get("documents_summary", documents))
                events.append(
                    {
                        "stage": stage,
                        "path": path,
                        "action": "summarization_failed",
                        "method": "llm",
                        "before_tokens": before,
                        "after_tokens": after,
                        "fallback_reason": str(exc),
                    }
                )
                changed = fallback_changed or changed
        return changed

    def _apply_llm_rule(
        self,
        payload: dict[str, Any],
        policy: ContextPolicy,
        rule: CompressionRule,
        events: list[dict[str, Any]],
        *,
        stage: Literal["global", "decision", "answer"],
        query: str,
        summary_cache: dict[str, Any],
    ) -> bool:
        value = _rule_value(payload, rule.path)
        if value is None:
            return False
        preserved: list[Any] = []
        summarizable = value
        if isinstance(value, list) and rule.preserve_last > 0:
            preserved = copy.deepcopy(value[-rule.preserve_last :])
            summarizable = value[: -rule.preserve_last]
            if not summarizable:
                return False
        before = estimate_tokens(value)
        if estimate_tokens(summarizable) <= rule.target_tokens:
            return False

        summary = self.summarizer.summarize(
            query=query,
            path=rule.path,
            value=summarizable,
            target_tokens=rule.target_tokens,
            chunk_token_budget=policy.summary_chunk_token_budget,
            cache=summary_cache,
        )
        summary_payload = {"compressed": True, **summary.model_dump(mode="json")}
        if estimate_tokens(summary_payload) > rule.target_tokens:
            raise ValueError(
                f"LLM summary exceeds target tokens: {estimate_tokens(summary_payload)} > {rule.target_tokens}"
            )
        replacement: Any = [summary_payload, *preserved] if isinstance(value, list) else summary_payload
        _replace_rule_value(payload, rule.path, replacement)
        after = estimate_tokens(_rule_value(payload, rule.path))
        events.append(
            {
                "stage": stage,
                "path": rule.path,
                "action": "llm_summarized",
                "method": "llm",
                "before_tokens": before,
                "after_tokens": after,
            }
        )
        return after < before

    def _apply_deterministic_rule(
        self,
        payload: dict[str, Any],
        rule: CompressionRule,
        events: list[dict[str, Any]],
    ) -> bool:
        if rule.path == "loop.tool_outputs":
            return _summarize_loop_tool_outputs(payload, events)
        if rule.path == "loop.old_step_history":
            return _summarize_old_step_history(payload, events)
        if rule.path == "session.recent_messages":
            return _limit_list(
                payload,
                ["session", "recent_messages"],
                max(rule.preserve_last, 1),
                max(rule.target_tokens, 1) * 4,
                rule.name,
                events,
            )
        if rule.path == "session.semantic_memory":
            return _limit_list(
                payload,
                ["session", "semantic_memory"],
                5,
                max(rule.target_tokens, 1) * 4,
                rule.name,
                events,
            )
        if rule.path in payload:
            return _compress_top_level(payload, rule.path, events)
        return False

    def _enforce_segment_limits(
        self,
        payload: dict[str, Any],
        policy: ContextPolicy,
        events: list[dict[str, Any]],
    ) -> None:
        for segment_name, limit in policy.segment_limits.items():
            segment = payload.get(segment_name)
            if segment is None or estimate_tokens(segment) <= limit:
                continue
            before = estimate_tokens(segment)
            _truncate_value(segment, max_chars=600, protected_paths=policy.protected_paths, path=segment_name)
            after = estimate_tokens(segment)
            if after < before:
                events.append(
                    {
                        "path": segment_name,
                        "action": "enforced_segment_limit",
                        "before_tokens": before,
                        "after_tokens": after,
                    }
                )

    def _truncate_strings(self, payload: dict[str, Any], policy: ContextPolicy, events: list[dict[str, Any]]) -> None:
        before = estimate_tokens(payload)
        _truncate_value(payload, max_chars=400, protected_paths=policy.protected_paths, path="")
        after = estimate_tokens(payload)
        if after < before:
            events.append(
                {
                    "path": "*",
                    "action": "truncated_strings",
                    "before_tokens": before,
                    "after_tokens": after,
                }
            )


def _summarize_loop_tool_outputs(payload: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    history = payload.get("loop", {}).get("step_history", [])
    if not isinstance(history, list):
        return False
    changed = False
    for index, step in enumerate(history):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        output = observation.get("output")
        if not isinstance(output, dict) or not isinstance(output.get("documents"), list):
            continue
        before = estimate_tokens(output)
        _summarize_document_output(output)
        after = estimate_tokens(output)
        events.append(
            {
                "path": f"loop.step_history[{index}].observation.output.documents",
                "action": "summarized_tool_output",
                "before_tokens": before,
                "after_tokens": after,
            }
        )
        changed = True
    return changed


def _summarize_document_output(output: dict[str, Any]) -> bool:
    documents = output.get("documents")
    if not isinstance(documents, list):
        return False
    output.pop("documents", None)
    output["documents_summary"] = [_document_summary(item) for item in documents if isinstance(item, dict)]
    output["document_count"] = len(documents)
    return True


def _summarize_old_step_history(payload: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    history = payload.get("loop", {}).get("step_history", [])
    if not isinstance(history, list) or len(history) <= 1:
        return False
    changed = False
    for index, step in enumerate(history[:-1]):
        if not isinstance(step, dict):
            continue
        before = estimate_tokens(step)
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        decision = step.get("decision") if isinstance(step.get("decision"), dict) else {}
        history[index] = {
            "step_index": step.get("step_index"),
            "decision": decision.get("decision"),
            "tool_name": decision.get("tool_name") or observation.get("tool_name"),
            "status": observation.get("status"),
            "observation_summary": observation.get("summary"),
        }
        after = estimate_tokens(history[index])
        if after < before:
            events.append(
                {
                    "path": f"loop.step_history[{index}]",
                    "action": "summarized_old_step",
                    "before_tokens": before,
                    "after_tokens": after,
                }
            )
            changed = True
    return changed


def _limit_list(
    payload: dict[str, Any],
    path_parts: list[str],
    keep_last: int,
    max_chars: int,
    action: str,
    events: list[dict[str, Any]],
) -> bool:
    parent = payload
    for part in path_parts[:-1]:
        parent = parent.get(part, {}) if isinstance(parent, dict) else {}
    key = path_parts[-1]
    values = parent.get(key) if isinstance(parent, dict) else None
    if not isinstance(values, list) or not values:
        return False
    before = estimate_tokens(values)
    limited = values[-keep_last:]
    parent[key] = [_truncate_copy(item, max_chars) for item in limited]
    after = estimate_tokens(parent[key])
    if after >= before:
        return False
    events.append(
        {
            "path": ".".join(path_parts),
            "action": action,
            "before_tokens": before,
            "after_tokens": after,
        }
    )
    return True


def _compress_top_level(payload: dict[str, Any], key: str, events: list[dict[str, Any]]) -> bool:
    if key not in payload:
        return False
    before = estimate_tokens(payload[key])
    payload[key] = _truncate_copy(payload[key], 800)
    after = estimate_tokens(payload[key])
    if after >= before:
        return False
    events.append({"path": key, "action": f"compressed_{key}", "before_tokens": before, "after_tokens": after})
    return True


def _document_summary(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text") or "")
    return {
        "document_id": item.get("document_id"),
        "title": item.get("title"),
        "source": item.get("source"),
        "score": item.get("score"),
        "text_preview": _shorten(text, 240) if text else "",
    }


def _truncate_copy(value: Any, max_chars: int) -> Any:
    copied = copy.deepcopy(value)
    _truncate_value(copied, max_chars=max_chars, protected_paths=[], path="")
    return copied


def _truncate_value(value: Any, *, max_chars: int, protected_paths: list[str], path: str) -> Any:
    if _is_protected(path, protected_paths):
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            child_path = f"{path}.{key}" if path else str(key)
            value[key] = _truncate_value(item, max_chars=max_chars, protected_paths=protected_paths, path=child_path)
    elif isinstance(value, list):
        value_len = len(value)
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            negative_child_path = f"{path}[-{value_len - index}]"
            if _is_protected(negative_child_path, protected_paths):
                continue
            value[index] = _truncate_value(item, max_chars=max_chars, protected_paths=protected_paths, path=child_path)
    elif isinstance(value, str):
        return _shorten(value, max_chars)
    return value


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...[truncated]"


def _is_protected(path: str, protected_paths: list[str]) -> bool:
    for protected in protected_paths:
        pattern = re.escape(protected).replace(r"\[\*\]", r"\[-?\d+\]")
        if re.fullmatch(pattern, path) or re.match(pattern + r"(?:\.|\[)", path):
            return True
    return False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _merge_policy(base: ContextPolicy, override: dict[str, Any]) -> ContextPolicy:
    merged = base.model_dump(mode="python")
    raw = copy.deepcopy(override)

    if "protected_paths" in raw:
        raw["protected_paths"] = list(dict.fromkeys([*base.protected_paths, *raw["protected_paths"]]))
    if "segment_limits" in raw:
        raw["segment_limits"] = {**base.segment_limits, **raw["segment_limits"]}
    if "compression_rules" in raw:
        rules = {rule.name: rule.model_dump(mode="python") for rule in base.compression_rules}
        for rule in raw["compression_rules"]:
            rule_data = rule.model_dump(mode="python") if isinstance(rule, CompressionRule) else dict(rule)
            rule_name = str(rule_data.get("name") or "")
            if not rule_name:
                raise ValueError("compression rule name is required")
            rules[rule_name] = {**rules.get(rule_name, {}), **rule_data}
        raw["compression_rules"] = list(rules.values())

    merged.update(raw)
    return ContextPolicy.model_validate(merged)


def _configured_policy_overrides() -> dict[str, dict[str, Any]]:
    from finpilot.config import settings

    configured = getattr(settings, "context_policies", {}) or {}
    return {name: dict(raw_policy) for name, raw_policy in configured.items()}


def _resolve_protected_paths(payload: dict[str, Any], protected_paths: list[str]) -> list[str]:
    return [_resolve_negative_path(payload, path) for path in protected_paths]


def _resolve_negative_path(payload: dict[str, Any], path: str) -> str:
    current: Any = payload
    resolved: list[str] = []
    for part in path.split("."):
        match = re.fullmatch(r"([^\[]+)(?:\[(-?\d+|\*)\])?", part)
        if match is None or not isinstance(current, dict):
            return path
        key, raw_index = match.groups()
        current = current.get(key)
        resolved_part = key
        if raw_index is not None:
            if raw_index == "*":
                resolved.append(f"{key}[*]")
                current = current[0] if isinstance(current, list) and current else None
                continue
            if not isinstance(current, list):
                return path
            index = int(raw_index)
            actual_index = index if index >= 0 else len(current) + index
            if actual_index < 0 or actual_index >= len(current):
                return path
            resolved_part = f"{key}[{actual_index}]"
            current = current[actual_index]
        resolved.append(resolved_part)
    return ".".join(resolved)


def _protected_over_budget_paths(
    payload: dict[str, Any],
    protected_paths: list[str],
    effective_input_budget: int,
) -> list[str]:
    over_budget: list[str] = []
    for path in protected_paths:
        values = _lookup_path_values(payload, path)
        if values and estimate_tokens(values[0] if len(values) == 1 else values) > effective_input_budget:
            over_budget.append(path)
    return over_budget


def _lookup_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        match = re.fullmatch(r"([^\[]+)(?:\[(-?\d+)\])?", part)
        if match is None or not isinstance(current, dict):
            return None
        current = current.get(match.group(1))
        index = match.group(2)
        if index is not None:
            if not isinstance(current, list):
                return None
            try:
                current = current[int(index)]
            except IndexError:
                return None
    return current


def _lookup_path_values(payload: Any, path: str) -> list[Any]:
    parts = path.split(".")

    def walk(current: Any, remaining: list[str]) -> list[Any]:
        if not remaining:
            return [current]
        if not isinstance(current, dict):
            return []
        match = re.fullmatch(r"([^\[]+)(?:\[(-?\d+|\*)\])?", remaining[0])
        if match is None:
            return []
        value = current.get(match.group(1))
        raw_index = match.group(2)
        if raw_index is None:
            return walk(value, remaining[1:])
        if not isinstance(value, list):
            return []
        if raw_index == "*":
            results: list[Any] = []
            for item in value:
                results.extend(walk(item, remaining[1:]))
            return results
        try:
            selected = value[int(raw_index)]
        except IndexError:
            return []
        return walk(selected, remaining[1:])

    return walk(payload, parts)


def _rule_value(payload: dict[str, Any], path: str) -> Any:
    if path == "loop.tool_outputs":
        documents: list[dict[str, Any]] = []
        history = payload.get("loop", {}).get("step_history", [])
        for step in history if isinstance(history, list) else []:
            observation = step.get("observation") if isinstance(step, dict) else None
            output = observation.get("output") if isinstance(observation, dict) else None
            values = output.get("documents") if isinstance(output, dict) else None
            if isinstance(values, list):
                documents.extend(item for item in values if isinstance(item, dict))
        return documents or None
    return _lookup_path(payload, path)


def _tool_document_outputs(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any], list[Any]]]:
    targets: list[tuple[int, dict[str, Any], list[Any]]] = []
    history = payload.get("loop", {}).get("step_history", [])
    for index, step in enumerate(history if isinstance(history, list) else []):
        observation = step.get("observation") if isinstance(step, dict) else None
        output = observation.get("output") if isinstance(observation, dict) else None
        documents = output.get("documents") if isinstance(output, dict) else None
        if isinstance(documents, list):
            targets.append((index, output, documents))
    return targets


def _replace_rule_value(payload: dict[str, Any], path: str, replacement: Any) -> None:
    if path == "loop.tool_outputs":
        history = payload.get("loop", {}).get("step_history", [])
        for step in history if isinstance(history, list) else []:
            observation = step.get("observation") if isinstance(step, dict) else None
            output = observation.get("output") if isinstance(observation, dict) else None
            documents = output.get("documents") if isinstance(output, dict) else None
            if not isinstance(documents, list):
                continue
            output.pop("documents", None)
            output["documents_summary"] = copy.deepcopy(replacement if isinstance(replacement, list) else [replacement])
            output["document_count"] = len(documents)
        return

    parts = path.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(part)
    if isinstance(current, dict):
        current[parts[-1]] = replacement


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" or "\uac00" <= char <= "\ud7af"
