from __future__ import annotations

import copy
import json
import math
from typing import Any

from pydantic import BaseModel, Field


class ContextSegment(BaseModel):
    name: str
    value: Any
    priority: int = 50


class ContextPolicy(BaseModel):
    token_budget: int
    trigger_ratio: float = 0.85
    strategy: str = "deterministic"
    protected_paths: list[str] = Field(default_factory=list)
    segment_limits: dict[str, int] = Field(default_factory=dict)
    compression_order: list[str] = Field(default_factory=list)


class PromptContextBundle(BaseModel):
    payload: dict[str, Any]
    usage: dict[str, Any]
    compression_events: list[dict[str, Any]]


DEFAULT_CONTEXT_POLICIES: dict[str, ContextPolicy] = {
    "global_default": ContextPolicy(
        token_budget=64_000,
        protected_paths=["session.user_message", "route", "subagent_results"],
        compression_order=["global_history", "completed_subagents", "memory"],
    ),
    "agent_default": ContextPolicy(
        token_budget=32_000,
        protected_paths=["session.user_message", "sub_agent.agent_name", "sub_agent.goal"],
        segment_limits={"session": 8_000, "sub_agent": 4_000, "loop": 18_000},
        compression_order=[
            "loop.tool_outputs",
            "loop.old_step_history",
            "session.recent_messages",
            "session.semantic_memory",
            "session.long_term_memory",
        ],
    ),
    "finance_qa_agent": ContextPolicy(
        token_budget=32_000,
        protected_paths=[
            "session.user_message",
            "sub_agent.agent_name",
            "sub_agent.goal",
            "sub_agent.allowed_tools",
            "loop.step_history[-1]",
        ],
        segment_limits={"session": 8_000, "sub_agent": 4_000, "loop": 18_000},
        compression_order=[
            "loop.tool_outputs",
            "loop.old_step_history",
            "session.recent_messages",
            "session.semantic_memory",
            "session.long_term_memory",
        ],
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
    def __init__(self, policies: dict[str, ContextPolicy] | None = None):
        self.policies = dict(DEFAULT_CONTEXT_POLICIES)
        self.policies.update(_configured_policies())
        if policies:
            self.policies.update(policies)

    def build(self, policy_name: str, segments: list[ContextSegment]) -> PromptContextBundle:
        policy = self.policies.get(policy_name) or self.policies["agent_default"]
        payload = {segment.name: _to_jsonable(segment.value) for segment in sorted(segments, key=lambda item: item.priority)}
        before_tokens = estimate_tokens(payload)
        trigger_tokens = int(policy.token_budget * policy.trigger_ratio)
        events: list[dict[str, Any]] = []

        if before_tokens > trigger_tokens:
            payload = copy.deepcopy(payload)
            self._compress(payload, policy, events, trigger_tokens)

        after_tokens = estimate_tokens(payload)
        usage = {
            "policy": policy_name,
            "strategy": policy.strategy,
            "token_counter": "heuristic",
            "token_budget": policy.token_budget,
            "trigger_tokens": trigger_tokens,
            "estimated_tokens_before": before_tokens,
            "estimated_tokens_after": after_tokens,
            "compressed": bool(events),
        }
        return PromptContextBundle(payload=payload, usage=usage, compression_events=events)

    def _compress(
        self,
        payload: dict[str, Any],
        policy: ContextPolicy,
        events: list[dict[str, Any]],
        trigger_tokens: int,
    ) -> None:
        for action in policy.compression_order:
            before = estimate_tokens(payload)
            changed = self._apply_action(payload, policy, action, events)
            after = estimate_tokens(payload)
            if changed and after <= trigger_tokens:
                return
            if changed and after < before:
                continue

        self._enforce_segment_limits(payload, policy, events)
        if estimate_tokens(payload) > policy.token_budget:
            self._truncate_strings(payload, policy, events)

    def _apply_action(
        self,
        payload: dict[str, Any],
        policy: ContextPolicy,
        action: str,
        events: list[dict[str, Any]],
    ) -> bool:
        if action == "loop.tool_outputs":
            return _summarize_loop_tool_outputs(payload, events)
        if action == "loop.old_step_history":
            return _summarize_old_step_history(payload, events)
        if action == "session.recent_messages":
            return _limit_list(payload, ["session", "recent_messages"], 4, 500, action, events)
        if action == "session.semantic_memory":
            return _limit_list(payload, ["session", "semantic_memory"], 5, 400, action, events)
        if action == "session.long_term_memory":
            return _limit_list(payload, ["session", "long_term_memory"], 5, 400, action, events)
        return _compress_top_level(payload, action, events)

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
        documents = output.pop("documents")
        output["documents_summary"] = [_document_summary(item) for item in documents if isinstance(item, dict)]
        output["document_count"] = len(documents)
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
    return any(path == protected or path.startswith(protected + ".") for protected in protected_paths)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _configured_policies() -> dict[str, ContextPolicy]:
    try:
        from finpilot.config import settings
    except Exception:
        return {}
    configured = getattr(settings, "context_policies", {}) or {}
    policies: dict[str, ContextPolicy] = {}
    for name, raw_policy in configured.items():
        try:
            policies[name] = raw_policy if isinstance(raw_policy, ContextPolicy) else ContextPolicy.model_validate(raw_policy)
        except Exception:
            continue
    return policies


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" or "\uac00" <= char <= "\ud7af"
