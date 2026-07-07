from __future__ import annotations

import html
import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from finpilot.config import settings
from finpilot.models import GraphState

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: StrictStr = Field(min_length=1, max_length=1000)
    count: StrictInt = Field(default=5, ge=1, le=10)
    freshness: StrictStr | None = Field(default=None, max_length=32)


class FetchUrlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: StrictStr = Field(min_length=1, max_length=2000)
    max_chars: StrictInt | None = Field(default=None, ge=1, le=50000)


def register_web_tools(registry) -> None:
    from finpilot.agent.tools import ToolSpec

    registry.register(
        ToolSpec(
            name="web_search",
            description="Search the public web through Brave Search API and return sourced snippets.",
            when_to_use="Use when the answer needs current or external public web information.",
            arguments={"query": "string", "count": "integer, optional", "freshness": "string, optional"},
            args_model=WebSearchArgs,
            executor=_web_search,
        )
    )
    registry.register(
        ToolSpec(
            name="fetch_url",
            description="Fetch readable text from a specific public http or https URL.",
            when_to_use="Use when the user supplies or the agent already has a specific public URL to inspect.",
            arguments={"url": "string", "max_chars": "integer, optional"},
            args_model=FetchUrlArgs,
            executor=_fetch_url,
        )
    )


def _web_search(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    del state
    api_key = (settings.brave_search_api_key or "").strip()
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is required for web_search")
    params: dict[str, Any] = {
        "q": parameters["query"],
        "count": int(parameters.get("count") or 5),
        "safesearch": "moderate",
        "extra_snippets": "true",
    }
    if parameters.get("freshness"):
        params["freshness"] = parameters["freshness"]
    response = httpx.get(
        BRAVE_WEB_SEARCH_URL,
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        params=params,
        timeout=settings.web_search_timeout_seconds,
    )
    response.raise_for_status()
    results = response.json().get("web", {}).get("results", [])
    content = []
    for rank, item in enumerate(results, start=1):
        snippets = [str(item.get("description") or "")]
        snippets.extend(str(snippet) for snippet in item.get("extra_snippets", []) if snippet)
        content.append(
            {
                "source": str(item.get("url") or ""),
                "title": str(item.get("title") or item.get("url") or f"result {rank}"),
                "text": "\n".join(snippet for snippet in snippets if snippet),
                "metadata": {"kind": "web_search", "rank": rank},
            }
        )
    return {"content": content}


def _fetch_url(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    del state
    url = str(parameters["url"])
    _validate_public_url(url)
    max_chars = int(parameters.get("max_chars") or settings.web_fetch_max_chars)
    response = httpx.get(
        url,
        headers={"User-Agent": "FinPilot/0.1"},
        timeout=settings.web_search_timeout_seconds,
        follow_redirects=True,
    )
    response.raise_for_status()
    text = _extract_text(response.text)
    truncated = len(text) > max_chars
    text = text[:max_chars]
    return {
        "content": [
            {
                "source": url,
                "title": url,
                "text": text,
                "metadata": {
                    "kind": "web_page",
                    "content_type": response.headers.get("content-type", ""),
                    "truncated": truncated,
                },
            }
        ]
    }


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL is not allowed: only public http and https URLs are supported")
    host = parsed.hostname.strip().lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("URL is not allowed: local network targets are blocked")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_reserved:
        raise ValueError("URL is not allowed: local network targets are blocked")


def _extract_text(value: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()

