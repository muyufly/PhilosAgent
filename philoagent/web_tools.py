"""DeepSeek Search & Fetch transport used by the agent's existing web tools.

The API key is sent only to DeepSeek's official HTTPS origin. Evidence contains
hashes and URL identities, never credentials or the retrieved page body.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from philoagent.evidence import append_event


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
REQUEST_TIMEOUT = (10, 45)


class DeepSeekWebError(RuntimeError):
    """A safe provider error that does not include headers or response bodies."""


def _endpoint(operation: str) -> str:
    base = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).strip().rstrip("/")
    try:
        parsed = urlsplit(base)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname == "api.deepseek.com"
            and parsed.port in (None, 443)
            and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment
            and parsed.path in ("", "/v1")
        )
    except ValueError:
        valid = False
    if not valid:
        raise DeepSeekWebError(
            "DEEPSEEK_BASE_URL must be https://api.deepseek.com (optionally ending in /v1)."
        )
    return f"{DEFAULT_DEEPSEEK_BASE_URL}/v1/tools/{operation}"


def _key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise DeepSeekWebError("DeepSeek web tools require DEEPSEEK_API_KEY or OPENAI_API_KEY.")
    return key


def _valid_url(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > 8192:
        return False
    try:
        parsed = urlsplit(value)
        # Accessing port also rejects malformed ports before sending a request.
        parsed.port
        return (
            parsed.scheme in ("http", "https") and bool(parsed.hostname)
            and parsed.username is None and parsed.password is None
            and not any(char.isspace() or ord(char) < 32 for char in value)
        )
    except ValueError:
        return False


def _evidence_url(value: Any) -> str | None:
    if not _valid_url(value):
        return None
    parsed = urlsplit(value)
    # Query strings and fragments can contain private tokens; omit them in logs.
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _http_error(operation: str, status: int) -> DeepSeekWebError:
    if status == 401:
        message = "authentication failed; check DEEPSEEK_API_KEY or OPENAI_API_KEY"
    elif status == 402:
        return DeepSeekWebError(f"TERMINATE_QUOTA: DeepSeek {operation} insufficient balance (HTTP 402).")
    elif status == 429:
        return DeepSeekWebError(f"TERMINATE_QUOTA: DeepSeek {operation} rate limit exceeded (HTTP 429).")
    elif status == 400:
        message = "invalid request parameters"
    elif status == 403:
        message = "access forbidden; check API permissions"
    elif status == 404 and operation == "fetch":
        message = "page unavailable or not found"
    elif status == 422:
        message = "request validation failed"
    elif status >= 500:
        message = "provider service unavailable"
    else:
        message = "request failed"
    return DeepSeekWebError(f"DeepSeek {operation}: {message} (HTTP {status}).")


def _post(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = _endpoint(operation)
    key = _key()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    event: dict[str, Any] = {
        "event": "web_provider_request", "backend": "deepseek", "operation": operation,
        "endpoint": endpoint, "request_payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "http_status": None, "response_sha256": None, "source_urls": [],
    }
    if operation == "fetch":
        event["source_urls"] = [_evidence_url(payload["url"])]
    try:
        try:
            response = requests.post(
                endpoint, json=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT, allow_redirects=False,
            )
        except requests.Timeout:
            raise DeepSeekWebError(f"DeepSeek {operation} request timed out.") from None
        except requests.RequestException:
            # Request exception strings can include headers/URLs. Never echo them.
            raise DeepSeekWebError(f"DeepSeek {operation} network request failed.") from None
        event["http_status"] = response.status_code
        event["response_sha256"] = hashlib.sha256(response.content).hexdigest()
        if response.status_code != 200:
            raise _http_error(operation, response.status_code)
        try:
            data = response.json()
        except ValueError:
            raise DeepSeekWebError(f"DeepSeek {operation} returned invalid JSON.") from None
        if not isinstance(data, dict):
            raise DeepSeekWebError(f"DeepSeek {operation} returned an invalid response object.")
        if operation == "search" and isinstance(data.get("results"), list):
            event["source_urls"] = [
                identity for result in data["results"] if isinstance(result, dict)
                if (identity := _evidence_url(result.get("url"))) is not None
            ]
        return data
    finally:
        evidence_root = os.getenv("PHILO_EVIDENCE_DIR", "").strip()
        if evidence_root:
            append_event(Path(evidence_root) / "web_requests.jsonl", event)


def deepseek_search(query: str, count: int = 5) -> tuple[list[dict[str, Any]], str]:
    """Return the existing FlashOAgents search result shape without inventing sources."""
    if not isinstance(query, str) or not query.strip() or len(query) > 128:
        return [], "DeepSeek search query must contain 1 to 128 Unicode characters."
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 50:
        return [], "DeepSeek search count must be an integer from 1 to 50."
    data = _post("search", {"query": query, "count": count, "need_content": False})
    if not isinstance(data.get("results"), list):
        raise DeepSeekWebError("DeepSeek search response is missing the results list.")
    results = []
    for page in data["results"]:
        if not isinstance(page, dict) or not _valid_url(page.get("url")):
            raise DeepSeekWebError("DeepSeek search returned an invalid source URL.")
        results.append({
            "idx": len(results) + 1, "title": str(page.get("title") or "No title"),
            "link": page["url"],
            "snippet": str(page.get("snippet") or page.get("summary") or "No snippet"),
            "date": f"\nDate published: {page['date']}" if page.get("date") else "",
            "source": f"\nSource: {page.get('site_name') or urlsplit(page['url']).hostname}",
        })
    return results, ""


def deepseek_fetch(url: str) -> str:
    """Fetch Markdown using the provider, preserving the requested source and title."""
    if not _valid_url(url):
        return "Error reading page: URL must be HTTP(S), at most 8192 characters, without credentials."
    data = _post("fetch", {"url": url})
    content = data.get("content")
    if not isinstance(content, str) or not content.strip():
        return "Error reading page: DeepSeek fetch returned no Markdown content."
    title = data.get("title")
    heading = f"# {title}\n\n" if isinstance(title, str) and title.strip() else ""
    return f"{heading}Source: {url}\n\n{content}"
