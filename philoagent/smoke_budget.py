"""Fail-closed accounting for one serial, real-model smoke experiment.

The request count is a hard cap. Tokens use a deliberately conservative admission
estimate (serialized request UTF-8 bytes + output cap + 4,096 overhead), followed
by provider-reported settlement. It is a SOFT STOP, not a tokenizer-independent
hard cap: one admitted request can exceed its estimate, and an interrupted or
failed request can have unknown usage. Either condition prevents further calls.
This module does not load credentials, choose models, or perform any API calls.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .live_smoke import AuditedClient


class BudgetRefused(RuntimeError):
    """No request was sent because the smoke budget cannot authorize it."""


class BudgetCallError(RuntimeError):
    """Sanitized failure which also bypasses the native model's SDK retries."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    values = usage if isinstance(usage, dict) else (
        usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)
    )
    prompt, completion = values.get("prompt_tokens"), values.get("completion_tokens")
    if any(type(value) is not int or value < 0 for value in (prompt, completion)):
        return None
    total = values.get("total_tokens", prompt + completion)
    if type(total) is not int or total < prompt + completion:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _response_record(response):
    if response is None:
        return None
    choices = []
    for choice in response.choices:
        message = choice.message.model_dump()
        # Store model output needed for review, never SDK/client configuration.
        choices.append({"finish_reason": getattr(choice, "finish_reason", None),
                        "message": {key: message[key] for key in
                                    ("role", "content", "tool_calls", "reasoning_content")
                                    if key in message}})
    return {"choices": choices}


class Budget:
    """Persistent budget shared by clients in a single serial Python process.

    Reopening a completed ledger is supported. A pending request means its
    outcome is unknown and locks the ledger; there is intentionally no automatic
    reset/retry operation. Concurrent processes and concurrent calls are not
    supported.
    """

    safety_overhead = 4096

    def __init__(self, root, max_calls=60, max_tokens=250000):
        if any(type(value) is not int or value <= 0 for value in (max_calls, max_tokens)):
            raise ValueError("Budget limits must be positive integers")
        self.root = Path(root).resolve()
        self.path = self.root / "budget.json"
        limits = {"max_calls": max_calls, "max_tokens": max_tokens}
        if self.path.exists():
            self._state = json.loads(self.path.read_text(encoding="utf-8"))
            if self._state.get("limits") != limits:
                raise BudgetRefused("Existing budget limits cannot be changed")
            if any(item["status"] == "started" for item in self._state["attempts"]):
                self._state["blocked_reason"] = "unfinished_attempt_usage_unknown"
                self._save()
        else:
            self._state = {"version": 1, "limits": limits, "model": None,
                           "request_count_scope": "hard_cap_before_network",
                           "token_limit_scope": "soft_stop_with_conservative_reservation",
                           "token_estimate": "serialized_request_utf8_bytes + max_output + 4096",
                           "token_tail_limit": "At most one admitted request beyond the estimate; no tokenizer-independent numeric bound. Failed or interrupted request usage may be unknown.",
                           "actual_tokens": 0, "unknown_usage_attempts": 0,
                           "blocked_reason": None, "attempts": []}
            self._save()

    @property
    def state(self):
        self._refresh()
        return deepcopy(self._state)

    def summary(self):
        state = self.state
        state["attempt_count"] = len(state.pop("attempts"))
        state["ledger_path"] = str(self.path)
        return state

    def _save(self):
        try:
            _atomic_json(self.path, self._state)
        except Exception:
            self._state["blocked_reason"] = "ledger_persistence_failed"
            raise BudgetRefused("Ledger persistence failed") from None

    def _refresh(self):
        # A serial child process may have used the same ledger since this
        # object was constructed. Never overwrite those completed attempts.
        if self._state.get("blocked_reason") == "ledger_persistence_failed":
            return
        try:
            disk = json.loads(self.path.read_text(encoding="utf-8"))
            if disk["limits"] != self._state["limits"]:
                raise ValueError("limits changed")
            self._state = disk
        except Exception:
            self._state["blocked_reason"] = "ledger_persistence_failed"
            raise BudgetRefused("Ledger could not be reloaded") from None

    def bind_model(self, model_id):
        self._refresh()
        if not isinstance(model_id, str) or not model_id:
            raise BudgetRefused("An existing model_id is required")
        current = self._state["model"]
        if current is not None and current != model_id:
            raise BudgetRefused("Model switching is forbidden in this smoke budget")
        if current is None:
            self._state["model"] = model_id
            self._save()

    def begin(self, request, *, label, request_sha256, reservation):
        self._refresh()
        state = self._state
        if state["blocked_reason"]:
            raise BudgetRefused("Smoke budget is blocked: " + state["blocked_reason"])
        if any(item["status"] == "started" for item in state["attempts"]):
            raise BudgetRefused("Only serial requests are permitted")
        if len(state["attempts"]) >= state["limits"]["max_calls"]:
            raise BudgetRefused("Request hard cap reached")
        if state["actual_tokens"] + reservation > state["limits"]["max_tokens"]:
            raise BudgetRefused("Token reservation exceeds remaining soft-stop allowance")
        number = len(state["attempts"]) + 1
        item = {"attempt": number, "label": label, "status": "started",
                "request_sha256": request_sha256, "reserved_tokens": reservation,
                "max_output": request["max_tokens"], "audit_path": f"attempts/{number:04d}.json"}
        state["attempts"].append(item)
        # Persist attempt + request hash before either network dispatch or audit
        # creation. A crash after this write prevents retry on reopening.
        self._save()
        try:
            _atomic_json(self.root / item["audit_path"], {**item, "request": request})
        except Exception:
            state["blocked_reason"] = "request_audit_persistence_failed"
            self._save()
            raise BudgetRefused("Request audit persistence failed") from None
        return number

    def finish(self, number, *, response, usage, error_type=None):
        item = self._state["attempts"][number - 1]
        if item["status"] != "started":
            raise BudgetRefused("Attempt has already been settled")
        if usage is None:
            self._state["unknown_usage_attempts"] += 1
            error_type = error_type or "MissingOrInvalidUsage"
        else:
            self._state["actual_tokens"] += usage["total_tokens"]
            if usage["total_tokens"] > item["reserved_tokens"]:
                error_type = error_type or "TokenReservationExceeded"
            if usage["completion_tokens"] > item["max_output"]:
                error_type = error_type or "OutputTokenLimitExceeded"
        item.update(status="error" if error_type else "success", usage=usage)
        if error_type:
            item["error_type"] = error_type
            self._state["blocked_reason"] = error_type
        try:
            audit_path = self.root / item["audit_path"]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.update(item)
            audit["response"] = _response_record(response)
            _atomic_json(audit_path, audit)
        except Exception:
            self._state["blocked_reason"] = "response_audit_persistence_failed"
            self._save()
            raise BudgetRefused("Response audit persistence failed") from None
        self._save()
        return error_type


class _BudgetedAuditedClient(AuditedClient):
    def __init__(self, client, budget, snapshot, *, label, model_id, max_output):
        super().__init__(client, budget.root / "api_calls.jsonl", snapshot.items,
                         label=label, timeout=90, max_tokens=max_output)
        self.budget = budget
        self.model_id = model_id
        self._transport = self.client
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self._send)))
        self._response = None

    def _send(self, **kwargs):
        self._response = self._transport.chat.completions.create(**kwargs)
        return self._response

    def create(self, **kwargs):
        if os.environ.get("PHILO_SMOKE_THINKING", "disabled") != "disabled":
            raise BudgetRefused("This smoke requires PHILO_SMOKE_THINKING=disabled")
        requested_model = kwargs.get("model")
        self.budget.bind_model(requested_model)
        if self.model_id is not None and requested_model != self.model_id:
            raise BudgetRefused("Model switching is forbidden in this smoke budget")
        if kwargs.get("stream"):
            raise BudgetRefused("Streaming cannot be audited by this serial smoke")
        requested_output = kwargs.get("max_tokens", self.max_tokens)
        if type(requested_output) is not int or requested_output <= 0:
            raise BudgetRefused("max_tokens must be a positive integer")
        kwargs["max_tokens"] = min(requested_output, self.max_tokens)
        if "max_completion_tokens" in kwargs:
            raise BudgetRefused("Use max_tokens for this smoke's audited output cap")
        kwargs["temperature"] = 0
        # Prevent a caller from overriding the configured SDK timeout.
        kwargs.pop("timeout", None)
        if set(kwargs.get("extra_body") or {}).intersection(
                {"model", "messages", "temperature", "max_tokens", "max_completion_tokens", "stream"}):
            raise BudgetRefused("extra_body cannot override audited request fields")
        if requested_model.lower().startswith("deepseek"):
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "thinking": {"type": "disabled"}}
        request = {key: kwargs.get(key) for key in ("messages", "model", "max_tokens", "temperature")}
        serialized = _json(kwargs).encode("utf-8")
        reservation = len(serialized) + kwargs["max_tokens"] + self.budget.safety_overhead
        number = self.budget.begin(request, label=self.label,
                                   request_sha256=hashlib.sha256(serialized).hexdigest(), reservation=reservation)
        self._response = None
        error_type = None
        try:
            super().create(**kwargs)
        except Exception as exc:
            error_type = type(exc).__name__
        try:
            usage = _usage(self._response)
        except Exception:
            usage = None
        error_type = self.budget.finish(number, response=self._response, usage=usage, error_type=error_type)
        if error_type:
            # No provider exception message/URL survives, and native SDK-specific
            # exception retry branches cannot issue another request.
            raise BudgetCallError(error_type) from None
        return self._response


def wrap_native_model(model, budget, snapshot, label, max_output=4096):
    """Wrap an already configured model in place; return the same model."""
    if type(max_output) is not int or max_output <= 0:
        raise ValueError("max_output must be a positive integer")
    if isinstance(model.client, _BudgetedAuditedClient):
        raise ValueError("Model is already wrapped")
    model_id = getattr(model, "model_id", None)
    if model_id is not None:
        budget.bind_model(model_id)
    model.client = _BudgetedAuditedClient(model.client, budget, snapshot, label=label,
                                        model_id=model_id, max_output=max_output)
    return model
