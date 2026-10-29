"""Shared, durable accounting at the SDK boundary of the HSS experiment.

Calls are recorded under a short ledger lock and occupy cross-process slots.
The default policy enforces whole-experiment limits and stops on unknown cost.
An explicit ledger ``enforcement_policy.mode=observe_only`` keeps accounting
while allowing work past limits, unknown usage and dead request owners.

This module neither loads credentials nor sends requests during installation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from unittest.mock import patch
from uuid import uuid4

from .ablation_runtime import (
    AuditSink, AuditedClient, AuditedRequestError, EmptyModelContentError,
    _CONTEXT, digest, items_digest,
)


class BudgetRefused(RuntimeError):
    """A request was not dispatched because the shared ledger refused it."""


class BudgetCallError(AuditedRequestError):
    """A sanitized provider failure; usage is retained when available."""


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_INSIDE_AUDIT: ContextVar[bool] = ContextVar("philo_hss_inside_audit", default=False)
_ACTIVE_SLOT: ContextVar[int | None] = ContextVar("philo_hss_active_slot", default=None)
_SECRET_KEY = re.compile(r"api[_-]?key|authorization|api[_-]?base|base[_-]?url|password|secret|extra_headers|default_headers", re.I)


def _safe(value, secrets=()):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        value = vars(value)
    if isinstance(value, dict):
        return {str(key): _safe(part, secrets) for key, part in value.items()
                if not _SECRET_KEY.search(str(key))}
    if isinstance(value, (tuple, list)):
        return [_safe(part, secrets) for part in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(str(secret), "[redacted]")
        return value
    if value is None or type(value) in (int, float, bool):
        return value
    return str(value)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _file_lock(path, timeout):
    """Bounded process and thread lock; Windows locks a single persistent byte."""
    path = Path(path)
    with _LOCKS_GUARD:
        local = _LOCKS.setdefault(str(path.resolve()), threading.Lock())
    started = time.monotonic()
    if not local.acquire(timeout=timeout):
        raise BudgetRefused("Shared budget lock wait timed out; no request sent")
    stream = None
    locked = False
    try:
        stream = path.open("a+b")
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except (OSError, BlockingIOError):
                if time.monotonic() - started >= timeout:
                    raise BudgetRefused("Shared budget lock wait timed out; no request sent") from None
                time.sleep(min(0.05, max(0.001, timeout / 10)))
        # Windows permits locking a byte beyond EOF. Initialize only after
        # acquiring it: competing initializers must not write a locked byte.
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        yield
    finally:
        if stream is not None:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        local.release()


def _usage(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif value is not None and not isinstance(value, dict):
        value = vars(value)
    if not isinstance(value, dict):
        return None
    values = {key: value.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if any(type(part) is not int or part < 0 for part in values.values()):
        return None
    if values["total_tokens"] < values["prompt_tokens"] + values["completion_tokens"]:
        return None
    return values


def _pid_alive(pid):
    """Read process liveness without Windows os.kill(pid, 0), which can kill."""
    if type(pid) is not int or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied: conservatively alive.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_identity(pid):
    """Read process birth and image without signalling or mutating the process."""
    if not _pid_alive(pid):
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
        kernel.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD,
                                                     wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return {"pid": pid, "identity_unavailable": True}
        try:
            birth, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            image = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(image))
            if (not kernel.GetProcessTimes(handle, ctypes.byref(birth), ctypes.byref(exited),
                                           ctypes.byref(system), ctypes.byref(user))
                    or not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size))):
                return {"pid": pid, "identity_unavailable": True}
            ticks = (birth.dwHighDateTime << 32) | birth.dwLowDateTime
            return {"pid": pid, "started_unix": ticks / 10000000 - 11644473600,
                    "executable": image.value}
        finally:
            kernel.CloseHandle(handle)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime "))
        return {"pid": pid, "started_unix": boot + int(stat[19]) / os.sysconf("SC_CLK_TCK"),
                "executable": os.readlink(f"/proc/{pid}/exe")}
    except (OSError, ValueError, StopIteration):
        return {"pid": pid, "identity_unavailable": True}


def process_matches(pid, *, identity=None, started_unix=None, executable=None):
    """Conservatively verify a recorded owner's birth/image, including PID reuse.

    ``started_unix`` alone is the time at which an old record observed its owner;
    a currently running process born after that observation cannot be its owner.
    Unreadable identity remains live rather than authorizing unsafe settlement.
    """
    current = process_identity(pid)
    if current is None:
        return False
    if current.get("identity_unavailable"):
        return True
    if identity and identity.get("started_unix") is not None:
        if abs(current["started_unix"] - identity["started_unix"]) > .01:
            return False
    if started_unix is not None and current["started_unix"] > started_unix + .01:
        return False
    expected_image = (identity or {}).get("executable") or executable
    if expected_image:
        actual_image = current["executable"]
        if not os.path.dirname(expected_image):
            actual_image = os.path.basename(actual_image)
        if os.path.normcase(os.path.normpath(actual_image)) != os.path.normcase(os.path.normpath(expected_image)):
            return False
    return True


def _owner_alive(record):
    return process_matches(record.get("pid"), identity=record.get("process_identity"),
                           started_unix=record.get("started_unix"))


class HSSBudget:
    """One durable whole-experiment budget shared by all jobs and SDK clients."""

    def __init__(self, root, max_calls=60000, max_tokens=350000000, *, max_inflight=1, lock_timeout=3600):
        if any(type(value) is not int or value <= 0 for value in (max_calls, max_tokens, max_inflight)):
            raise ValueError("Budget limits must be positive integers")
        if lock_timeout <= 0:
            raise ValueError("Lock timeout must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "attempts").mkdir(exist_ok=True)
        self.path = self.root / "budget.json"
        self.limits = {"max_calls": max_calls, "max_tokens": max_tokens}
        self.max_inflight = max_inflight
        self.observe_only = False
        self.lock_timeout = lock_timeout
        with self._ledger_lock():
            if self.path.exists():
                self._read()
            else:
                self._save({"version": 2, "limits": self.limits, "max_inflight_requests": max_inflight, "attempt_count": 0,
                            "successful_requests": 0, "failed_requests": 0,
                            "prompt_tokens": 0, "completion_tokens": 0, "actual_tokens": 0,
                            "unknown_usage_attempts": 0, "pending": {}, "blocked_reason": None,
                            "request_count_scope": "hard_cap_before_network",
                            "token_limit_scope": "whole_experiment_actual_usage_soft_stop",
                            "token_tail_limit": f"At most {max_inflight} admitted in-flight responses can cross the token soft stop; unknown usage blocks subsequent requests."})

    def _ledger_lock(self):
        return _file_lock(self.root / "ledger.lock", self.lock_timeout)

    @contextmanager
    def serial_call(self):
        """Acquire one bounded request slot (the historical API name is kept)."""
        with self._ledger_lock():
            self._read()  # Live observation policy/capacity overrides old manifest values.
        started = time.monotonic()
        while time.monotonic() - started < self.lock_timeout:
            for index in range(self.max_inflight):
                name = "network.lock" if self.max_inflight == 1 else f"network.{index}.lock"
                lock = _file_lock(self.root / name, min(.005, self.lock_timeout))
                try:
                    lock.__enter__()
                except BudgetRefused:
                    continue
                token = _ACTIVE_SLOT.set(index)
                try:
                    yield index
                finally:
                    _ACTIVE_SLOT.reset(token)
                    lock.__exit__(None, None, None)
                return
            time.sleep(min(.01, self.lock_timeout / 10))
        raise BudgetRefused("Shared budget request slot wait timed out; no request sent")

    def _read(self):
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            raise BudgetRefused("Shared budget ledger is unreadable") from None
        self.observe_only = (state.get("enforcement_policy") or {}).get("mode") == "observe_only"
        if self.observe_only:
            self.limits = state["limits"]
            self.max_inflight = state.get("max_inflight_requests", 1)
            if type(self.max_inflight) is not int or self.max_inflight <= 0:
                raise BudgetRefused("Shared request concurrency must be a positive integer")
            if state.get("blocked_reason"):
                state["observed_blocked_reason"] = state["blocked_reason"]
            state["blocked_reason"] = None
            return state
        if state.get("limits") != self.limits:
            raise BudgetRefused("Existing budget limits cannot be changed")
        if state.get("max_inflight_requests", 1) != self.max_inflight:
            raise BudgetRefused("Existing budget in-flight capacity cannot be changed")
        return state

    def _save(self, state):
        try:
            state.setdefault("reserved_unknown_tokens", 0)
            state.setdefault("reserved_unknown_attempts", [])
            state["effective_tokens"] = state["actual_tokens"] + state["reserved_unknown_tokens"]
            _atomic_json(self.path, state)
        except Exception:
            raise BudgetRefused("Shared budget ledger persistence failed") from None

    def summary(self):
        with self._ledger_lock():
            state = self._read()
        state["pending_count"] = len(state["pending"])
        state["inflight_count"] = len(state["pending"])
        state["capacity"] = self.max_inflight
        state["orphaned_pending_count"] = sum(not _owner_alive(item) for item in state["pending"].values())
        state["live_inflight_count"] = state["inflight_count"] - state["orphaned_pending_count"]
        if not self.observe_only and state["orphaned_pending_count"] and not state["blocked_reason"]:
            state["blocked_reason"] = "orphaned_pending_usage_unknown"
        state["ledger_path"] = str(self.path)
        # Readers must not turn another process's live pending request into a
        # blocked experiment. Dispatch checks the original owner's liveness.
        state["request_count"] = state["attempt_count"]
        state["input_tokens"] = state["prompt_tokens"]
        state["output_tokens"] = state["completion_tokens"]
        state["total_tokens"] = state["actual_tokens"]
        state.setdefault("reserved_unknown_tokens", 0)
        state.setdefault("reserved_unknown_attempts", [])
        state["effective_tokens"] = state["actual_tokens"] + state["reserved_unknown_tokens"]
        state["usage_complete"] = not (state["pending_count"] or state["unknown_usage_attempts"])
        state["usage_accounted"] = not state["pending_count"] and state["unknown_usage_attempts"] == len(state["reserved_unknown_attempts"])
        return state

    def begin(self, request, **metadata):
        """Persist an attempt before dispatch; callers hold a request slot over I/O."""
        with self._ledger_lock():
            state = self._read()
            if not self.observe_only and state["blocked_reason"]:
                raise BudgetRefused("Shared budget blocked: " + state["blocked_reason"])
            if not self.observe_only and any(not _owner_alive(item) for item in state["pending"].values()):
                state["blocked_reason"] = "orphaned_pending_usage_unknown"
                self._save(state)
                raise BudgetRefused("Shared budget blocked: orphaned_pending_usage_unknown")
            occupied = (sum(_owner_alive(item) for item in state["pending"].values())
                        if self.observe_only else len(state["pending"]))
            if occupied >= self.max_inflight:
                raise BudgetRefused("Shared budget capacity occupied: unfinished_attempt remains live")
            if not self.observe_only and state["attempt_count"] >= self.limits["max_calls"]:
                raise BudgetRefused("Experiment request hard cap reached")
            if not self.observe_only and state["actual_tokens"] + state.get("reserved_unknown_tokens", 0) >= self.limits["max_tokens"]:
                raise BudgetRefused("Experiment token soft-stop reached")
            number = state["attempt_count"] + 1
            record = {**_safe(metadata), "attempt": number, "status": "started",
                      "request": _safe(request), "request_sha256": digest(_safe(request)),
                      "pid": os.getpid(), "slot": _ACTIVE_SLOT.get(), "started_unix": time.time(),
                      "process_identity": process_identity(os.getpid())}
            state["attempt_count"] = number
            state["pending"][str(number)] = {"attempt": number, "pid": os.getpid(),
                                             "slot": record["slot"], "role": record.get("role"), "started_unix": record["started_unix"],
                                             "process_identity": record["process_identity"]}
            # A persistence failure after this point leaves a pending marker.
            self._save(state)
            try:
                _atomic_json(self.root / "attempts" / f"{number:06d}.json", record)
            except Exception:
                raise BudgetRefused("Request audit persistence failed; pending attempt blocks dispatch") from None
            return number

    def finish(self, number, *, usage=None, response=None, status="success", error_type=None,
               usage_reservation=None, **metadata):
        with self._ledger_lock():
            state = self._read()
            if str(number) not in state["pending"]:
                raise BudgetRefused("Attempt is not pending; duplicate settlement refused")
            path = self.root / "attempts" / f"{number:06d}.json"
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                raise BudgetRefused("Request receipt is unreadable; pending attempt retained") from None
            usage = _usage(usage)
            reservation_bound = None
            if usage is None and usage_reservation is not None:
                if (not isinstance(usage_reservation, dict) or error_type not in _CONNECTION_ERRORS
                        or response is not None or not isinstance(usage_reservation.get("reason"), str)
                        or not usage_reservation["reason"].strip()):
                    raise BudgetRefused("Only an explicit bounded connection failure may reserve unknown usage")
                reservation_bound = _text_request_reservation(record["request"])
                if any(usage_reservation.get(key) != value for key, value in reservation_bound.items()):
                    raise BudgetRefused("Unknown-usage reservation differs from the authoritative request bound")
            if usage is None:
                state["unknown_usage_attempts"] += 1
                status = "error"
                error_type = error_type or "MissingOrInvalidUsage"
                if reservation_bound is not None:
                    state.setdefault("reserved_unknown_attempts", []).append(number)
                    state["reserved_unknown_tokens"] = state.get("reserved_unknown_tokens", 0) + reservation_bound["reserved_unknown_tokens"]
                if (state["unknown_usage_attempts"] != len(state.get("reserved_unknown_attempts", []))
                        and not self.observe_only and not state["blocked_reason"]):
                    state["blocked_reason"] = "unknown_usage"
            else:
                for name in ("prompt_tokens", "completion_tokens"):
                    state[name] += usage[name]
                state["actual_tokens"] += usage["total_tokens"]
            state["successful_requests" if status == "success" else "failed_requests"] += 1
            try:
                record.update(_safe(metadata), status=status, usage=usage, response=_safe(response),
                              error_type=error_type, finished_unix=time.time())
                if reservation_bound is not None:
                    _attach_usage_reservation(record, reservation_bound, usage_reservation["reason"])
                _atomic_json(path, record)
            except Exception:
                raise BudgetRefused("Response audit persistence failed; pending attempt retained") from None
            del state["pending"][str(number)]
            self._save(state)
            return deepcopy(record)


def reconcile_orphaned_attempts(budget_root, reason):
    """Explicitly reserve unknown cost of dead owners, preserving unknown usage.

    This is a technical interruption repair, not a usage receipt. The bound
    assumes text-only messages/tool schemas, at most one token per UTF-8 byte,
    requested output as a hard maximum, and 4096 + 64/message protocol overhead.
    Future admission uses confirmed actual tokens plus these reserved bounds.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Orphan reconciliation requires an explicit reason")
    root = Path(budget_root).resolve(strict=True)
    with _file_lock(root / "ledger.lock", 30):
        state = json.loads((root / "budget.json").read_text(encoding="utf-8"))
        prior_reserved = state.get("reserved_unknown_attempts", [])
        if state["unknown_usage_attempts"] != len(prior_reserved):
            raise BudgetRefused("Unreserved unknown usage cannot be cleared by orphan reconciliation")
        prepared = []
        for identifier, owner in state["pending"].items():
            path = root / "attempts" / f"{int(identifier):06d}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if _owner_alive(record):
                raise BudgetRefused("An original request owner remains alive; reconciliation refused")
            if record.get("status") != "started" or record.get("response") is not None or record.get("usage") is not None:
                raise BudgetRefused("Pending receipt has settlement evidence; inspect it instead of reserving")
            request = record.get("request")
            if not isinstance(request, dict):
                raise BudgetRefused("Pending request cannot supply a conservative output bound")
            limit = request.get("max_completion_tokens", request.get("max_tokens"))
            if type(limit) is not int or limit <= 0:
                raise BudgetRefused("Pending request lacks a positive output bound")
            messages = request.get("messages", [])
            def text_only(message):
                if not isinstance(message, dict):
                    return False
                content = message.get("content")
                return (content is None or isinstance(content, str)
                        or isinstance(content, list) and all(
                            isinstance(block, dict) and block.get("type") in {"text", "input_text"}
                            and isinstance(block.get("text"), str) for block in content))
            if not isinstance(messages, list) or not all(text_only(message) for message in messages):
                raise BudgetRefused("Only text-only requests can use byte-based reservations")
            request_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
            bound = request_bytes + limit + 4096 + 64 * len(messages)
            record.update(status="interrupted_usage_reserved", usage=None, response=None,
                          error_type="ProcessExited", finished_unix=time.time(),
                          reserved_unknown_tokens=bound,
                          reconciliation={"reason": reason, "at_unix": time.time(),
                              "owner_identity_verified_exited": True,
                              "request_utf8_bytes": request_bytes, "requested_output_bound": limit,
                              "protocol_overhead_bound": 4096 + 64 * len(messages),
                              "bound_assumptions": "text-only; <=1 token per UTF-8 byte; requested output hard maximum; conservative protocol overhead",
                              "confirmed_actual_usage": False})
            prepared.append((identifier, path, record))
        # Validate every owner and request before writing any reconciliation.
        for identifier, path, record in prepared:
            _atomic_json(path, record)
            state.setdefault("reserved_unknown_attempts", []).append(int(identifier))
            state["reserved_unknown_tokens"] = state.get("reserved_unknown_tokens", 0) + record["reserved_unknown_tokens"]
            state["unknown_usage_attempts"] += 1
            state["failed_requests"] += 1
            del state["pending"][identifier]
        if state.get("blocked_reason") in {"unknown_usage", "orphaned_pending_usage_unknown"}:
            state["blocked_reason"] = None
        state["effective_tokens"] = state["actual_tokens"] + state.get("reserved_unknown_tokens", 0)
        receipt = {"reason": reason, "reconciled_attempts": [int(item[0]) for item in prepared],
                   "reserved_unknown_tokens_added": sum(item[2]["reserved_unknown_tokens"] for item in prepared),
                   "actual_tokens": state["actual_tokens"], "effective_tokens": state["effective_tokens"],
                   "unknown_usage_attempts": state["unknown_usage_attempts"],
                   "blocked_reason": state.get("blocked_reason")}
        if prepared:
            state.setdefault("reconciliations", []).append({**receipt, "at_unix": time.time()})
            _atomic_json(root / "budget.json", state)
        return receipt


_CONNECTION_ERRORS = {"APIConnectionError", "APITimeoutError"}


def _text_request_reservation(request):
    """Return the stated conservative text-only bound, never actual usage."""
    if not isinstance(request, dict):
        raise BudgetRefused("Request cannot supply a conservative output bound")
    limit = request.get("max_completion_tokens")
    if limit is None:
        limit = request.get("max_tokens")
    if type(limit) is not int or limit <= 0:
        raise BudgetRefused("Request lacks a positive output bound")
    messages = request.get("messages")
    def text_only(message):
        if not isinstance(message, dict) or any(message.get(key) for key in ("audio", "images", "attachments")):
            return False
        content = message.get("content")
        return (content is None or isinstance(content, str)
                or isinstance(content, list) and all(
                    isinstance(block, dict) and block.get("type") in {"text", "input_text"}
                    and isinstance(block.get("text"), str) for block in content))
    modalities = request.get("modalities") or []
    if (not isinstance(messages, list) or not all(text_only(message) for message in messages)
            or request.get("audio") or not isinstance(modalities, (list, tuple))
            or any(value != "text" for value in modalities)):
        raise BudgetRefused("Only text-only requests can use byte-based reservations")
    request_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
    overhead = 4096 + 64 * len(messages)
    return {"method": "conservative_text_request_bound", "request_utf8_bytes": request_bytes,
            "requested_output_bound": limit, "protocol_overhead_bound": overhead,
            "reserved_unknown_tokens": request_bytes + limit + overhead,
            "bound_assumptions": "text-only; <=1 token per UTF-8 byte; requested output hard maximum; conservative protocol overhead",
            "confirmed_actual_usage": False}


def _unreserved_content(record):
    return {key: value for key, value in record.items() if key not in {"usage_reservation", "reserved_unknown_tokens"}}


def _attach_usage_reservation(record, bound, reason, original_receipt_sha256=None):
    original = _unreserved_content(record)
    original_bytes = (json.dumps(original, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    metadata = {**bound, "reason": reason, "at_unix": time.time(),
                "original_receipt_sha256": original_receipt_sha256 or digest(original_bytes),
                "original_record_content_sha256": digest(original)}
    record.update(reserved_unknown_tokens=bound["reserved_unknown_tokens"], usage_reservation=metadata)
    return metadata


def reconcile_settled_unknown_attempts(budget_root, reason, *, expected_attempts, expected_hashes):
    """Explicitly reserve pinned, settled connection errors without inventing usage.

    Original error status, usage=None, response=None, failure and unknown counts
    remain intact. Exact original receipt hashes authorize each reservation;
    repeated calls with those same hashes cannot charge the reservation twice.
    Maintenance and unrelated blockers are never released by this operation.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Settled unknown reconciliation requires an explicit reason")
    if (not isinstance(expected_attempts, (list, tuple)) or not expected_attempts
            or any(type(number) is not int or number <= 0 for number in expected_attempts)
            or len(set(expected_attempts)) != len(expected_attempts) or not isinstance(expected_hashes, dict)):
        raise ValueError("Reconciliation requires unique positive attempt IDs and exact hashes")
    hashes = {}
    for key, value in expected_hashes.items():
        match = re.fullmatch(r"([0-9]+)(?:\.json)?", str(key))
        if not match or not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("Invalid pinned attempt hash")
        identifier = int(match.group(1))
        if identifier in hashes:
            raise ValueError("Duplicate pinned attempt hash")
        hashes[identifier] = value
    if set(hashes) != set(expected_attempts):
        raise ValueError("Attempt IDs and pinned receipt hashes must match exactly")
    root = Path(budget_root).resolve(strict=True)
    with _file_lock(root / "ledger.lock", 30):
        state = json.loads((root / "budget.json").read_text(encoding="utf-8"))
        existing = set(state.get("reserved_unknown_attempts", []))
        prepared, already, reservations = [], [], {}
        for number in sorted(expected_attempts):
            path = root / "attempts" / f"{number:06d}.json"
            raw = path.read_bytes()
            record = json.loads(raw.decode("utf-8"))
            if (record.get("attempt") != number or str(number) in state["pending"]
                    or record.get("status") != "error" or record.get("error_type") not in _CONNECTION_ERRORS
                    or record.get("usage") is not None or record.get("response") is not None
                    or not isinstance(record.get("finished_unix"), (int, float))):
                raise BudgetRefused("Only settled, response-free unknown connection failures may be reserved")
            bound = _text_request_reservation(record.get("request"))
            reserved = record.get("reserved_unknown_tokens", 0)
            if reserved:
                metadata = record.get("usage_reservation", {})
                if (reserved != bound["reserved_unknown_tokens"]
                        or metadata.get("original_receipt_sha256") != hashes[number]
                        or metadata.get("method") != bound["method"]
                        or metadata.get("original_record_content_sha256") != digest(_unreserved_content(record))):
                    raise BudgetRefused("Existing reservation does not match the pinned original receipt")
            else:
                if number in existing or digest(raw) != hashes[number]:
                    raise BudgetRefused("Settled unknown receipt differs from its pinned hash")
                metadata = _attach_usage_reservation(record, bound, reason, hashes[number])
            reservations[str(number)] = {"reserved_unknown_tokens": bound["reserved_unknown_tokens"],
                                         "usage_reservation": metadata}
            if number in existing:
                already.append(number)
            else:
                prepared.append((number, path, record))
        if state["unknown_usage_attempts"] < len(existing) + len(prepared):
            raise BudgetRefused("Unknown usage counters cannot support the requested reservations")
        for number, path, record in prepared:
            _atomic_json(path, record)
            state.setdefault("reserved_unknown_attempts", []).append(number)
            state["reserved_unknown_tokens"] = state.get("reserved_unknown_tokens", 0) + record["reserved_unknown_tokens"]
        changed_blocker = (state.get("blocked_reason") == "unknown_usage"
                           and state["unknown_usage_attempts"] == len(state.get("reserved_unknown_attempts", [])))
        if changed_blocker:
            state["blocked_reason"] = None
        state["effective_tokens"] = state["actual_tokens"] + state.get("reserved_unknown_tokens", 0)
        receipt = {"reason": reason, "reconciled_attempts": [item[0] for item in prepared],
                   "already_reserved_attempts": already, "attempt_reservations": reservations,
                   "reserved_unknown_tokens_added": sum(item[2]["reserved_unknown_tokens"] for item in prepared),
                   "actual_tokens": state["actual_tokens"], "effective_tokens": state["effective_tokens"],
                   "unknown_usage_attempts": state["unknown_usage_attempts"],
                   "blocked_reason": state.get("blocked_reason")}
        if prepared:
            state.setdefault("reconciliations", []).append({**receipt, "at_unix": time.time()})
        if prepared or changed_blocker:
            _atomic_json(root / "budget.json", state)
        return receipt


def summarize_job_usage(budget_root, job_id):
    """Read cumulative job usage across resumed or archived worker sessions.

    Global attempt receipts outlive per-worker ``api_calls.jsonl`` files. Keep
    this separate from ``sink.events`` because task execution baselines need
    only the current worker's events. No budget admission or mutation occurs.
    """
    root = Path(budget_root).resolve()
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("Cumulative usage requires a nonempty job ID")
    if not (root / "budget.json").is_file():
        raise FileNotFoundError("Shared budget ledger does not exist")
    with _file_lock(root / "ledger.lock", 30):
        ledger = json.loads((root / "budget.json").read_text(encoding="utf-8"))
        records = [record for path in sorted((root / "attempts").glob("*.json"))
                   if (record := json.loads(path.read_text(encoding="utf-8"))).get("job_id") == job_id]
        pending = set(ledger["pending"])
    result = {"scope": "cumulative_global_budget_attempts_for_job_id", "job_id": job_id,
              "request_count": len(records), "successful_request_count": 0,
              "failed_request_count": 0, "pending_request_count": 0,
              "unknown_usage_attempts": 0, "reserved_unknown_tokens": 0,
              "reserved_unknown_attempts": [], "prompt_tokens": 0, "completion_tokens": 0,
              "total_tokens": 0, "roles": {}, "attempt_ids": []}
    for record in records:
        number = record["attempt"]
        result["attempt_ids"].append(number)
        role = result["roles"].setdefault(record.get("role", "unscoped"), {
            "requests": 0, "successful_requests": 0, "failed_requests": 0,
            "pending_requests": 0, "unknown_usage_attempts": 0, "reserved_unknown_tokens": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "orientation_present_requests": 0})
        role["requests"] += 1
        if str(number) in pending or record.get("status") == "started":
            result["pending_request_count"] += 1
            role["pending_requests"] += 1
            continue
        succeeded = record.get("status") == "success"
        result["successful_request_count" if succeeded else "failed_request_count"] += 1
        role["successful_requests" if succeeded else "failed_requests"] += 1
        role["orientation_present_requests"] += int(record.get("orientation_present_in_request", False))
        usage = _usage(record.get("usage"))
        if usage is None:
            result["unknown_usage_attempts"] += 1
            role["unknown_usage_attempts"] += 1
            if type(record.get("reserved_unknown_tokens")) is int and record["reserved_unknown_tokens"] > 0:
                result["reserved_unknown_tokens"] += record["reserved_unknown_tokens"]
                result["reserved_unknown_attempts"].append(number)
                role["reserved_unknown_tokens"] += record["reserved_unknown_tokens"]
        else:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                result[field] += usage[field]
                role[field] += usage[field]
    result["usage_complete"] = not (result["pending_request_count"] or result["unknown_usage_attempts"])
    result["effective_tokens"] = result["total_tokens"] + result["reserved_unknown_tokens"]
    result["reserved_unknown_attempt_ids"] = list(result["reserved_unknown_attempts"])
    result["usage_accounted"] = not result["pending_request_count"] and result["unknown_usage_attempts"] == len(result["reserved_unknown_attempts"])
    return result


class HSSAuditSink(AuditSink):
    def __init__(self, run_dir, workspace, budget, protocol):
        protocol = dict(protocol)
        model = protocol.get("model", protocol.get("model_id"))
        if not isinstance(model, str) or not model:
            raise ValueError("HSS protocol requires model")
        super().__init__(Path(run_dir) / "api_calls.jsonl", Path(run_dir) / "lineage", model=model,
                         max_tokens=protocol.get("actor_max_output_tokens", 8192),
                         generation_tokens=protocol.get("generation_max_output_tokens", 60000),
                         timeout=protocol.get("request_timeout_seconds", 300), thinking="provider-default")
        self.budget, self.protocol = budget, protocol
        self.connection_recovery = protocol.get("connection_recovery")
        if self.connection_recovery is not None:
            if not isinstance(self.connection_recovery, dict):
                raise ValueError("connection_recovery must be a configuration mapping")
            self.connection_recovery = {"max_additional_attempts": 2, "reserve_unknown_usage": True,
                                        "backoff_seconds": [1, 3], **self.connection_recovery}
            connection = self.connection_recovery
            if (set(connection) != {"max_additional_attempts", "reserve_unknown_usage", "backoff_seconds"}
                    or type(connection["max_additional_attempts"]) is not int
                    or not 0 <= connection["max_additional_attempts"] <= 2
                    or connection["reserve_unknown_usage"] is not True
                    or not isinstance(connection["backoff_seconds"], (list, tuple))
                    or len(connection["backoff_seconds"]) != 2
                    or any(type(delay) not in (int, float) or not 0 <= delay <= 60
                           for delay in connection["backoff_seconds"])):
                raise ValueError("Invalid bounded connection recovery configuration")
        self.transport_recovery = protocol.get("transport_recovery")
        if self.transport_recovery is not None:
            if not isinstance(self.transport_recovery, dict):
                raise ValueError("transport_recovery must be a configuration mapping")
            self.transport_recovery = {"max_additional_attempts": 2, "min_output_tokens": 32768,
                                       "max_output_tokens": 65536, **self.transport_recovery}
            recovery = self.transport_recovery
            if (any(type(recovery[key]) is not int for key in recovery)
                    or set(recovery) != {"max_additional_attempts", "min_output_tokens", "max_output_tokens"}
                    or not 0 <= recovery["max_additional_attempts"] <= 2
                    or not 0 < recovery["min_output_tokens"] <= recovery["max_output_tokens"] <= 65536):
                raise ValueError("Invalid bounded transport recovery configuration")
        self.role_floors = protocol.get("role_min_output_tokens", {})
        if (not isinstance(self.role_floors, dict)
                or any(not isinstance(role, str) or type(limit) is not int or limit <= 0
                       for role, limit in self.role_floors.items())):
            raise ValueError("role_min_output_tokens must map roles to positive integer limits")
        self.job_id = Path(run_dir).name
        self.workspace = Path(workspace).resolve()

    def role_limit(self, role):
        field, default = {
            "generation": ("generation_max_output_tokens", 60000),
            "analysis": ("analysis_max_output_tokens", 8192),
            "validator": ("validation_max_output_tokens", 32768),
            "validation": ("validation_max_output_tokens", 32768),
            "repair": ("validation_max_output_tokens", 32768),
            "judge": ("judge_max_output_tokens", 8192),
        }.get(role, ("actor_max_output_tokens", 8192))
        return self.protocol.get(field, default)

    def orientation(self, context):
        provider = context.get("provider")
        if not provider or not (self.lineage_root / provider).exists():
            return ()
        return super().orientation(context)


def _response_parts(response):
    value = _safe(response)
    choices = value.get("choices", []) if isinstance(value, dict) else []
    content = any(bool(str((choice.get("message") or {}).get("content") or "").strip()) for choice in choices)
    tools = any(bool((choice.get("message") or {}).get("tool_calls")) for choice in choices)
    usage = _usage(value.get("usage")) if isinstance(value, dict) else None
    # Never serialize SDK transport/client metadata attached to a response.
    return {"id": value.get("id"), "choices": choices, "usage": usage}, content, tools, usage


class HSSAuditedClient(AuditedClient):
    """Complete-output SDK audit, preserving native output requests and thinking."""

    def __init__(self, client, sink, context=None):
        super().__init__(client, sink, context)
        self.secrets = tuple(str(getattr(client, name, "") or "") for name in ("api_key", "base_url"))

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, *args):
        return self.client.__exit__(*args)

    def create(self, **kwargs):
        if _INSIDE_AUDIT.get():
            return self.client.chat.completions.create(**kwargs)
        return _audited_dispatch(self.sink, self.client.chat.completions.create, kwargs,
                                 context=self.context, secrets=self.secrets)


def _exception_audit(error, secrets):
    """Retain useful diagnostics without exception messages, URLs or headers."""
    chain, seen, current = [], set(), error
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    identifier = getattr(error, "request_id", None)
    if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", identifier)
            or identifier.lower().startswith(("sk-", "bearer"))
            or any(secret and secret in identifier for secret in secrets)):
        identifier = None
    return {"exception_chain": chain, "http_status": status if type(status) is int and 100 <= status <= 599 else None,
            "request_id": identifier}


def _audited_dispatch(sink, dispatch, kwargs, *, context=None, secrets=(), transport="openai"):
    context = {**(context or {}), **_CONTEXT.get()}
    context.setdefault("job_id", sink.job_id)
    context.setdefault("role", "repair" if transport == "litellm" else "unscoped")
    role = context["role"]
    kwargs = dict(kwargs)
    if kwargs.get("stream"):
        raise BudgetRefused("Streaming is unsupported by the settled-usage audit")
    requested_model = kwargs.get("model")
    model = sink.protocol.get("judge_model", sink.model) if role == "judge" else sink.model
    if transport == "litellm" and str(requested_model).startswith("openai/"):
        model = "openai/" + model.removeprefix("openai/")
    kwargs["model"] = model
    if kwargs.get("max_tokens") is None and kwargs.get("max_completion_tokens") is None:
        kwargs["max_tokens"] = sink.role_limit(role)
    output_field = "max_completion_tokens" if kwargs.get("max_completion_tokens") is not None else "max_tokens"
    original_output_tokens = kwargs[output_field]
    role_floor = sink.role_floors.get(role)
    if role_floor is not None:
        kwargs[output_field] = max(kwargs[output_field], role_floor)
    kwargs["timeout"] = sink.timeout
    kwargs.pop("max_retries", None)
    if transport == "litellm":
        kwargs["num_retries"] = 0
    # provider-default deliberately leaves extra_body.thinking untouched.
    for key in ("api_key", "api_base", "base_url"):
        if kwargs.get(key):
            secrets = (*secrets, str(kwargs[key]))
    items = sink.orientation(context)
    prompt = json.dumps(kwargs.get("messages", []), ensure_ascii=False)
    recovery = sink.transport_recovery
    with sink.budget._ledger_lock():
        sink.budget._read()
    observe_only = sink.budget.observe_only
    connection = sink.connection_recovery
    if observe_only and connection is None:
        connection = {"max_additional_attempts": 2, "backoff_seconds": [1, 3]}
    chain_id = uuid4().hex if recovery is not None or connection is not None else None
    failed_attempts = []
    previous_error = None
    recovery_index = 0
    while True:
        request = _safe(kwargs, secrets)
        event = {**_safe(context, secrets), "model": model, "requested_model": requested_model,
                 "transport": transport, "request": request,
                 "prompt_sha256": digest(request.get("messages", [])), "request_sha256": digest(request),
                 "orientation_items_sha256": items_digest(items), "orientation_item_count": len(items),
                 "orientation_present_in_request": bool(items) and all(item in prompt for item in items),
                 "max_tokens": kwargs[output_field], "original_requested_output_tokens": original_output_tokens,
                 "role_min_output_tokens": role_floor,
                 "thinking": "provider-default", "timeout_seconds": sink.timeout,
                 "sdk_max_retries": 0, "status_scope": "sdk_request_and_output"}
        if chain_id is not None:
            event.update(recovery_chain_id=chain_id, recovery_index=recovery_index,
                         retry_of_attempt=failed_attempts[-1] if failed_attempts else None,
                         recovery_reason=previous_error, transport_recovered=False)
        # Each recovery acquires a fresh slot and admits/settles its own request.
        # Observation mode preserves unknown cost without blocking recovery.
        with sink.budget.serial_call():
            number = sink.budget.begin(request, **{key: value for key, value in event.items() if key != "request"})
            event["attempt"] = number
            started = time.monotonic()
            response_record = usage = None
            error_type = None
            result = None
            interrupted = None
            token = _INSIDE_AUDIT.set(True)
            try:
                result = dispatch(**kwargs)
                response_record, content, tools, usage = _response_parts(result)
                response_record = _safe(response_record, secrets)
                finish_reasons = [choice.get("finish_reason") for choice in response_record["choices"]]
                event.update(response_has_content=content, response_has_tool_calls=tools,
                             response_finish_reasons=finish_reasons)
                if usage is None:
                    error_type = "MissingOrInvalidUsage"
                    if observe_only and (content or tools) and "length" not in finish_reasons:
                        event["response_accepted_without_usage"] = True
                elif not content and not tools:
                    error_type = "EmptyModelContentError"
                elif recovery is not None and "length" in finish_reasons:
                    error_type = "TruncatedModelResponse"
            except BaseException as exc:
                error_type = type(exc).__name__
                event.update(_exception_audit(exc, secrets))
                usage = _usage(getattr(exc, "usage", None))
                if usage is None and isinstance(getattr(exc, "body", None), dict):
                    usage = _usage(exc.body.get("usage"))
                if not isinstance(exc, Exception):
                    interrupted = exc
            finally:
                _INSIDE_AUDIT.reset(token)
                event.update(status="error" if error_type else "success", usage=usage,
                             response=response_record, error_type=error_type,
                             duration_seconds=round(time.monotonic() - started, 3))
                if (not error_type or event.get("response_accepted_without_usage")) and failed_attempts:
                    event.update(transport_recovered=True, recovered_attempts=list(failed_attempts))
                reservation = None
                if not observe_only and connection is not None and usage is None and response_record is None and error_type in _CONNECTION_ERRORS:
                    try:
                        reservation = {**_text_request_reservation(request),
                            "reason": "Opt-in bounded connection recovery: preserve unknown usage under a conservative text-request reservation"}
                    except BudgetRefused:
                        event["usage_reservation_refused"] = True
                settled = sink.budget.finish(number, usage=usage, response=response_record,
                                   status=event["status"], error_type=error_type,
                                   usage_reservation=reservation,
                                   **{key: value for key, value in event.items()
                                      if key in {"duration_seconds", "response_finish_reasons", "transport_recovered", "recovered_attempts",
                                                 "exception_chain", "http_status", "request_id", "response_accepted_without_usage"}})
                if settled.get("reserved_unknown_tokens", 0) > 0:
                    event.update(reserved_unknown_tokens=settled["reserved_unknown_tokens"],
                                 usage_reservation=settled["usage_reservation"])
                sink.append(event)
        if interrupted is not None:
            raise interrupted
        if not error_type or event.get("response_accepted_without_usage"):
            return result
        transient = (error_type in _CONNECTION_ERRORS
                     or observe_only and event.get("http_status") in {408, 429, 500, 502, 503, 504})
        if (connection is not None and transient
                and (observe_only or usage is not None or event.get("reserved_unknown_tokens", 0) > 0)
                and recovery_index < connection["max_additional_attempts"]):
            failed_attempts.append(number)
            previous_error = error_type
            time.sleep(connection["backoff_seconds"][recovery_index])
            recovery_index += 1
            continue
        if (recovery is not None and usage is not None
                and error_type in {"EmptyModelContentError", "TruncatedModelResponse"}
                and recovery_index < recovery["max_additional_attempts"]):
            expanded = min(recovery["max_output_tokens"],
                           max(recovery["min_output_tokens"], kwargs[output_field] * 2))
            if expanded > kwargs[output_field]:
                failed_attempts.append(number)
                previous_error = error_type
                recovery_index += 1
                kwargs[output_field] = expanded
                continue
        if error_type == "EmptyModelContentError":
            raise EmptyModelContentError("Model response contains no assistant text or tool calls")
        raise BudgetCallError(error_type) from None


def _disable_native_query_retries(stack, workspace):
    """Let the audit own retries instead of mini-swe-agent's outer tenacity loop.

    The native validator imports this model lazily. Import its local module now
    so that future validator initialization receives the same patched class.
    The undecorated query still routes through audited LiteLLM, and ExitStack
    restores the exact original decorator when this worker's runtime closes.
    """
    name = "minisweagent.models.litellm_model"
    module = sys.modules.get(name)
    if module is None:
        source = Path(workspace) / "mini-swe-agent" / "src"
        if not (source / "minisweagent/models/litellm_model.py").is_file():
            return False
        with patch.object(sys, "path", [str(source), *sys.path]), \
                patch("dotenv.load_dotenv", return_value=False):
            module = importlib.import_module(name)
    original = module.LitellmModel._query
    query = inspect.unwrap(original)
    if query is not original:
        stack.enter_context(patch.object(module.LitellmModel, "_query", query))
    return True


def install_hss_audit(stack, run_dir, workspace, budget_root, protocol):
    """Wrap SDK construction (including imported aliases) and repair LiteLLM.

    Native execution supplies role/job/round scopes through audit_scope. Factory
    patches cover imports made after installation; existing native bindings are
    replaced by identity, including aliases of OpenAI/Client imported by name.
    """
    import openai

    budget = HSSBudget(budget_root, max_calls=protocol.get("max_calls", 60000),
                       max_tokens=protocol.get("max_tokens", 350000000),
                       max_inflight=protocol.get("max_inflight_requests", 1))
    sink = HSSAuditSink(run_dir, workspace, budget, protocol)
    originals = tuple({openai.OpenAI, openai.Client})

    def factory_for(original):
        def factory(*args, **kwargs):
            kwargs.update(timeout=sink.timeout, max_retries=0)
            return HSSAuditedClient(original(*args, **kwargs), sink)
        return factory

    factories = {original: factory_for(original) for original in originals}
    # Snapshot before patching: mutations must not change what other aliases match.
    bindings = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if module is None or name.startswith(("openai.", "philoagent.hss_budget")):
            continue
        for attr, value in list(vars(module).items()):
            for original, factory in factories.items():
                if value is original:
                    bindings.append((module, attr, factory))
    for module, attr, factory in bindings:
        stack.enter_context(patch.object(module, attr, factory))
    try:
        # LiteLLM's import-time load_dotenv must not replace the manager's
        # already frozen environment or read credentials from a second file.
        with patch("dotenv.load_dotenv", return_value=False):
            litellm = importlib.import_module("litellm")
    except ImportError:
        litellm = None
    if litellm is not None:
        original_completion = litellm.completion

        def completion(*args, **kwargs):
            if args:
                if len(args) > 2:
                    raise BudgetRefused("Repair completion requires named request arguments")
                for key, value in zip(("model", "messages"), args):
                    kwargs[key] = value
            if _INSIDE_AUDIT.get():
                return original_completion(**kwargs)
            return _audited_dispatch(sink, original_completion, kwargs, transport="litellm")

        for module in list(sys.modules.values()):
            if module is None or getattr(module, "__name__", "").startswith("philoagent.hss_budget"):
                continue
            for attr, value in list(vars(module).items()):
                if value is original_completion:
                    stack.enter_context(patch.object(module, attr, completion))
        sink.native_transport_retries_disabled = _disable_native_query_retries(stack, workspace)
    return sink
