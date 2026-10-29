"""Offline, shared accounting tests: never load .env or contact a provider."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import importlib
import multiprocessing
import os
from pathlib import Path
import sys
import threading
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import time
import unittest
from unittest.mock import patch

from philoagent.ablation_runtime import EmptyModelContentError, audit_scope
from philoagent.hss_budget import (
    BudgetCallError, BudgetRefused, HSSAuditSink, HSSAuditedClient, HSSBudget,
    install_hss_audit, summarize_job_usage, process_identity, process_matches,
    reconcile_orphaned_attempts, reconcile_settled_unknown_attempts,
)


def response(*, content="done", tools=None, usage=True, tokens=15, finish_reason="stop"):
    return SimpleNamespace(id="offline-response", choices=[SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content, tool_calls=tools), finish_reason=finish_reason)],
        usage={"prompt_tokens": tokens - 3, "completion_tokens": 3, "total_tokens": tokens} if usage else None)


class FakeClient:
    created = []

    def __init__(self, callback=None, **kwargs):
        self.callback = callback or (lambda **kw: response())
        self.options = kwargs
        self.api_key = kwargs.get("api_key", "secret-key")
        self.base_url = kwargs.get("base_url", "https://secret-provider.example/v1")
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        type(self).created.append(self)

    def with_options(self, **kwargs):
        self.options.update(kwargs)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.callback(**kwargs)


def _client(root, callback=None, *, protocol=None, **limits):
    budget = HSSBudget(Path(root) / "shared", **limits)
    sink = HSSAuditSink(Path(root) / "job", root, budget, {"model": "deepseek-offline", **(protocol or {})})
    raw = FakeClient(callback)
    return HSSAuditedClient(raw, sink), raw, sink


def _call(client, **kwargs):
    return client.chat.completions.create(model="requested", messages=[{"role": "user", "content": "offline task"}], **kwargs)


def _connection_failure():
    import httpx
    from openai import APIConnectionError
    request = httpx.Request("POST", "https://secret-provider.example/v1")
    error = APIConnectionError(request=request)
    error.__cause__ = httpx.ConnectError("secret-key https://secret-provider.example/v1", request=request)
    return error


def _process_worker(root, count, queue):
    """A top-level target works with real Windows spawn, not only Unix fork."""
    try:
        budget = HSSBudget(root, max_calls=100, max_tokens=100000)
        for _ in range(count):
            with budget.serial_call():
                number = budget.begin({"messages": ["offline"]}, role="actor")
                time.sleep(0.015)
                budget.finish(number, usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
        queue.put("done")
    except Exception as exc:
        queue.put(type(exc).__name__)


def _parallel_process_worker(root, ready, release, *, orphan=False):
    try:
        budget = HSSBudget(root, max_calls=3, max_tokens=100000, max_inflight=2)
        with budget.serial_call():
            number = budget.begin({"messages": ["parallel offline"]}, role="actor")
            ready.put(number)
            if orphan:
                return
            if not release.wait(timeout=15):
                raise TimeoutError("parent did not release offline requests")
            budget.finish(number, usage={"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20})
    except Exception as exc:
        ready.put(type(exc).__name__)


class HSSBudgetTests(unittest.TestCase):
    def test_observe_only_uses_live_capacity_and_keeps_unknown_and_orphan_receipts(self):
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary, max_calls=1, max_tokens=1)
            number = budget.begin({"messages": [], "max_tokens": 1})
            budget.finish(number, status="error", error_type="InternalServerError")
            state = json.loads(budget.path.read_text(encoding="utf-8"))
            state["enforcement_policy"] = {"mode": "observe_only", "reason": "User authorized unrestricted experiment"}
            state["max_inflight_requests"] = 16
            state["pending"]["historic-orphan"] = {"pid": -1, "attempt": 0}
            budget.path.write_text(json.dumps(state), encoding="utf-8")
            reopened = HSSBudget(temporary, max_calls=60000, max_tokens=350000000, max_inflight=6)
            with reopened.serial_call():
                new_number = reopened.begin({"messages": [], "max_tokens": 1})
                reopened.finish(new_number, usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})
            summary = reopened.summary()
            self.assertEqual(summary["capacity"], 16)
            self.assertEqual(summary["unknown_usage_attempts"], 1)
            self.assertIsNone(summary["blocked_reason"])
            self.assertEqual(summary["actual_tokens"], 15)
            self.assertEqual(summary["orphaned_pending_count"], 1)
            self.assertEqual(summary["observed_blocked_reason"], "unknown_usage")
            original = json.loads((Path(temporary) / "attempts/000001.json").read_text(encoding="utf-8"))
            self.assertIsNone(original["usage"])
            self.assertEqual(original["error_type"], "InternalServerError")

    def test_observe_only_retries_server_errors_for_both_transports_without_global_stop(self):
        from philoagent.hss_budget import _audited_dispatch
        for transport in ("openai", "litellm"):
            with self.subTest(transport=transport), TemporaryDirectory() as temporary:
                budget = HSSBudget(temporary, max_calls=1, max_tokens=1)
                state = json.loads(budget.path.read_text(encoding="utf-8"))
                state["enforcement_policy"] = {"mode": "observe_only"}
                budget.path.write_text(json.dumps(state), encoding="utf-8")
                sink = HSSAuditSink(Path(temporary) / "job", temporary, HSSBudget(temporary), {"model": "offline"})
                calls = []
                def dispatch(**kwargs):
                    calls.append(kwargs)
                    if len(calls) < 3:
                        error = RuntimeError("provider failed")
                        error.status_code = 500 if len(calls) == 1 else 503
                        raise error
                    return response()
                with patch("philoagent.hss_budget.time.sleep"):
                    result = _audited_dispatch(sink, dispatch, {"messages": []}, transport=transport)
                self.assertEqual(len(calls), 3)
                self.assertIsNotNone(result)
                self.assertEqual(sink.budget.summary()["unknown_usage_attempts"], 2)
                self.assertEqual(sink.budget.summary()["actual_tokens"], 15)
                self.assertIsNone(sink.budget.summary()["blocked_reason"])
                self.assertEqual(sink.events[-1]["recovered_attempts"], [1, 2])

    def test_observe_only_keeps_usable_response_without_usage(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, lambda **kwargs: response(usage=False))
            state = json.loads(sink.budget.path.read_text(encoding="utf-8"))
            state["enforcement_policy"] = {"mode": "observe_only"}
            sink.budget.path.write_text(json.dumps(state), encoding="utf-8")
            result = _call(client)
            self.assertEqual(result.choices[0].message.content, "done")
            self.assertEqual(sink.budget.summary()["unknown_usage_attempts"], 1)
            self.assertIsNone(sink.events[0]["usage"])
            self.assertEqual(sink.events[0]["error_type"], "MissingOrInvalidUsage")
            self.assertTrue(sink.events[0]["response_accepted_without_usage"])

    def test_settled_unknown_reservation_preserves_error_usage_and_maintenance_idempotently(self):
        from philoagent.ablation_runtime import digest
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary)
            n = budget.begin({"messages": [{"role": "user", "content": "text only"}], "max_tokens": 32768},
                             role="actor", job_id="job-1")
            budget.finish(n, status="error", error_type="APIConnectionError")
            path = Path(temporary) / "attempts/000001.json"
            original_hash = digest(path.read_bytes())
            with budget._ledger_lock():
                state = budget._read()
                state["blocked_reason"] = "maintenance_connection_repair"
                budget._save(state)
            receipt = reconcile_settled_unknown_attempts(temporary, "User authorized bounded unknown-cost reservation",
                        expected_attempts=[n], expected_hashes={n: original_hash})
            saved = json.loads(path.read_text(encoding="utf8"))
            summary = budget.summary()
            self.assertEqual(saved["status"], "error")
            self.assertEqual(saved["error_type"], "APIConnectionError")
            self.assertIsNone(saved["usage"])
            self.assertIsNone(saved["response"])
            self.assertGreater(saved["reserved_unknown_tokens"], 32768)
            self.assertEqual(saved["usage_reservation"]["original_receipt_sha256"], original_hash)
            self.assertEqual(summary["actual_tokens"], 0)
            self.assertEqual(summary["unknown_usage_attempts"], 1)
            self.assertEqual(summary["failed_requests"], 1)
            self.assertEqual(summary["blocked_reason"], "maintenance_connection_repair")
            cumulative = summarize_job_usage(temporary, "job-1")
            self.assertEqual(cumulative["reserved_unknown_attempt_ids"], [n])
            self.assertTrue(cumulative["usage_accounted"])
            self.assertFalse(cumulative["usage_complete"])
            again = reconcile_settled_unknown_attempts(temporary, "Same explicit repair",
                        expected_attempts=[n], expected_hashes={"000001.json": original_hash})
            self.assertEqual(again["reconciled_attempts"], [])
            self.assertEqual(again["already_reserved_attempts"], [n])
            self.assertEqual(budget.summary()["reserved_unknown_tokens"], receipt["reserved_unknown_tokens_added"])

    def test_settled_unknown_reservation_refuses_bad_hash_known_usage_or_multimodal(self):
        from philoagent.ablation_runtime import digest
        for mode in ("bad_hash", "known_usage", "multimodal", "nonconnection"):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                budget = HSSBudget(temporary)
                messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "private"}}]}] if mode == "multimodal" else []
                n = budget.begin({"messages": messages, "max_tokens": 32768})
                budget.finish(n, status="error", error_type="ConnectionError" if mode == "nonconnection" else "APIConnectionError",
                              usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15} if mode == "known_usage" else None)
                path = Path(temporary) / "attempts/000001.json"
                before, ledger_before = path.read_bytes(), (Path(temporary) / "budget.json").read_bytes()
                expected = "0" * 64 if mode == "bad_hash" else digest(before)
                with self.assertRaises((BudgetRefused, ValueError)):
                    reconcile_settled_unknown_attempts(temporary, "Explicit pinned reservation only",
                        expected_attempts=[n], expected_hashes={n: expected})
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual((Path(temporary) / "budget.json").read_bytes(), ledger_before)

    def test_two_settled_connection_reservations_share_exact_counter_and_release_only_when_accounted(self):
        from philoagent.ablation_runtime import digest
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary, max_inflight=2)
            numbers = [budget.begin({"messages": [], "max_tokens": 32768}, job_id="parallel") for _ in range(2)]
            for n in numbers:
                budget.finish(n, status="error", error_type="APIConnectionError")
            hashes = {n: digest((Path(temporary) / "attempts" / f"{n:06d}.json").read_bytes()) for n in numbers}
            def reserve(n):
                return reconcile_settled_unknown_attempts(temporary, "Independent explicit bounded connection recovery",
                        expected_attempts=[n], expected_hashes={n: hashes[n]})
            with ThreadPoolExecutor(max_workers=2) as executor:
                receipts = list(executor.map(reserve, numbers))
            summary = budget.summary()
            self.assertEqual(summary["unknown_usage_attempts"], 2)
            self.assertEqual(summary["failed_requests"], 2)
            self.assertEqual(summary["reserved_unknown_tokens"], sum(item["reserved_unknown_tokens_added"] for item in receipts))
            self.assertEqual(set(summary["reserved_unknown_attempts"]), set(numbers))
            self.assertIsNone(summary["blocked_reason"])
            self.assertIn("unknown_usage", [item["blocked_reason"] for item in receipts])
            with ThreadPoolExecutor(max_workers=2) as executor:
                again = list(executor.map(reserve, numbers))
            self.assertEqual(sum(item["reserved_unknown_tokens_added"] for item in again), 0)
            self.assertEqual(budget.summary()["actual_tokens"], 0)

    def test_reserved_connection_receipt_cannot_be_changed_after_original_hash_authorization(self):
        from philoagent.ablation_runtime import digest
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary)
            n = budget.begin({"messages": [{"role": "user", "content": "one"}], "max_tokens": 32768})
            budget.finish(n, status="error", error_type="APIConnectionError")
            path = Path(temporary) / "attempts/000001.json"
            original_hash = digest(path.read_bytes())
            args = {"expected_attempts": [n], "expected_hashes": {n: original_hash}}
            reconcile_settled_unknown_attempts(temporary, "Explicit original receipt", **args)
            changed = json.loads(path.read_text(encoding="utf8"))
            changed["request"]["messages"][0]["content"] = "two"  # Same byte bound, different paid request.
            path.write_text(json.dumps(changed), encoding="utf8")
            with self.assertRaisesRegex(BudgetRefused, "reservation"):
                reconcile_settled_unknown_attempts(temporary, "Must not authorize a changed receipt", **args)

    def test_connection_failure_is_explicitly_reserved_retried_and_linked(self):
        with TemporaryDirectory() as temporary:
            calls = []
            def dispatch(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise _connection_failure()
                return response()
            client, raw, sink = _client(temporary, dispatch, protocol={"connection_recovery": {}})
            with patch("philoagent.hss_budget.time.sleep") as sleeping, audit_scope(task_execution_id="connection-task"):
                _call(client, max_tokens=32768)
            self.assertEqual(len(raw.calls), 2)
            sleeping.assert_called_once_with(1)
            error, success = sink.events
            self.assertEqual(error["status"], "error")
            self.assertIsNone(error["usage"])
            self.assertGreater(error["reserved_unknown_tokens"], 32768)
            self.assertEqual(error["exception_chain"], ["APIConnectionError", "ConnectError"])
            self.assertEqual(success["recovered_attempts"], [error["attempt"]])
            self.assertEqual(success["retry_of_attempt"], error["attempt"])
            self.assertTrue(success["transport_recovered"])
            summary = sink.budget.summary()
            self.assertEqual(summary["attempt_count"], 2)
            self.assertEqual(summary["failed_requests"], 1)
            self.assertEqual(summary["unknown_usage_attempts"], 1)
            self.assertEqual(summary["actual_tokens"], 15)
            self.assertIsNone(summary["blocked_reason"])
            self.assertEqual(summary["effective_tokens"], summary["reserved_unknown_tokens"] + 15)
            audits = "\n".join(p.read_text(encoding="utf8") for p in Path(temporary).rglob("*.json*"))
            self.assertNotIn("secret-key", audits)
            self.assertNotIn("secret-provider", audits)
            from philoagent.hss_tasks_runtime import NativeHSSRunner
            callback = object.__new__(NativeHSSRunner)
            callback.usage_reader = lambda: sink.events
            task_usage = callback._usage("connection-task")
            self.assertFalse(task_usage["usage_complete"])
            self.assertTrue(task_usage["usage_accounted"])
            self.assertEqual(task_usage["unknown_usage_attempts"], 1)
            self.assertEqual(task_usage["reserved_unknown_attempt_ids"], [error["attempt"]])
            self.assertEqual(task_usage["effective_tokens"], summary["effective_tokens"])
            self.assertEqual(task_usage["unrecovered_failed_request_count"], 0)

    def test_connection_error_with_known_usage_is_retried_without_a_second_reservation(self):
        with TemporaryDirectory() as temporary:
            count = []
            def dispatch(**kwargs):
                count.append(kwargs)
                if len(count) == 1:
                    error = _connection_failure()
                    error.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                    raise error
                return response()
            client, raw, sink = _client(temporary, dispatch, protocol={"connection_recovery": {}})
            with patch("philoagent.hss_budget.time.sleep"):
                _call(client)
            summary = sink.budget.summary()
            self.assertEqual(len(raw.calls), 2)
            self.assertEqual(summary["actual_tokens"], 30)
            self.assertEqual(summary["unknown_usage_attempts"], 0)
            self.assertEqual(summary["reserved_unknown_tokens"], 0)
            self.assertEqual(sink.events[1]["recovered_attempts"], [1])

    def test_connection_recovery_is_bounded_and_leaves_unknown_reserved_after_exhaustion(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, lambda **kw: (_ for _ in ()).throw(_connection_failure()),
                                       protocol={"connection_recovery": {}})
            with patch("philoagent.hss_budget.time.sleep") as sleeping:
                with self.assertRaisesRegex(BudgetCallError, "^APIConnectionError$"):
                    _call(client)
            self.assertEqual(len(raw.calls), 3)
            self.assertEqual([call.args[0] for call in sleeping.call_args_list], [1, 3])
            summary = sink.budget.summary()
            self.assertEqual(summary["unknown_usage_attempts"], 3)
            self.assertEqual(summary["failed_requests"], 3)
            self.assertEqual(summary["actual_tokens"], 0)
            self.assertEqual(summary["reserved_unknown_attempts"], [1, 2, 3])
            self.assertIsNone(summary["blocked_reason"])
            self.assertTrue(all(not event.get("transport_recovered") for event in sink.events))

    def test_connection_recovery_respects_request_cap_and_effective_token_limit(self):
        for limits, expected in (({"max_calls": 1}, "hard cap"), ({"max_tokens": 10}, "soft-stop")):
            with self.subTest(limits=limits), TemporaryDirectory() as temporary:
                client, raw, sink = _client(temporary, lambda **kw: (_ for _ in ()).throw(_connection_failure()),
                                           protocol={"connection_recovery": {}}, **limits)
                with patch("philoagent.hss_budget.time.sleep"), self.assertRaisesRegex(BudgetRefused, expected):
                    _call(client)
                self.assertEqual(len(raw.calls), 1)
                summary = sink.budget.summary()
                self.assertEqual(summary["attempt_count"], 1)
                self.assertEqual(summary["actual_tokens"], 0)
                self.assertGreater(summary["effective_tokens"], 10)
                self.assertEqual(summary["pending_count"], 0)

    def test_atomic_connection_reservation_does_not_temporarily_block_an_independent_client(self):
        with TemporaryDirectory() as temporary:
            count = []
            def first_dispatch(**kwargs):
                count.append(kwargs)
                if len(count) == 1:
                    raise _connection_failure()
                return response()
            first, raw, sink = _client(temporary, first_dispatch, max_inflight=2,
                protocol={"connection_recovery": {"backoff_seconds": [0, 0]}})
            other, other_raw, _ = _client(temporary, max_inflight=2)
            settled, release = threading.Event(), threading.Event()
            original_finish = sink.budget.finish
            def gated_finish(*args, **kwargs):
                result = original_finish(*args, **kwargs)
                if kwargs.get("error_type") == "APIConnectionError":
                    settled.set()
                    if not release.wait(5):
                        raise AssertionError("Parent did not release offline error settlement")
                return result
            with patch.object(sink.budget, "finish", side_effect=gated_finish), ThreadPoolExecutor(max_workers=1) as executor:
                running = executor.submit(_call, first)
                try:
                    self.assertTrue(settled.wait(5))
                    _call(other)  # Must succeed even before the first dispatch regains control.
                finally:
                    release.set()
                running.result(timeout=5)
            summary = sink.budget.summary()
            self.assertEqual(summary["attempt_count"], 3)
            self.assertEqual(summary["actual_tokens"], 30)
            self.assertEqual(summary["failed_requests"], 1)
            self.assertEqual(summary["unknown_usage_attempts"], 1)
            self.assertEqual(summary["pending_count"], 0)
            self.assertIsNone(summary["blocked_reason"])

    def test_connection_recovery_still_hard_blocks_nontext_or_missing_usage_output(self):
        for mode in ("image", "missing_usage"):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                callback = ((lambda **kw: (_ for _ in ()).throw(_connection_failure())) if mode == "image"
                            else (lambda **kw: response(usage=False)))
                client, raw, sink = _client(temporary, callback, protocol={"connection_recovery": {}})
                with self.assertRaises(BudgetCallError):
                    if mode == "image":
                        client.chat.completions.create(model="offline", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "private-image"}}]}])
                    else:
                        _call(client)
                self.assertEqual(len(raw.calls), 1)
                self.assertEqual(sink.budget.summary()["blocked_reason"], "unknown_usage")
                self.assertEqual(sink.budget.summary()["reserved_unknown_tokens"], 0)

    def test_exception_audit_keeps_only_types_status_and_safe_request_id(self):
        import httpx
        from openai import APIStatusError
        with TemporaryDirectory() as temporary:
            request = httpx.Request("POST", "https://secret-provider.example/v1")
            error = APIStatusError("secret-key private provider message", response=httpx.Response(503, request=request,
                    headers={"x-request-id": "trace_123", "Authorization": "secret-key"}), body=None)
            error.__cause__ = RuntimeError("secret-key https://secret-provider.example/v1")
            client, raw, sink = _client(temporary, lambda **kw: (_ for _ in ()).throw(error))
            with self.assertRaises(BudgetCallError):
                _call(client)
            self.assertEqual(sink.events[0]["exception_chain"], ["APIStatusError", "RuntimeError"])
            self.assertEqual(sink.events[0]["http_status"], 503)
            self.assertEqual(sink.events[0]["request_id"], "trace_123")
            saved = json.loads((Path(temporary) / "shared/attempts/000001.json").read_text(encoding="utf8"))
            self.assertEqual(saved["exception_chain"], sink.events[0]["exception_chain"])
            audit = (Path(temporary) / "job/api_calls.jsonl").read_text(encoding="utf8")
            self.assertNotIn("secret-key", audit)
            self.assertNotIn("secret-provider", audit)
            self.assertNotIn("private provider message", audit)

    def test_native_swe_transport_does_not_retry_budget_refusal_or_unknown_usage(self):
        from tenacity import stop_after_attempt
        mini = Path(__file__).resolve().parents[1] / "vendor/MemEvolve/Flash-Searcher-main/mini-swe-agent/src"
        with TemporaryDirectory() as temporary, patch.object(sys, "path", [str(mini), *sys.path]), \
                patch.dict(os.environ, {"MSWEA_SILENT_STARTUP": "1", "MSWEA_GLOBAL_CONFIG_DIR": temporary}), \
                patch("dotenv.load_dotenv", return_value=False):
            module = importlib.import_module("minisweagent.models.litellm_model")
        original = module.LitellmModel._query
        self.assertTrue(hasattr(original, "__wrapped__"))
        for mode in ("budget_refused", "usage_unknown"):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary, ExitStack() as stack:
                sleeping, callbacks = [], []
                stack.enter_context(patch.object(original.retry, "stop", stop_after_attempt(2)))
                stack.enter_context(patch.object(original.retry, "sleep", side_effect=lambda seconds: sleeping.append(seconds)))
                def offline_completion(**kwargs):
                    callbacks.append(kwargs)
                    return response(usage=False)
                stack.enter_context(patch.object(module.litellm, "completion", offline_completion))
                sink = install_hss_audit(stack, Path(temporary) / "job", temporary,
                                         Path(temporary) / "budget", {"model": "offline"})
                if mode == "budget_refused":
                    with sink.budget._ledger_lock():
                        state = sink.budget._read()
                        state["blocked_reason"] = "maintenance_test"
                        sink.budget._save(state)
                client = module.LitellmModel(model_name="offline", cost_tracking="ignore_errors")
                exception = BudgetRefused if mode == "budget_refused" else BudgetCallError
                with patch.object(sink.budget, "begin", wraps=sink.budget.begin) as begin:
                    with self.assertRaises(exception):
                        client._query([{"role": "user", "content": "offline"}])
                    self.assertEqual(begin.call_count, 1)
                self.assertEqual(sleeping, [])
                self.assertEqual(len(callbacks), int(mode == "usage_unknown"))
                self.assertEqual(sink.budget.summary()["attempt_count"], int(mode == "usage_unknown"))
            self.assertIs(module.LitellmModel._query, original)

    def test_process_matches_rejects_reused_pid_birth_and_executable(self):
        identity = process_identity(__import__("os").getpid())
        self.assertTrue(identity["executable"])
        self.assertGreater(identity["started_unix"], 0)
        self.assertTrue(process_matches(identity["pid"], identity=identity))
        current = {"pid": 1234, "started_unix": 2000., "executable": "C:/tools/node.exe"}
        with patch("philoagent.hss_budget.process_identity", return_value=current):
            self.assertFalse(process_matches(1234, started_unix=1000.))
            self.assertFalse(process_matches(1234, identity={**current, "executable": "C:/Python/python.exe"}))
            self.assertTrue(process_matches(1234, identity=current))

    def test_orphan_reconciliation_reserves_unknown_cost_without_fabricating_usage(self):
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary, max_tokens=100000)
            request = {"model": "offline", "messages": [{"role": "user", "content": "中文任务"}], "max_tokens": 32768}
            number = budget.begin(request, role="actor", job_id="job-1")
            original = json.loads((Path(temporary) / "attempts/000001.json").read_text(encoding="utf-8"))
            with patch("philoagent.hss_budget.process_matches", return_value=False):
                receipt = reconcile_orphaned_attempts(temporary, "Explicit user-authorized process-loss repair")
            summary = budget.summary()
            bound = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + 32768 + 4096 + 64
            self.assertEqual(receipt["reconciled_attempts"], [number])
            self.assertEqual(summary["actual_tokens"], 0)
            self.assertEqual(summary["unknown_usage_attempts"], 1)
            self.assertEqual(summary["reserved_unknown_tokens"], bound)
            self.assertEqual(summary["effective_tokens"], bound)
            self.assertEqual(summary["reserved_unknown_attempts"], [number])
            self.assertEqual(summary["pending_count"], 0)
            self.assertIsNone(summary["blocked_reason"])
            saved = json.loads((Path(temporary) / "attempts/000001.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "interrupted_usage_reserved")
            self.assertIsNone(saved["usage"])
            self.assertEqual(saved["reserved_unknown_tokens"], bound)
            self.assertEqual(saved["request"], original["request"])
            self.assertEqual(saved["error_type"], "ProcessExited")
            new = budget.begin({"messages": [], "max_tokens": 100})
            budget.finish(new, usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})
            self.assertEqual(budget.summary()["actual_tokens"], 15)
            self.assertEqual(budget.summary()["effective_tokens"], bound + 15)
            self.assertEqual(json.loads((Path(temporary) / "budget.json").read_text())["effective_tokens"], bound + 15)
            cumulative = summarize_job_usage(temporary, "job-1")
            self.assertEqual(cumulative["unknown_usage_attempts"], 1)
            self.assertEqual(cumulative["reserved_unknown_tokens"], bound)
            self.assertFalse(cumulative["usage_complete"])
            with patch("philoagent.hss_budget.process_matches", return_value=False):
                repeat = reconcile_orphaned_attempts(temporary, "Repeat must not double reserve")
            self.assertEqual(repeat["reconciled_attempts"], [])
            self.assertEqual(budget.summary()["reserved_unknown_tokens"], bound)

    def test_reserved_unknown_cost_blocks_when_effective_budget_reaches_soft_limit(self):
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary, max_tokens=100)
            budget.begin({"messages": [], "max_tokens": 500})
            with patch("philoagent.hss_budget.process_matches", return_value=False):
                reconcile_orphaned_attempts(temporary, "User authorized orphan reconciliation")
            with self.assertRaisesRegex(BudgetRefused, "soft-stop"):
                budget.begin({"messages": [], "max_tokens": 100})
            self.assertEqual(budget.summary()["actual_tokens"], 0)

    def test_orphan_reconciliation_refuses_live_owner_unbounded_or_unreserved_unknown(self):
        for mode in ("live", "unbounded", "other_unknown"):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                budget = HSSBudget(temporary, max_inflight=2)
                if mode == "other_unknown":
                    first = budget.begin({"messages": [], "max_tokens": 100})
                    budget.finish(first, status="error", error_type="ConnectionError")
                elif mode == "unbounded":
                    budget.begin({"messages": []})
                else:
                    budget.begin({"messages": [], "max_tokens": 100})
                with patch("philoagent.hss_budget.process_matches", return_value=mode == "live"):
                    with self.assertRaises(BudgetRefused):
                        reconcile_orphaned_attempts(temporary, "User authorized safe reconciliation only")
                self.assertEqual(budget.summary().get("reserved_unknown_tokens", 0), 0)

    def test_reconciliation_preserves_maintenance_and_accepts_old_pid_reuse_evidence(self):
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary)
            budget.begin({"messages": [], "max_tokens": 100})
            path = Path(temporary) / "attempts/000001.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record.pop("process_identity")  # Original run did not record birth.
            path.write_text(json.dumps(record), encoding="utf-8")
            with budget._ledger_lock():
                state = budget._read()
                state["blocked_reason"] = "maintenance_repair_r3"
                budget._save(state)
            reused = {"pid": record["pid"], "started_unix": record["started_unix"] + 30,
                      "executable": "C:/tools/node_repl.exe"}
            with patch("philoagent.hss_budget.process_identity", return_value=reused):
                receipt = reconcile_orphaned_attempts(temporary, "Repair after app restart and PID reuse")
            self.assertEqual(receipt["reconciled_attempts"], [1])
            self.assertEqual(budget.summary()["blocked_reason"], "maintenance_repair_r3")
            with self.assertRaisesRegex(BudgetRefused, "maintenance_repair_r3"):
                budget.begin({"messages": [], "max_tokens": 100})

    def test_orphan_text_blocks_reserve_but_image_blocks_refuse(self):
        for kind, allowed in (("text", True), ("input_text", True), ("image_url", False)):
            with self.subTest(kind=kind), TemporaryDirectory() as temporary:
                budget = HSSBudget(temporary)
                request = {"model": "offline", "max_tokens": 32768, "messages": [
                    {"role": "user", "content": [{"type": kind, "text": "Chinese plain text"}]}]}
                budget.begin(request, job_id="job-1")
                with patch("philoagent.hss_budget.process_matches", return_value=False):
                    if allowed:
                        reconcile_orphaned_attempts(temporary, "Preserve text-block interruption cost")
                    else:
                        with self.assertRaisesRegex(BudgetRefused, "text-only"):
                            reconcile_orphaned_attempts(temporary, "Image size is not byte-token bounded")
                summary = budget.summary()
                self.assertEqual(summary["usage_accounted"], allowed)
                if allowed:
                    cumulative = summarize_job_usage(temporary, "job-1")
                    self.assertTrue(cumulative["usage_accounted"])
                    self.assertFalse(cumulative["usage_complete"])
                    self.assertEqual(cumulative["reserved_unknown_attempt_ids"], [1])

    def test_cumulative_job_usage_survives_distinct_worker_sessions_and_audit_file_reset(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            budget = HSSBudget(root / "budget")
            replies = iter([response(content=None), response()])
            first_sink = HSSAuditSink(root / "old-worker/job-1", root, budget,
                                      {"model": "offline", "transport_recovery": {}})
            first = HSSAuditedClient(FakeClient(lambda **kw: next(replies)), first_sink)
            with audit_scope(role="memory_retrieval", job_id="job-1"):
                _call(first)
            # A reset archives the whole prior job directory. New worker events
            # must stay local while cumulative accounting keeps every attempt.
            first_sink.path.unlink()
            second_sink = HSSAuditSink(root / "new-worker/job-1", root, budget, {"model": "offline"})
            second = HSSAuditedClient(FakeClient(lambda **kw: response(tokens=17)), second_sink)
            with audit_scope(role="actor", job_id="job-1"):
                _call(second)
            other_sink = HSSAuditSink(root / "new-worker/job-2", root, budget, {"model": "offline"})
            other = HSSAuditedClient(FakeClient(lambda **kw: response(tokens=19)), other_sink)
            with audit_scope(role="actor", job_id="job-2"):
                _call(other)
            usage = summarize_job_usage(root / "budget", "job-1")
            self.assertEqual(len(second_sink.events), 1)
            self.assertEqual(usage["scope"], "cumulative_global_budget_attempts_for_job_id")
            self.assertEqual(usage["request_count"], 3)
            self.assertEqual(usage["successful_request_count"], 2)
            self.assertEqual(usage["failed_request_count"], 1)
            self.assertEqual(usage["total_tokens"], 47)
            self.assertEqual(usage["prompt_tokens"] + usage["completion_tokens"], 47)
            self.assertEqual(usage["attempt_ids"], [1, 2, 3])
            self.assertEqual(usage["roles"]["memory_retrieval"]["total_tokens"], 30)
            self.assertEqual(usage["roles"]["actor"]["total_tokens"], 17)
            self.assertEqual(usage["pending_request_count"], 0)
            self.assertEqual(usage["unknown_usage_attempts"], 0)
            self.assertTrue(usage["usage_complete"])

    def test_cumulative_job_usage_distinguishes_live_pending_and_unknown_usage(self):
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary)
            number = budget.begin({"model": "offline"}, job_id="job-1", role="actor")
            pending = summarize_job_usage(temporary, "job-1")
            self.assertEqual(pending["request_count"], 1)
            self.assertEqual(pending["pending_request_count"], 1)
            self.assertEqual(pending["unknown_usage_attempts"], 0)
            self.assertFalse(pending["usage_complete"])
            budget.finish(number, error_type="ConnectionError", status="error")
            unknown = summarize_job_usage(temporary, "job-1")
            self.assertEqual(unknown["pending_request_count"], 0)
            self.assertEqual(unknown["unknown_usage_attempts"], 1)
            self.assertEqual(unknown["failed_request_count"], 1)
            self.assertEqual(unknown["total_tokens"], 0)
            self.assertFalse(unknown["usage_complete"])

    def test_request_is_durable_before_dispatch_and_full_output_redacts_transport_secrets(self):
        with TemporaryDirectory() as temporary:
            def dispatch(**kwargs):
                ledger = json.loads((Path(temporary) / "shared/budget.json").read_text(encoding="utf-8"))
                self.assertEqual(ledger["attempt_count"], 1)
                self.assertEqual(list(ledger["pending"]), ["1"])
                reader = HSSBudget(Path(temporary) / "shared")
                self.assertIsNone(reader.summary()["blocked_reason"])
                self.assertEqual(reader.summary()["pending_count"], 1)
                self.assertTrue((Path(temporary) / "shared/attempts/000001.json").is_file())
                return response(content="A complete literary response")

            client, raw, sink = _client(temporary, dispatch)
            with audit_scope(role="generation", provider="native", job_id="job-1", round=2, task_execution_id="task-1"):
                _call(client, max_tokens=60000, timeout=999, extra_headers={"Authorization": "secret-key"},
                      extra_body={"thinking": {"type": "enabled"}})
            request = raw.calls[0]
            self.assertEqual(request["max_tokens"], 60000)
            self.assertEqual(request["timeout"], 300)
            self.assertEqual(request["extra_body"]["thinking"], {"type": "enabled"})
            self.assertEqual(sink.events[0]["round"], 2)
            self.assertEqual(sink.events[0]["task_execution_id"], "task-1")
            self.assertEqual(sink.events[0]["response"]["choices"][0]["message"]["content"], "A complete literary response")
            all_audits = "\n".join(path.read_text(encoding="utf-8") for path in Path(temporary).rglob("*.json*"))
            self.assertNotIn("secret-key", all_audits)
            self.assertNotIn("secret-provider", all_audits)
            self.assertEqual(sink.budget.summary()["actual_tokens"], 15)

    def test_request_token_limits_are_preserved_and_only_missing_limits_use_role_defaults(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary)
            for role, expected in (("actor", 8192), ("analysis", 8192), ("generation", 60000),
                                   ("validation", 32768), ("judge", 8192), ("repair", 32768)):
                with audit_scope(role=role):
                    _call(client)
                self.assertEqual(raw.calls[-1]["max_tokens"], expected)
                self.assertNotIn("thinking", raw.calls[-1].get("extra_body", {}))
            _call(client, max_tokens=70000)
            self.assertEqual(raw.calls[-1]["max_tokens"], 70000)
            _call(client, max_completion_tokens=12000)
            self.assertEqual(raw.calls[-1]["max_completion_tokens"], 12000)
            self.assertNotIn("max_tokens", raw.calls[-1])

    def test_whole_budget_has_soft_stop_without_single_request_reservation(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, max_tokens=10)
            _call(client, max_tokens=60000)
            self.assertEqual(sink.budget.summary()["actual_tokens"], 15)
            with self.assertRaisesRegex(BudgetRefused, "soft-stop"):
                _call(client)
            self.assertEqual(len(raw.calls), 1)

    def test_unknown_usage_locks_the_entire_budget_after_reopen(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, lambda **kw: response(usage=False))
            with self.assertRaisesRegex(BudgetCallError, "MissingOrInvalidUsage"):
                _call(client)
            again = HSSBudget(Path(temporary) / "shared")
            self.assertEqual(again.summary()["unknown_usage_attempts"], 1)
            with self.assertRaisesRegex(BudgetRefused, "unknown_usage"):
                _call(client)
            self.assertEqual(len(raw.calls), 1)
            self.assertIsNone(sink.events[0]["usage"])

    def test_network_exception_is_sanitized_and_pending_restart_never_retries(self):
        with TemporaryDirectory() as temporary:
            def fail(**kwargs):
                raise ConnectionError("secret-key https://secret-provider.example/v1")

            client, raw, sink = _client(temporary, fail)
            with self.assertRaisesRegex(BudgetCallError, "^ConnectionError$"):
                _call(client)
            self.assertEqual(sink.budget.summary()["unknown_usage_attempts"], 1)
            with self.assertRaises(BudgetRefused):
                _call(client)
            self.assertEqual(len(raw.calls), 1)
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary)
            budget.begin({"model": "offline"})
            reopened = HSSBudget(temporary)
            with self.assertRaisesRegex(BudgetRefused, "unfinished_attempt"):
                reopened.begin({"model": "offline"})
            self.assertEqual(reopened.summary()["attempt_count"], 1)

    def test_empty_response_is_rejected_and_billed_tools_only_response_is_valid(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, lambda **kw: response(content=None))
            with self.assertRaises(EmptyModelContentError):
                _call(client)
            self.assertEqual(sink.budget.summary()["actual_tokens"], 15)
            self.assertEqual(sink.events[0]["status"], "error")
            raw.callback = lambda **kw: response(content=None, tools=[{"id": "tool-1", "type": "function"}])
            _call(client)
            self.assertEqual(sink.budget.summary()["actual_tokens"], 30)

    def test_known_usage_empty_response_recovers_with_bigger_audited_request(self):
        with TemporaryDirectory() as temporary:
            replies = iter([response(content=None), response(content="complete answer")])
            client, raw, sink = _client(temporary, lambda **kw: next(replies),
                                       protocol={"transport_recovery": {}})
            with audit_scope(role="memory_retrieval", task_execution_id="recovery-task"):
                result = _call(client, max_tokens=8192, extra_body={"thinking": {"type": "enabled"}})
            self.assertEqual(result.choices[0].message.content, "complete answer")
            self.assertEqual([call["max_tokens"] for call in raw.calls], [8192, 32768])
            self.assertEqual(raw.calls[1]["extra_body"], raw.calls[0]["extra_body"])
            self.assertEqual([event["status"] for event in sink.events], ["error", "success"])
            self.assertEqual(sink.events[0]["error_type"], "EmptyModelContentError")
            self.assertEqual(sink.events[1]["recovered_attempts"], [sink.events[0]["attempt"]])
            self.assertTrue(sink.events[1]["transport_recovered"])
            self.assertEqual(sink.events[1]["recovery_chain_id"], sink.events[0]["recovery_chain_id"])
            self.assertEqual(sink.events[1]["retry_of_attempt"], sink.events[0]["attempt"])
            self.assertEqual([event["recovery_index"] for event in sink.events], [0, 1])
            summary = sink.budget.summary()
            self.assertEqual((summary["attempt_count"], summary["actual_tokens"], summary["failed_requests"]), (2, 30, 1))
            saved = json.loads((Path(temporary) / "shared/attempts/000002.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["recovered_attempts"], [1])
            self.assertEqual(saved["response_finish_reasons"], ["stop"])

    def test_truncated_text_is_not_returned_as_success_and_recovery_is_bounded(self):
        with TemporaryDirectory() as temporary:
            replies = iter([response(content="partial report", finish_reason="length"),
                            response(content="still partial", finish_reason="length"),
                            response(content="complete report")])
            client, raw, sink = _client(temporary, lambda **kw: next(replies),
                                       protocol={"transport_recovery": {}})
            result = _call(client, max_completion_tokens=8192)
            self.assertEqual(result.choices[0].message.content, "complete report")
            self.assertEqual([call["max_completion_tokens"] for call in raw.calls], [8192, 32768, 65536])
            self.assertTrue(all("max_tokens" not in call for call in raw.calls))
            self.assertEqual([event["status"] for event in sink.events], ["error", "error", "success"])
            self.assertEqual(sink.events[0]["error_type"], "TruncatedModelResponse")
            self.assertEqual(sink.events[0]["response_finish_reasons"], ["length"])
            self.assertEqual(sink.events[-1]["recovered_attempts"], [1, 2])
            self.assertEqual(sink.budget.summary()["actual_tokens"], 45)

    def test_recovery_exhaustion_retains_every_known_usage_failure(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, lambda **kw: response(content=None),
                                       protocol={"transport_recovery": {}})
            with self.assertRaises(EmptyModelContentError):
                _call(client, max_tokens=800)
            self.assertEqual([call["max_tokens"] for call in raw.calls], [800, 32768, 65536])
            self.assertTrue(all(event["status"] == "error" for event in sink.events))
            self.assertTrue(all(not event.get("transport_recovered") for event in sink.events))
            self.assertEqual(sink.budget.summary()["failed_requests"], 3)
            self.assertEqual(sink.budget.summary()["actual_tokens"], 45)
            self.assertIsNone(sink.budget.summary()["blocked_reason"])

    def test_recovery_never_retries_unknown_usage_or_network_failure(self):
        for callback in (lambda **kw: response(content=None, usage=False),
                         lambda **kw: (_ for _ in ()).throw(ConnectionError("private transport failure"))):
            with self.subTest(callback=callback), TemporaryDirectory() as temporary:
                client, raw, sink = _client(temporary, callback, protocol={"transport_recovery": {}})
                with self.assertRaises(BudgetCallError):
                    _call(client)
                self.assertEqual(len(raw.calls), 1)
                self.assertEqual(sink.budget.summary()["unknown_usage_attempts"], 1)
                self.assertEqual(sink.budget.summary()["blocked_reason"], "unknown_usage")

    def test_recovery_obeys_request_hard_cap_and_configured_output_ceiling(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, lambda **kw: response(content=None), max_calls=2,
                                       protocol={"transport_recovery": {}})
            with self.assertRaisesRegex(BudgetRefused, "hard cap"):
                _call(client)
            self.assertEqual(len(raw.calls), 2)
            self.assertEqual(sink.budget.summary()["actual_tokens"], 30)
            self.assertEqual(sink.budget.summary()["pending_count"], 0)
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, lambda **kw: response(finish_reason="length"),
                                       protocol={"transport_recovery": {"max_output_tokens": 60000}})
            with self.assertRaisesRegex(BudgetCallError, "TruncatedModelResponse"):
                _call(client, max_tokens=60000)
            self.assertEqual(len(raw.calls), 1)

    def test_explicit_role_floors_prevent_small_native_requests_before_dispatch(self):
        with TemporaryDirectory() as temporary:
            client, raw, sink = _client(temporary, protocol={"role_min_output_tokens": {
                "trajectory_summary": 8192, "memory_retrieval": 16384}})
            with audit_scope(role="trajectory_summary"):
                _call(client, max_tokens=800)
            with audit_scope(role="memory_retrieval"):
                _call(client, max_completion_tokens=1000)
            self.assertEqual(raw.calls[0]["max_tokens"], 8192)
            self.assertEqual(raw.calls[1]["max_completion_tokens"], 16384)
            self.assertNotIn("max_tokens", raw.calls[1])
            self.assertEqual(sink.events[0]["original_requested_output_tokens"], 800)
            self.assertEqual(sink.events[0]["role_min_output_tokens"], 8192)

    def test_threads_wait_for_live_request_and_share_exact_hard_cap(self):
        with TemporaryDirectory() as temporary:
            clients = [_client(temporary, lambda **kw: (time.sleep(0.02), response())[1], max_calls=5, max_inflight=3)[0]
                       for _ in range(9)]

            def invoke(client):
                try:
                    _call(client)
                    return "done"
                except BudgetRefused as exc:
                    return str(exc)

            with ThreadPoolExecutor(max_workers=9) as executor:
                results = list(executor.map(invoke, clients))
            self.assertEqual(results.count("done"), 5)
            self.assertTrue(all(result == "done" or "hard cap" in result for result in results))
            summary = clients[0].sink.budget.summary()
            self.assertEqual(summary["attempt_count"], 5)
            self.assertEqual(summary["actual_tokens"], 75)
            self.assertEqual(summary["pending_count"], 0)

    def test_real_processes_share_durable_usage_without_overwriting_or_overlap(self):
        with TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            processes = [context.Process(target=_process_worker, args=(temporary, 3, queue)) for _ in range(3)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual([queue.get(timeout=2) for _ in processes], ["done"] * 3)
            queue.close()
            budget = HSSBudget(temporary, max_calls=100, max_tokens=100000)
            self.assertEqual(budget.summary()["attempt_count"], 9)
            self.assertEqual(budget.summary()["actual_tokens"], 90)
            records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((Path(temporary) / "attempts").glob("*.json"))]
            self.assertEqual(len(records), 9)
            self.assertTrue(all(previous["finished_unix"] <= following["started_unix"]
                                for previous, following in zip(records, records[1:])))

    def test_lock_wait_is_bounded_and_limits_cannot_change(self):
        with TemporaryDirectory() as temporary:
            budget = HSSBudget(temporary, lock_timeout=0.03)
            with budget.serial_call():
                with self.assertRaisesRegex(BudgetRefused, "timed out"):
                    with budget.serial_call():
                        self.fail("must not acquire twice")
            with self.assertRaisesRegex(BudgetRefused, "cannot be changed"):
                HSSBudget(temporary, max_calls=60001)

    def test_two_real_process_requests_are_simultaneously_pending_and_settle_exactly(self):
        with TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            ready, release = context.Queue(), context.Event()
            budget = HSSBudget(temporary, max_calls=3, max_tokens=100000, max_inflight=2)
            processes = [context.Process(target=_parallel_process_worker, args=(temporary, ready, release)) for _ in range(2)]
            try:
                for process in processes:
                    process.start()
                self.assertEqual(sorted(ready.get(timeout=15) for _ in processes), [1, 2])
                live = budget.summary()
                self.assertEqual(live["capacity"], 2)
                self.assertEqual(live["inflight_count"], 2)
                self.assertEqual(live["attempt_count"], 2)
                self.assertEqual(live["actual_tokens"], 0)
                self.assertIsNone(live["blocked_reason"])
            finally:
                release.set()
                for process in processes:
                    if process.pid is not None:
                        process.join(timeout=20)
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=5)
            self.assertTrue(all(process.exitcode == 0 for process in processes))
            self.assertEqual(budget.summary()["actual_tokens"], 40)
            self.assertEqual(budget.summary()["inflight_count"], 0)
            number = budget.begin({"messages": ["third"]})
            budget.finish(number, usage={"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20})
            with self.assertRaisesRegex(BudgetRefused, "hard cap"):
                budget.begin({"messages": ["must refuse fourth"]})
            self.assertEqual(budget.summary()["actual_tokens"], 60)
            ready.close()

    def test_dead_process_pending_blocks_other_dispatch_but_live_pending_does_not(self):
        with TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            ready, release = context.Queue(), context.Event()
            process = context.Process(target=_parallel_process_worker, args=(temporary, ready, release), kwargs={"orphan": True})
            process.start()
            self.assertEqual(ready.get(timeout=15), 1)
            process.join(timeout=20)
            self.assertEqual(process.exitcode, 0)
            budget = HSSBudget(temporary, max_calls=3, max_tokens=100000, max_inflight=2)
            with self.assertRaisesRegex(BudgetRefused, "orphaned_pending"):
                with budget.serial_call():
                    budget.begin({"messages": ["must not duplicate dead request"]})
            self.assertEqual(budget.summary()["blocked_reason"], "orphaned_pending_usage_unknown")
            self.assertEqual(budget.summary()["attempt_count"], 1)
            ready.close()

    def test_install_covers_new_clients_imported_aliases_with_options_and_judge_model(self):
        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = fake_openai.Client = FakeClient
        native = ModuleType("offline_native_aliases")
        native.OpenAI = FakeClient
        fake_litellm = ModuleType("litellm")
        fake_litellm.completion = lambda **kw: response()
        with TemporaryDirectory() as temporary, patch.dict("sys.modules", {
            "openai": fake_openai, "offline_native_aliases": native, "litellm": fake_litellm,
        }), ExitStack() as stack:
            sink = install_hss_audit(stack, Path(temporary) / "job", temporary, Path(temporary) / "shared",
                                     {"model": "deepseek-offline", "judge_model": "judge-offline"})
            one = fake_openai.OpenAI(api_key="secret-key", base_url="https://secret-provider.example/v1")
            two = native.OpenAI()
            three = fake_openai.Client()
            self.assertIsInstance(one, HSSAuditedClient)
            self.assertIsInstance(two, HSSAuditedClient)
            self.assertIsInstance(three, HSSAuditedClient)
            four = one.with_options(max_retries=9, timeout=999)
            with audit_scope(role="judge", task_execution_id="judge-task"):
                _call(four)
            self.assertEqual(one.client.calls[0]["model"], "judge-offline")
            self.assertEqual(one.client.options["max_retries"], 0)
            self.assertEqual(one.client.options["timeout"], 300)
            self.assertEqual(sink.events[0]["task_execution_id"], "judge-task")

    def test_litellm_repairs_use_same_budget_without_double_counting_nested_sdk(self):
        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = fake_openai.Client = FakeClient
        fake_litellm = ModuleType("litellm")
        observed = []

        def raw_completion(**kwargs):
            observed.append(kwargs)
            sdk = fake_openai.OpenAI()
            return _call(sdk)

        fake_litellm.completion = raw_completion
        with TemporaryDirectory() as temporary, patch.dict("sys.modules", {
            "openai": fake_openai, "litellm": fake_litellm,
        }), ExitStack() as stack:
            sink = install_hss_audit(stack, Path(temporary) / "job", temporary, Path(temporary) / "shared",
                                     {"model": "deepseek-offline"})
            with audit_scope(role="repair", provider="candidate-1"):
                fake_litellm.completion(model="openai/deepseek-offline", messages=[], api_key="secret-key")
            self.assertEqual(sink.budget.summary()["attempt_count"], 1)
            self.assertEqual(sink.budget.summary()["actual_tokens"], 15)
            self.assertEqual(len(sink.events), 1)
            self.assertEqual(sink.events[0]["transport"], "litellm")
            self.assertEqual(sink.events[0]["role"], "repair")
            self.assertEqual(observed[0]["num_retries"], 0)
            self.assertEqual(observed[0]["max_tokens"], 32768)
            self.assertNotIn("api_key", sink.events[0]["request"])

    def test_install_suppresses_litellm_import_time_dotenv_loading(self):
        import dotenv
        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = fake_openai.Client = FakeClient
        fake_litellm = ModuleType("litellm")
        fake_litellm.completion = lambda **kw: response()
        imported = []

        def import_litellm(name):
            self.assertEqual(name, "litellm")
            imported.append(dotenv.load_dotenv())
            return fake_litellm

        with TemporaryDirectory() as temporary, patch.dict("sys.modules", {"openai": fake_openai}), ExitStack() as stack:
            with patch("dotenv.load_dotenv", side_effect=AssertionError("must not read .env")), \
                    patch("philoagent.hss_budget.importlib.import_module", side_effect=import_litellm):
                install_hss_audit(stack, Path(temporary) / "job", temporary, Path(temporary) / "shared",
                                  {"model": "deepseek-offline"})
            self.assertEqual(imported, [False])


if __name__ == "__main__":
    unittest.main()
