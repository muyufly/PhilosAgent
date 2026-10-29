"""Offline accounting checks; never load .env or issue real model requests."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from philoagent.smoke_budget import Budget, BudgetCallError, BudgetRefused, wrap_native_model


def response(prompt=12, completion=3, *, usage=True, content="done"):
    message = SimpleNamespace(content=content, tool_calls=None,
                              model_dump=lambda: {"role": "assistant", "content": content})
    return SimpleNamespace(id="offline", choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=SimpleNamespace(model_dump=lambda: {
                               "prompt_tokens": prompt, "completion_tokens": completion,
                               "total_tokens": prompt + completion}) if usage else None)


class FakeClient:
    def __init__(self, callback):
        self.callback, self.calls, self.options = callback, [], None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.callback(**kwargs)


def wrapped(budget, callback=lambda **kwargs: response(), model_id="deepseek-offline"):
    client = FakeClient(callback)
    model = SimpleNamespace(model_id=model_id, client=client)
    wrap_native_model(model, budget, SimpleNamespace(items=["philosophy sentence"]), "offline", 50)
    return model, client


def call(model, **kwargs):
    return model.client.chat.completions.create(model=model.model_id,
               messages=[{"role": "user", "content": "philosophy sentence"}], **kwargs)


class SmokeBudgetTests(unittest.TestCase):
    def test_persisted_before_network_and_response_audit_excludes_transport_secrets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            budget = Budget(root)

            def dispatch(**kwargs):
                ledger = json.loads((root / "budget.json").read_text(encoding="utf-8"))
                self.assertEqual(ledger["attempts"][0]["status"], "started")
                self.assertEqual(len(ledger["attempts"][0]["request_sha256"]), 64)
                self.assertTrue((root / "attempts/0001.json").is_file())
                return response()

            model, client = wrapped(budget, dispatch)
            result = call(model, max_tokens=10000, temperature=0.9, timeout=900,
                          extra_headers={"Authorization": "secret-key"})
            self.assertEqual(result.choices[0].message.content, "done")
            self.assertEqual(client.options, {"timeout": 90, "max_retries": 0})
            self.assertEqual(client.calls[0]["max_tokens"], 50)
            self.assertEqual(client.calls[0]["temperature"], 0)
            self.assertNotIn("timeout", client.calls[0])
            self.assertEqual(client.calls[0]["extra_body"]["thinking"], {"type": "disabled"})
            files = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.json*"))
            self.assertNotIn("secret-key", files)
            audit = json.loads((root / "attempts/0001.json").read_text(encoding="utf-8"))
            self.assertIn("philosophy sentence", json.dumps(audit["request"]))
            self.assertEqual(audit["response"]["choices"][0]["message"]["content"], "done")
            self.assertEqual(budget.summary()["actual_tokens"], 15)
            self.assertIn("soft_stop", budget.summary()["token_limit_scope"])
            self.assertTrue(json.loads((root / "api_calls.jsonl").read_text(encoding="utf-8"))["orientation_present_in_request"])

    def test_shared_hard_cap_survives_reopen(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary, max_calls=2)
            first, first_client = wrapped(budget)
            second, second_client = wrapped(budget)
            call(first)
            call(second)
            reopened = Budget(temporary, max_calls=2)
            third, third_client = wrapped(reopened)
            with self.assertRaisesRegex(BudgetRefused, "hard cap"):
                call(third)
            self.assertEqual(len(first_client.calls) + len(second_client.calls), 2)
            self.assertEqual(third_client.calls, [])
            self.assertEqual(reopened.summary()["actual_tokens"], 30)

    def test_reservation_rejects_before_network_without_claiming_token_hard_cap(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary, max_tokens=100)
            model, client = wrapped(budget)
            with self.assertRaisesRegex(BudgetRefused, "soft-stop"):
                call(model)
            self.assertEqual(client.calls, [])
            self.assertEqual(budget.summary()["attempt_count"], 0)
            self.assertIn("no tokenizer-independent numeric bound", budget.summary()["token_tail_limit"])

    def test_missing_usage_blocks_next_call_and_reopened_budget(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary)
            model, client = wrapped(budget, lambda **kwargs: response(usage=False))
            with self.assertRaisesRegex(BudgetCallError, "MissingOrInvalidUsage"):
                call(model)
            with self.assertRaises(BudgetRefused):
                call(model)
            reopened = Budget(temporary)
            other, other_client = wrapped(reopened)
            with self.assertRaises(BudgetRefused):
                call(other)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(other_client.calls, [])
            self.assertEqual(reopened.summary()["unknown_usage_attempts"], 1)

    def test_exception_is_sanitized_and_prevents_native_retry(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary)

            def fail(**kwargs):
                raise ConnectionError("secret-url-with-secret-key")

            model, client = wrapped(budget, fail)
            with self.assertRaisesRegex(BudgetCallError, "^ConnectionError$"):
                call(model)
            with self.assertRaises(BudgetRefused):
                call(model)
            self.assertEqual(len(client.calls), 1)
            content = "\n".join(path.read_text(encoding="utf-8") for path in Path(temporary).rglob("*.json*"))
            self.assertNotIn("secret-url", content)

    def test_pending_attempt_blocks_after_restart(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary)
            budget.begin({"max_tokens": 50}, label="interrupted", request_sha256="abc", reservation=5000)
            reopened = Budget(temporary)
            model, client = wrapped(reopened)
            with self.assertRaisesRegex(BudgetRefused, "unfinished_attempt"):
                call(model)
            self.assertEqual(client.calls, [])

    def test_empty_content_is_settled_but_fail_closed(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary)
            model, client = wrapped(budget, lambda **kwargs: response(content=None))
            with self.assertRaisesRegex(BudgetCallError, "EmptyModelContentError"):
                call(model)
            self.assertEqual(budget.summary()["actual_tokens"], 15)
            with self.assertRaises(BudgetRefused):
                call(model)
            self.assertEqual(len(client.calls), 1)

    def test_reservation_underestimate_settles_actual_then_blocks(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary, max_tokens=10000)
            model, client = wrapped(budget, lambda **kwargs: response(prompt=12000))
            with self.assertRaisesRegex(BudgetCallError, "TokenReservationExceeded"):
                call(model)
            self.assertEqual(budget.summary()["actual_tokens"], 12003)
            with self.assertRaises(BudgetRefused):
                call(model)
            self.assertEqual(len(client.calls), 1)

    def test_output_overrun_and_invalid_usage_stop_further_calls(self):
        for provider_response, expected in ((response(completion=51), "OutputTokenLimitExceeded"),
                                            (response(prompt=-1), "MissingOrInvalidUsage")):
            with self.subTest(expected=expected), TemporaryDirectory() as temporary:
                budget = Budget(temporary)
                model, client = wrapped(budget, lambda **kwargs: provider_response)
                with self.assertRaisesRegex(BudgetCallError, expected):
                    call(model)
                with self.assertRaises(BudgetRefused):
                    call(model)
                self.assertEqual(len(client.calls), 1)

    def test_model_switch_and_thinking_enable_never_dispatch(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary)
            model, client = wrapped(budget)
            with self.assertRaisesRegex(BudgetRefused, "Model switching"):
                wrapped(budget, model_id="new-model")
            model.model_id = "mutated-model"
            with self.assertRaisesRegex(BudgetRefused, "Model switching"):
                call(model)
            model.model_id = "deepseek-offline"
            with patch.dict(os.environ, {"PHILO_SMOKE_THINKING": "enabled"}):
                with self.assertRaisesRegex(BudgetRefused, "requires"):
                    call(model)
            self.assertEqual(client.calls, [])

    def test_cannot_raise_limits_or_recover_missing_ledger_persistence(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary)
            with self.assertRaisesRegex(BudgetRefused, "limits cannot be changed"):
                Budget(temporary, max_calls=61)
            model, client = wrapped(budget)
            with patch("philoagent.smoke_budget._atomic_json", side_effect=OSError("secret-path")):
                with self.assertRaisesRegex(BudgetRefused, "Ledger persistence failed"):
                    call(model)
            with self.assertRaises(BudgetRefused):
                call(model)
            self.assertEqual(client.calls, [])

    def test_raw_client_wrapper_binds_request_model_and_reopens_same_budget(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary, max_calls=100, max_tokens=300000)
            client = FakeClient(lambda **kwargs: response())
            holder = SimpleNamespace(client=client)
            wrap_native_model(holder, budget, SimpleNamespace(items=[]), "summary", 50)
            holder.client.chat.completions.create(model="deepseek-offline", messages=[])
            self.assertEqual(budget.summary()["model"], "deepseek-offline")
            with self.assertRaisesRegex(BudgetRefused, "Model switching"):
                holder.client.chat.completions.create(model="different-model", messages=[])
            reopened = Budget(temporary, max_calls=100, max_tokens=300000)
            self.assertEqual(reopened.summary()["attempt_count"], 1)

    def test_serial_budget_instances_refresh_instead_of_overwriting_child_attempts(self):
        with TemporaryDirectory() as temporary:
            parent = Budget(temporary, max_calls=2)
            child = Budget(temporary, max_calls=2)
            parent_model, parent_client = wrapped(parent)
            child_model, child_client = wrapped(child)
            call(parent_model)
            call(child_model)
            self.assertEqual(parent.summary()["attempt_count"], 2)
            self.assertEqual(parent.summary()["actual_tokens"], 30)
            with self.assertRaisesRegex(BudgetRefused, "hard cap"):
                call(parent_model)
            self.assertEqual(len(parent_client.calls) + len(child_client.calls), 2)

    def test_extra_body_cannot_bypass_model_or_output_limits(self):
        with TemporaryDirectory() as temporary:
            budget = Budget(temporary)
            model, client = wrapped(budget)
            for extra in ({"model": "unapproved"}, {"max_tokens": 999999}, {"temperature": 1}):
                with self.subTest(extra=extra), self.assertRaisesRegex(BudgetRefused, "extra_body"):
                    call(model, extra_body=extra)
            self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
