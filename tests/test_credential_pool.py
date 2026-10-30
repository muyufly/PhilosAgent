"""Offline credential routing: all workers share one active credential."""

import json
from types import SimpleNamespace

import pytest

from philoagent.credential_pool import CredentialPool, is_quota_exhausted


def pool(tmp_path):
    config = tmp_path / "credentials.local.json"
    config.write_text(json.dumps({"version": 1, "active_slot": "A", "slots": {
        "A": {"api_key": "offline-secret-A"}, "B": {"api_key": "offline-secret-B"}}}), encoding="utf-8")
    return CredentialPool(config, tmp_path / "state.json")


def test_all_workers_use_active_until_confirmed_quota_exhaustion(tmp_path):
    first = pool(tmp_path)
    other = CredentialPool(first.config_path, first.state_path)
    assert first.select().slot == other.select().slot == "A"
    replacement = first.exhausted("A")
    assert replacement.slot == other.select().slot == "B"
    assert replacement.api_key == "offline-secret-B"
    assert other.exhausted("A").slot == "B", "Late old-key failures must not switch back"
    assert other.exhausted("B") is None
    with pytest.raises(RuntimeError, match="No usable credential"):
        first.select()
    public = first.state_path.read_text(encoding="utf-8") + repr(replacement)
    assert "offline-secret" not in public


def test_single_key_remains_usable_and_later_backup_can_resume_exhausted_state(tmp_path):
    config = tmp_path / "credentials.local.json"
    config.write_text(json.dumps({"version": 1, "active_slot": "A", "slots": {
        "A": {"api_key": "offline-secret-A"}}}), encoding="utf-8")
    routing = CredentialPool(config, tmp_path / "state.json")
    assert routing.select().slot == "A"
    assert routing.exhausted("A") is None
    with pytest.raises(RuntimeError, match="no unexhausted backup"):
        routing.select()
    value = json.loads(config.read_text())
    value["slots"]["B"] = {"api_key": "offline-secret-B"}
    config.write_text(json.dumps(value), encoding="utf-8")
    assert routing.select().slot == "B"


def test_late_quota_failures_from_multiple_workers_switch_only_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    routing = pool(tmp_path)
    with ThreadPoolExecutor(max_workers=6) as workers:
        choices = list(workers.map(lambda _: routing.exhausted("A"), range(10)))
    assert all(choice.slot == "B" for choice in choices)
    assert routing.select().slot == "B"


@pytest.mark.parametrize("status,body,expected", [
    (402, {"error": {"message": "Insufficient Balance"}}, True),
    (402, {"code": "insufficient_quota"}, True),
    (402, {"message": "余额不足"}, True),
    (402, {"message": "Payment Required"}, False),
    (429, {"code": "insufficient_quota"}, False),
    (401, {"message": "Insufficient balance"}, False),
])
def test_only_explicit_402_balance_or_quota_error_triggers_switch(status, body, expected):
    assert is_quota_exhausted(SimpleNamespace(status_code=status, body=body)) is expected


def test_openai_request_switches_key_once_and_retains_recovery_receipts(tmp_path, monkeypatch):
    from philoagent.hss_budget import HSSAuditSink, HSSAuditedClient, HSSBudget
    credential_pool = pool(tmp_path)
    monkeypatch.setenv("PHILO_HSS_CREDENTIAL_POOL", str(credential_pool.config_path))
    monkeypatch.setenv("PHILO_HSS_CREDENTIAL_STATE", str(credential_pool.state_path))
    budget = HSSBudget(tmp_path / "budget", max_calls=10, max_tokens=1000)
    state = json.loads(budget.path.read_text())
    state["enforcement_policy"] = {"mode": "observe_only"}
    budget.path.write_text(json.dumps(state), encoding="utf-8")
    sink = HSSAuditSink(tmp_path / "job", tmp_path, budget, {"model": "offline-model"})
    keys_used = []

    class OfflineQuotaError(Exception):
        status_code = 402
        body = {"error": {"message": "Insufficient Balance"}}

    class Client:
        api_key = "original-placeholder"
        base_url = "https://offline.invalid"

        def create(self, **kwargs):
            keys_used.append(self.api_key)
            if self.api_key == "offline-secret-A":
                raise OfflineQuotaError("do not serialize credentials")
            return SimpleNamespace(id="offline", choices=[SimpleNamespace(message=SimpleNamespace(
                content="complete answer", tool_calls=None), finish_reason="stop")],
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})

    raw = Client()
    raw.chat = SimpleNamespace(completions=SimpleNamespace(create=raw.create))
    client = HSSAuditedClient(raw, sink)
    result = client.create(model="offline-model", messages=[{"role": "user", "content": "task"}])
    assert result.choices[0].message.content == "complete answer"
    assert keys_used == ["offline-secret-A", "offline-secret-B"]
    receipts = [json.loads(path.read_text()) for path in sorted((tmp_path / "budget/attempts").glob("*.json"))]
    assert [record["credential_slot"] for record in receipts] == ["A", "B"]
    assert receipts[1]["recovered_attempts"] == [1]
    assert receipts[0]["credential_quota_exhausted"] is True
    assert receipts[0]["usage"] is None
    public = json.dumps(receipts) + sink.path.read_text()
    assert "offline-secret" not in public


def test_litellm_uses_shared_active_key_and_keeps_key_out_of_receipt(tmp_path, monkeypatch):
    from philoagent.hss_budget import HSSAuditSink, HSSBudget, _audited_dispatch
    routing = pool(tmp_path)
    routing.exhausted("A")
    monkeypatch.setenv("PHILO_HSS_CREDENTIAL_POOL", str(routing.config_path))
    monkeypatch.setenv("PHILO_HSS_CREDENTIAL_STATE", str(routing.state_path))
    sink = HSSAuditSink(tmp_path / "job", tmp_path, HSSBudget(tmp_path / "budget"), {"model": "offline"})
    requests = []
    def dispatch(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(id="offline", choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", tool_calls=None), finish_reason="stop")],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    _audited_dispatch(sink, dispatch, {"model": "openai/offline", "messages": []}, transport="litellm")
    assert requests[0]["api_key"] == "offline-secret-B"
    assert "api_key" not in sink.events[0]["request"]
    assert sink.events[0]["credential_slot"] == "B"
    assert "offline-secret" not in sink.path.read_text()


def test_native_worker_bootstrap_changes_only_credential_and_keeps_frozen_protocol(tmp_path, monkeypatch):
    import os
    from philoagent import hss_experiment as manager, hss_native

    routing = pool(tmp_path)
    values = {"DEFAULT_MODEL": "offline-model", "OPENAI_API_KEY": "old-placeholder",
              "OPENAI_API_BASE": "https://offline.invalid/v1", "WEB_SEARCH_PROVIDER": "deepseek",
              "WEB_ACCESS_PROVIDER": "deepseek", "PHILO_HSS_CREDENTIAL_POOL": str(routing.config_path)}
    root = tmp_path / "formal"
    target = root / "jobs/job-1"
    target.mkdir(parents=True)
    protocol = manager.protocol(values)
    manifest = {"jobs": [{"job_id": "job-1", "heldout_file": "heldout.jsonl"}],
                "protocol": protocol, "adapter_sha256": "offline-hash"}
    manager.write_json(root / "manifest.json", manifest)
    monkeypatch.setattr(manager, "initialize_job", lambda *args: target)
    monkeypatch.setattr(manager, "environment", lambda: dict(values))
    observed = []
    def execute(*args):
        observed.append((os.environ["OPENAI_API_KEY"], os.environ["OPENAI_API_BASE"], args[2]))
        return {"status": "complete"}
    monkeypatch.setattr(hss_native, "run_job", execute)
    # Patch the environment so bootstrap writes cannot affect other tests.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    manager.run_one(root, "job-1")
    assert observed == [("offline-secret-A", "https://offline.invalid/v1", protocol)]
    assert manager.read_json(root / "manifest.json") == manifest
    public = (target / "job_status.json").read_text() + (target / "result.json").read_text()
    assert "offline-secret" not in public
