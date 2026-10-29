"""Authorized input freeze, model configuration and manager invariants, offline."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from philoagent import hss_experiment as manager


VALUES = {"DEFAULT_MODEL": "deepseek-offline", "OPENAI_API_KEY": "never-serialize-me",
          "OPENAI_API_BASE": "https://private-endpoint.invalid/v1",
          "WEB_SEARCH_PROVIDER": "deepseek", "WEB_ACCESS_PROVIDER": "deepseek"}


def test_process_detection_is_read_only_and_can_see_the_current_process():
    assert manager.process_alive(os.getpid())
    assert not manager.process_alive(-1)
    assert not manager.process_alive(None)


def test_owned_process_rejects_reused_pid_and_checks_start_and_executable(monkeypatch):
    from philoagent import hss_budget
    calls = []
    def matches(pid, *, started_unix=None, executable=None):
        calls.append((pid, started_unix, executable))
        return False
    monkeypatch.setattr(hss_budget, "process_matches", matches)
    assert not manager.process_owned({"pid": 99, "started_utc": "2026-10-29T10:31:49+00:00"})
    assert calls[0][0] == 99
    assert calls[0][1] > 0
    assert calls[0][2] == "python.exe"


def test_total_budget_is_not_a_per_response_output_limit():
    config = manager.protocol(VALUES)
    assert config["max_tokens"] == 350000000
    assert config["max_calls"] == 60000
    assert config["actor_max_output_tokens"] == 32768
    assert config["judge_max_output_tokens"] == 16384
    assert config["generation_max_output_tokens"] == 60000
    assert config["actor_max_steps"] == 32
    assert config["analysis_max_steps"] == 20
    assert config["thinking"] == "provider-default"
    assert config["compact_provider"] is False
    assert config["per_task_spend_stop"] is False
    assert config["max_inflight_requests"] == 6


def test_public_configuration_never_exposes_credentials_or_endpoint():
    value = json.dumps(manager.configuration(VALUES))
    assert "never-serialize-me" not in value
    assert "private-endpoint" not in value
    assert manager.configuration(VALUES)["endpoint_sha256"]


def test_prepare_pins_full_authorized_matrix_and_exact_original_queries(tmp_path):
    root = tmp_path / "formal"
    manifest = manager.prepare(root, values=VALUES)
    assert len(manifest["jobs"]) == 27
    assert manifest["authorization"]["user_message"] == "批准开始正式运行"
    for repetition in (1, 2, 3):
        for domain in ("humanities", "social", "business"):
            jobs = [job for job in manifest["jobs"] if job["repetition"] == repetition and job["domain"] == domain]
            assert len(jobs) == 3
            for job in jobs:
                assert job["code_evolve"] is True
                assert job["orientation_evolve"] == (job["group"] == "full")
                learning = manager._rows(root / job["learning_file"])
                heldout = manager._rows(root / job["heldout_file"])
                assert [row["task_id"] for row in learning] == job["learning_order"]
                assert [row["task_id"] for row in heldout] == job["evaluation_order"]
                assert len(learning) == 8 and len(heldout) == 4
                assert set(job["learning_order"]).isdisjoint(job["evaluation_order"])
    text = (root / "manifest.json").read_text(encoding="utf-8")
    assert "never-serialize-me" not in text and "private-endpoint" not in text
    assert (root / "code_snapshot/philoagent/hss_experiment.py").is_file()
    assert not (root / "jobs").exists(), "Workers initialize isolated code lazily"
    assert manager.prepare(root, values=VALUES) == manifest


def test_frozen_inputs_or_model_changes_are_refused(tmp_path):
    root = tmp_path / "formal"
    manifest = manager.prepare(root, values=VALUES)
    task_file = root / manifest["jobs"][0]["learning_file"]
    task_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="initialized formal input"):
        manager.prepare(root, values=VALUES)
    root2 = tmp_path / "other"
    manager.prepare(root2, values=VALUES)
    with pytest.raises(ValueError, match="configuration changed"):
        manager.prepare(root2, values={**VALUES, "DEFAULT_MODEL": "another-model"})


def test_worker_source_history_pins_each_loaded_adapter_without_changing_protocol(tmp_path, monkeypatch):
    from philoagent import hss_native
    root = tmp_path / "formal"
    target = root / "jobs/job-1"
    target.mkdir(parents=True)
    protocol = manager.protocol(VALUES)
    protocol["revision"] = "repair-r4"
    original_protocol = dict(protocol)
    manifest = {"jobs": [{"job_id": "job-1", "heldout_file": "inputs/evaluation/job-1.jsonl"}],
                "protocol": protocol, "adapter_sha256": "frozen-r4-hash", "adapter_revision": "repair-r4",
                "adapter_snapshot": "revisions/repair-r4/code_snapshot.after/philoagent"}
    manager.write_json(root / "manifest.json", manifest)
    monkeypatch.setattr(manager, "initialize_job", lambda *args: target)
    monkeypatch.setattr(manager, "environment", lambda: VALUES)
    observed = []
    def execute(*args):
        source = manager.read_json(target / "worker_source.json")
        state = manager.read_json(target / "job_status.json")
        assert state["worker_adapter_sha256"] == source["adapter_sha256"]
        assert state["worker_source"] == source
        observed.append(source)
        if len(observed) == 1:
            raise RuntimeError("offline technical failure")
        return {"status": "complete"}
    monkeypatch.setattr(hss_native, "run_job", execute)
    with patch.dict(os.environ):
        with pytest.raises(RuntimeError, match="offline technical failure"):
            manager.run_one(root, "job-1")
        failed = manager.read_json(target / "job_status.json")
        assert failed["worker_source"] == observed[0]
        manifest.update(adapter_sha256="frozen-r5-hash", adapter_revision="repair-r5",
                        adapter_snapshot="revisions/repair-r5/code_snapshot.after/philoagent")
        manager.write_json(root / "manifest.json", manifest)
        result = manager.run_one(root, "job-1")
    assert result["worker_source"] == observed[1]
    assert result["worker_adapter_sha256"] == "frozen-r5-hash"
    assert manager.read_json(target / "job_status.json")["worker_source"] == observed[1]
    history = [json.loads(line) for line in (target / "source_history.jsonl").read_text(encoding="utf8").splitlines()]
    assert history == observed
    assert [item["adapter_revision"] for item in history] == ["repair-r4", "repair-r5"]
    assert [item["protocol_revision"] for item in history] == ["repair-r4", "repair-r4"]
    assert history[0]["snapshot_path"] == str(root / "revisions/repair-r4/code_snapshot.after/philoagent")
    assert history[1]["snapshot_path"] == str(root / "revisions/repair-r5/code_snapshot.after/philoagent")
    assert manifest["protocol"] == original_protocol
    assert history[0]["process_identity"]["pid"] == os.getpid()
    evidence = (target / "source_history.jsonl").read_text(encoding="utf8")
    assert "never-serialize-me" not in evidence and "private-endpoint" not in evidence


def test_concurrent_report_writers_do_not_share_the_same_temporary_file(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import threading
    barrier = threading.Barrier(2)
    real_replace = os.replace
    seen = []

    def synchronized_replace(source, target):
        if source not in seen:
            seen.append(source)
            barrier.wait(timeout=5)
        return real_replace(source, target)

    monkeypatch.setattr(manager.os, "replace", synchronized_replace)
    report = tmp_path / "report.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(manager.write_json, report, {"writer": index}) for index in range(2)]
        for future in futures:
            future.result()
    assert manager.read_json(report) in ({"writer": 0}, {"writer": 1})
    assert len(set(seen)) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_report_writer_retries_temporary_windows_destination_lock(tmp_path, monkeypatch):
    real_replace = os.replace
    calls = []
    sleeps = []

    def initially_locked(source, target):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("temporary Windows destination lock")
        return real_replace(source, target)

    monkeypatch.setattr(manager.os, "replace", initially_locked)
    monkeypatch.setattr(manager.time, "sleep", sleeps.append)
    report = tmp_path / "report.json"
    manager.write_json(report, {"writer": "confirmed"})
    assert manager.read_json(report) == {"writer": "confirmed"}
    assert len(calls) == 3 and len(set(calls)) == 1
    assert sleeps == [.01, .02]
    assert not list(tmp_path.glob("*.tmp"))


def test_report_writer_does_not_retry_permanent_lock_forever(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    def locked(source, target):
        calls.append(source)
        raise PermissionError("permanent destination lock")

    monkeypatch.setattr(manager.os, "replace", locked)
    monkeypatch.setattr(manager.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        manager.write_json(tmp_path / "report.json", {"writer": "unconfirmed"})
    assert len(calls) == 7
    assert len(sleeps) == 6 and sum(sleeps) < 1
