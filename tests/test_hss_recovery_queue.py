"""Offline recovery scheduling: no SDK, real worker, or live ledger is used."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from philoagent import hss_recovery_queue as queue


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def offline(tmp_path, monkeypatch):
    root = tmp_path / "formal"
    root.mkdir()
    jobs = ["one", "two"]
    def manifest(ids=jobs):
        put(root / "manifest.json", {"jobs": [{"job_id": i} for i in ids],
            "adapter_sha256": "adapter", "upstream_sha256": "upstream", "protocol": {}})
    manifest()
    monkeypatch.setattr(queue, "source_hash", lambda p: "adapter" if p == queue.PROJECT / "philoagent" else "upstream")
    monkeypatch.setattr(queue, "process_owned", lambda r: bool(r.get("owned")))
    monkeypatch.setattr(queue, "process_matches", lambda *a, **k: False)
    monkeypatch.setattr(queue, "process_identity", lambda pid: {"pid": pid, "started_unix": float(pid), "executable": "python.exe"})
    starts, sleeps = [], []
    def sleep(seconds):
        sleeps.append(seconds)
        assert len(sleeps) < 20, "queue did not make progress"
    monkeypatch.setattr(queue.time, "sleep", sleep)
    class Child:
        def __init__(self, command, **kwargs):
            self.job_id = command[-1]
            self.pid = 100 + len(starts)
            self.calls = 0
            starts.append((self.job_id, {k: kwargs[k] for k in ("cwd", "stderr")}))
            assert kwargs["cwd"] == queue.PROJECT
            assert "run-job" in command
        def poll(self):
            self.calls += 1
            if self.calls == 1:
                return None
            put(root / "jobs" / self.job_id / "result.json", {"status": "complete"})
            return 0
    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    return root, starts, sleeps, manifest, Child


def test_hash_wait_does_not_consume_or_create_job_directory(offline, monkeypatch):
    root, starts, sleeps, _, _ = offline
    ready = False
    def source_hash(path):
        if path == queue.PROJECT / "philoagent":
            return "adapter" if ready else "editing"
        return "upstream"
    def sleep(seconds):
        nonlocal ready
        if ready:
            return
        state = json.loads((root / "recovery_supervisor.r9.json").read_text())
        assert not starts and not state["attempted_job_ids"]
        assert state["pending_job_ids"] == ["one", "two"]
        assert not (root / "jobs").exists()
        assert state["waiting_reason"] == "adapter_source_not_frozen"
        ready = True
        sleeps.append(seconds)
    monkeypatch.setattr(queue, "source_hash", source_hash)
    monkeypatch.setattr(queue.time, "sleep", sleep)
    result = queue.run_queue(root, ["one", "two"], max_running=2)
    assert result["status"] == "complete"
    assert [i for i, _ in starts] == ["one", "two"]
    assert sleeps


def test_global_cap_includes_legacy_workers_and_waits_for_them(offline, monkeypatch):
    root, starts, sleeps, manifest, _ = offline
    manifest(["one", "two", "legacy-a", "legacy-b"])
    for name in ("legacy-a", "legacy-b"):
        put(root / "jobs" / name / "job_status.json", {"status": "running", "owned": True, "pid": 1 if name.endswith("a") else 2})
    def sleep(seconds):
        if sleeps:
            return
        assert not starts, "legacy workers already occupy the full cap"
        for name in ("legacy-a", "legacy-b"):
            put(root / "jobs" / name / "job_status.json", {"status": "complete"})
        sleeps.append(seconds)
    monkeypatch.setattr(queue.time, "sleep", sleep)
    result = queue.run_queue(root, ["one", "two"], max_running=2)
    assert result["status"] == "complete" and len(starts) == 2
    assert sleeps


def test_all_27_coverage_deduplicates_and_preserves_existing_work(offline):
    root, starts, _, manifest, _ = offline
    ids = [f"job-{i}" for i in range(27)]
    manifest(ids)
    for name in ids[:7]:
        put(root / "jobs" / name / "result.json", {"status": "complete"})
    result = queue.run_queue(root, ids + ids[7:10], max_running=10)
    assert result["status"] == "complete"
    assert len(result["job_ids"]) == 27
    assert len(starts) == len(result["attempted_job_ids"]) == 20
    assert set(result["skipped_job_ids"]) == set(ids[:7])
    assert len({name for name, _ in starts}) == 20
    # A restarted reconciler never replays a consumed item.
    again = queue.run_queue(root, ids, max_running=10)
    assert again["status"] == "complete" and len(starts) == 20


def test_preinit_exit_failure_is_persisted_outside_missing_job(offline, monkeypatch):
    root, starts, _, _, Child = offline
    class Failed(Child):
        def poll(self):
            return 1
    monkeypatch.setattr(queue.subprocess, "Popen", Failed)
    result = queue.run_queue(root, ["one"], max_running=1)
    assert result["status"] == "needs_attention"
    assert result["jobs"]["one"]["status"] == "failed"
    assert result["jobs"]["one"]["exit_code"] == 1
    assert not (root / "jobs/one").exists()
    events = [json.loads(line) for line in (root / "queue_logs/one.events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["dispatch_admitted", "worker_started", "worker_finished"]
    queue.run_queue(root, ["one"], max_running=1)
    assert len(starts) == 1


def test_spawn_error_and_crash_gap_are_not_retried(offline, monkeypatch):
    root, starts, _, _, _ = offline
    def fail(*a, **k):
        starts.append(("one", {}))
        raise OSError("private message must not enter public status")
    monkeypatch.setattr(queue.subprocess, "Popen", fail)
    result = queue.run_queue(root, ["one"], max_running=1)
    assert result["jobs"]["one"]["status"] == "launch_failed"
    assert result["jobs"]["one"]["error_type"] == "OSError"
    assert "private message" not in json.dumps(result)
    queue.run_queue(root, ["one"], max_running=1)
    assert len(starts) == 1
    # Persisted admission before Popen is an ambiguous crash window, never permission to replay.
    state = json.loads((root / "recovery_supervisor.r9.json").read_text())
    state["jobs"]["one"]["status"] = "dispatching"
    put(root / "recovery_supervisor.r9.json", state)
    result = queue.run_queue(root, ["one"], max_running=1)
    assert result["jobs"]["one"]["status"] == "ambiguous_dispatch"
    assert len(starts) == 1


def test_owned_selected_job_is_skipped_without_second_worker(offline, monkeypatch):
    root, starts, _, _, _ = offline
    put(root / "jobs/one/job_status.json", {"status": "running", "owned": True, "pid": 33})
    def sleep(_):
        put(root / "jobs/one/result.json", {"status": "complete"})
        put(root / "jobs/one/job_status.json", {"status": "complete"})
    monkeypatch.setattr(queue.time, "sleep", sleep)
    result = queue.run_queue(root, ["one"], max_running=1)
    assert not starts and result["skipped_job_ids"] == ["one"]


def test_live_legacy_supervisor_requires_consumed_dispatch_evidence(offline, monkeypatch):
    root, starts, _, _, _ = offline
    put(root / "supervisor.json", {"status": "running", "owned": True, "pid": 99,
        "process_identity": {"started_unix": 2.0}})
    with pytest.raises(ValueError, match="legacy.*dispatch"):
        queue.run_queue(root, ["one"], max_running=1)
    assert not starts
    (root / "supervisor.repair-r7.log").write_text(json.dumps({"event": "job_finished", "job_id": "one", "exit_code": 1}) + "\n")
    monkeypatch.setattr(queue, "process_owned", lambda r: bool(r.get("owned")) and not starts)
    result = queue.run_queue(root, ["one"], max_running=1)
    assert len(starts) == 1
    assert result["legacy_supervisor"]["pid"] == 99
    assert result["legacy_dispatch_evidence"]["covered_job_ids"] == ["one"]


def test_unknown_job_ids_are_refused_before_any_launch(offline):
    root, starts, *_ = offline
    with pytest.raises(ValueError, match="manifest"):
        queue.run_queue(root, ["missing"])
    assert not starts and not (root / "queue_logs").exists()


def test_budget_maintenance_wait_keeps_ids_unconsumed_and_does_not_write_ledger(offline, monkeypatch):
    root, starts, sleeps, _, _ = offline
    path = root / "budget/budget.json"
    put(path, {"blocked_reason": "maintenance", "max_inflight_requests": 6})
    def sleep(seconds):
        if sleeps:
            return
        state = json.loads((root / queue.STATE_NAME).read_text())
        assert state["waiting_reason"] == "budget_blocked:maintenance"
        assert not starts and not state["attempted_job_ids"]
        put(path, {"blocked_reason": None, "max_inflight_requests": 6})
        sleeps.append(seconds)
    monkeypatch.setattr(queue.time, "sleep", sleep)
    result = queue.run_queue(root, ["one"])
    assert len(starts) == 1 and result["jobs"]["one"]["status"] == "complete"
    assert json.loads(path.read_text()) == {"blocked_reason": None, "max_inflight_requests": 6}


def test_os_lifetime_queue_lock_prevents_second_dispatcher(offline):
    root, starts, *_ = offline
    (root / "queue_logs").mkdir()
    with queue._file_lock(root / "queue_logs/recovery-r9.lock", 0):
        with pytest.raises(Exception, match="lock wait timed out"):
            queue.run_queue(root, ["one"])
    assert not starts


def test_queue_state_is_reloaded_after_lifetime_lock_acquisition(offline, monkeypatch):
    root, starts, *_ = offline
    state = {"kind": queue.KIND, "job_ids": ["one"], "legacy_supervisor": {},
             "jobs": {"one": {"status": "pending", "attempted": False}}}
    put(root / queue.STATE_NAME, state)
    @contextmanager
    def another_queue_finishes_before_lock(*args):
        state["jobs"]["one"].update(status="failed", attempted=True, exit_code=1)
        put(root / queue.STATE_NAME, state)
        yield
    monkeypatch.setattr(queue, "_file_lock", another_queue_finishes_before_lock)
    result = queue.run_queue(root, ["one"])
    assert not starts, "stale pre-lock pending state replayed an already consumed attempt"
    assert result["jobs"]["one"]["status"] == "failed"


def test_resume_with_missing_full_identity_still_checks_process_birth(offline, monkeypatch):
    root, starts, *_ = offline
    state = {"kind": queue.KIND, "job_ids": ["one"], "legacy_supervisor": {},
             "jobs": {"one": {"status": "running", "attempted": True, "pid": 333,
                               "observed_unix": 123.0, "process_identity": None}}}
    put(root / queue.STATE_NAME, state)
    calls = []
    def reused_pid(pid, **kwargs):
        calls.append((pid, kwargs))
        return False
    monkeypatch.setattr(queue, "process_matches", reused_pid)
    result = queue.run_queue(root, ["one"])
    assert not starts and result["jobs"]["one"]["status"] == "failed"
    assert calls == [(333, {"identity": None, "started_unix": 123.0, "executable": "python.exe"})]
