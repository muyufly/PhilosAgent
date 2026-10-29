"""Migration preserves unknown reservations and reuses completed paid work."""

from pathlib import Path

from philoagent import hss_experiment as manager
from philoagent.hss_budget import HSSBudget
from philoagent.hss_repair_r3 import amend, release, task_usage


def test_task_usage_counts_all_sessions_and_keeps_reserved_unknown():
    record = {"task_execution_id": "new", "task_execution_ids": ["old", "new"],
              "phase_recovery_receipts": [{"recovered_attempts": [1]}]}
    events = [
        {"attempt": 1, "task_execution_id": "old", "status": "error",
         "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
        {"attempt": 2, "task_execution_id": "new", "status": "success",
         "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}},
        {"attempt": 3, "task_execution_id": "new", "status": "interrupted_usage_reserved",
         "usage": None, "reserved_unknown_tokens": 50000},
        {"attempt": 4, "task_execution_id": "other", "status": "success", "usage": {}},
    ]
    value = task_usage(record, events)
    assert value["total_tokens"] == 15
    assert value["request_count"] == 3
    assert value["reserved_unknown_tokens"] == 50000
    assert value["usage_complete"] is False and value["usage_accounted"] is True
    assert value["unrecovered_failed_attempt_ids"] == [3]


def test_native_quality_cost_uses_reservation_without_calling_it_confirmed_usage():
    from philoagent.hss_evaluation import RUBRICS, native_quality_summary, validate_grade
    grade = validate_grade({"dimensions": {name: {"score": 4, "reason": "Evidence"}
                                           for name in RUBRICS["humanities"]},
                            "overall_comment": "Complete work"}, "humanities")
    result = native_quality_summary([{"status": "completed", "grade": grade,
        "total_model_tokens": 15,
        "api_usage": {"reserved_unknown_tokens": 50000, "usage_complete": False}}])
    assert result["summary"]["tokens"] == {"total_tokens": 50015, "confirmed_total_tokens": 15,
                                            "reserved_unknown_tokens": 50000, "usage_complete": False}
    assert result["summary"]["quality_score"] == 80


def test_r3_migration_prepares_exact_phase_authorization_and_same_budget(tmp_path):
    root = tmp_path / "run"
    manifest = manager.prepare(root, values={"DEFAULT_MODEL": "offline", "OPENAI_API_KEY": "private"})
    manifest["protocol"]["revision"] = "repair-r2"
    manager.write_json(root / "manifest.json", manifest)
    job = root / "jobs" / manifest["jobs"][0]["job_id"]
    job.mkdir(parents=True)
    (job / "learning.jsonl").write_bytes((root / manifest["jobs"][0]["learning_file"]).read_bytes())
    task = manager._rows(job / "learning.jsonl")[0]
    record = {"task_execution_id": "paid", "task_identity": "old", "status": "actor_recovery_started",
              "task_id": task["task_id"], "global_task_index": 0, "provider": "lightweight_memory",
              "runtime_artifact": {}, "stage": "learning", "actor_recovery_attempts": 1,
              "memory_ingestion_complete": False}
    from philoagent.ablation_runtime import digest
    record["task_identity"] = digest({"task": task, "index": 0, "provider": "lightweight_memory",
        "artifact": {}, "stage": "learning", "dataset": "writingbench",
        "model": "offline", "protocol": manifest["protocol"]})
    path = job / "evolution/round_00/base_logs/1.json"
    manager.write_json(path, record)
    manager.write_json(job / "job_status.json", {"status": "running", "pid": -1})
    manager.write_json(root / "supervisor.json", {"status": "running", "pid": -1})
    b = HSSBudget(root / "budget", max_inflight=6)
    with b._ledger_lock():
        state = b._read()
        state.update(blocked_reason="maintenance_repair_20261029", maintenance={},
                     unknown_usage_attempts=1, reserved_unknown_attempts=[1],
                     reserved_unknown_tokens=50000, attempt_count=1)
        b._save(state)
    manager.write_json(root / "budget/attempts/000001.json", {
        "attempt": 1, "task_execution_id": "paid", "status": "interrupted_usage_reserved",
        "usage": None, "reserved_unknown_tokens": 50000})
    evidence = amend(root)
    new = manager.read_json(path)
    assert new["api_usage"]["usage_complete"] is False
    assert new["api_usage"]["reserved_unknown_tokens"] == 50000
    assert new["recovery_authorized"]["reserved_unknown_attempts"] == [1]
    assert new["recovery_authorized"]["known_failed_attempts"] == [1]
    assert new["recovery_authorized"]["allow_actor_retry"] is True
    assert new["actor_recovery_attempts"] == 1
    assert manager.read_json(job / "job_status.json")["status"] == "queued"
    assert amend(root) == evidence
    release(root)
    assert b.summary()["effective_tokens"] == 50000
    assert b.summary()["unknown_usage_attempts"] == 1
    assert b.summary()["blocked_reason"] is None
