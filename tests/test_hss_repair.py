"""Offline repair migration retains paid evidence and consistent native memory."""

from copy import deepcopy
import json

import pytest

from philoagent import hss_experiment as manager
from philoagent.hss_budget import HSSBudget
from philoagent.hss_repair import MAINTENANCE, REVISION, amend, repaired_protocol, release


@pytest.fixture
def interrupted(tmp_path):
    root = tmp_path / "run"
    manifest = manager.prepare(root, values={"DEFAULT_MODEL": "offline", "OPENAI_API_KEY": "never-log"})
    for key in ("revision", "resume_incomplete_tasks", "actor_recovery_max_attempts",
                "judge_recovery_max_attempts", "transport_recovery", "role_min_output_tokens"):
        manifest["protocol"].pop(key, None)
    manifest["protocol"].update(actor_max_output_tokens=8192, analysis_max_output_tokens=8192,
                                judge_max_output_tokens=8192)
    manager.write_json(root / "manifest.json", manifest)
    spec = manifest["jobs"][0]
    job = root / "jobs" / spec["job_id"]
    job.mkdir(parents=True)
    (job / "learning.jsonl").write_bytes((root / spec["learning_file"]).read_bytes())
    manager.write_json(job / "job_status.json", {"status": "failed", "pid": -1})
    manager.write_json(root / "supervisor.json", {"status": "needs_attention", "pid": -1})
    path = job / "evolution/round_00/base_logs/1.json"
    task = manager._rows(root / spec["learning_file"])[0]
    record = {"status": "judge_failed", "task_execution_id": "paid-old",
              "task_identity": "old-identity", "task_id": task["task_id"],
              "global_task_index": 0, "provider": "lightweight_memory",
              "runtime_artifact": {"code_sha256": "same-code"}, "stage": "learning",
              "answer": "A complete paid answer.", "judge_output": "malformed",
              "memory_ingestion_complete": False, "api_usage": {"total_tokens": 10}}
    manager.write_json(path, record)
    budget = HSSBudget(root / "budget", max_inflight=6)
    with budget.serial_call():
        attempt = budget.begin({}, role="actor", job_id=spec["job_id"], task_execution_id="paid-old")
        budget.finish(attempt, usage={"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                      response={"choices": [{"finish_reason": "stop"}]})
    with budget._ledger_lock():
        state = budget._read()
        state.update(blocked_reason=MAINTENANCE, maintenance={"reason": "test repair"})
        budget._save(state)
    return root, manifest, path, record, budget


def test_repair_raises_actual_output_space_without_changing_matrix_or_budget():
    old = manager.protocol({"DEFAULT_MODEL": "offline"})
    new = repaired_protocol(old)
    assert new["actor_max_output_tokens"] >= 32768
    assert new["role_min_output_tokens"]["trajectory_summary"] >= 8192
    assert new["actor_recovery_max_attempts"] == 1
    assert new["transport_recovery"]["max_additional_attempts"] == 2
    for field in ("max_tokens", "max_calls", "rounds", "task_batch_x", "candidates", "top_t", "thinking"):
        assert new[field] == old[field]


def test_migration_reuses_paid_answer_keeps_original_and_same_budget(interrupted):
    root, before, path, record, budget = interrupted
    evidence = amend(root)
    migrated = manager.read_json(path)
    assert migrated["answer"] == record["answer"]
    assert migrated["judge_output"] == record["judge_output"]
    assert migrated["task_identity"] != record["task_identity"]
    assert migrated["recovery_authorized"]["allow_actor_retry"] is False
    assert manager.read_json(migrated["protocol_migration"]["original_record"]) == record
    assert manager.read_json(root / "manifest.json")["jobs"] == before["jobs"]
    assert budget.summary()["actual_tokens"] == 10
    assert amend(root) == evidence
    release(root)
    assert budget.summary()["blocked_reason"] is None
    assert budget.summary()["attempt_count"] == 1


def test_ingested_truncation_restarts_native_lineage_and_archives_memory(interrupted):
    root, _, path, record, budget = interrupted
    record.update(status="completed", memory_ingestion_complete=True)
    manager.write_json(path, record)
    job = path.parents[3]
    memory = job / "workspace/storage/lightweight_memory/memory.json"
    memory.parent.mkdir(parents=True)
    memory.write_text('["truncated work influenced this memory"]', encoding="utf-8")
    attempt_path = root / "budget/attempts/000001.json"
    event = manager.read_json(attempt_path)
    event["response"]["choices"][0]["finish_reason"] = "length"
    manager.write_json(attempt_path, event)
    evidence = amend(root)
    assert not job.exists()
    assert len(evidence["restarted_lineages"]) == 1
    archive = root / "revisions" / REVISION / "restarted_jobs" / job.name
    assert manager.read_json(archive / "workspace/storage/lightweight_memory/memory.json") == ["truncated work influenced this memory"]
    assert budget.summary()["actual_tokens"] == 10


def test_migration_refuses_live_or_unsettled_workers(interrupted):
    root, _, _, _, budget = interrupted
    with budget._ledger_lock():
        state = budget._read()
        state["pending"]["2"] = {"pid": -1}
        budget._save(state)
    with pytest.raises(ValueError, match="zero pending"):
        amend(root)
    assert not (root / "revisions").exists()


def test_uningested_truncation_gets_explicit_bounded_actor_recovery(interrupted):
    root, _, path, _, _ = interrupted
    attempt_path = root / "budget/attempts/000001.json"
    event = manager.read_json(attempt_path)
    event["response"]["choices"][0]["finish_reason"] = "length"
    manager.write_json(attempt_path, event)
    amend(root)
    migrated = manager.read_json(path)
    assert migrated["response_truncated"] is True
    assert migrated["recovery_authorized"]["allow_actor_retry"] is True


def test_unknown_old_failure_refuses_before_any_record_or_archive_write(interrupted):
    root, _, path, record, _ = interrupted
    event_path = root / "budget/attempts/000001.json"
    event = manager.read_json(event_path)
    event.update(status="error", usage=None)
    manager.write_json(event_path, event)
    with pytest.raises(ValueError, match="Unknown prior task usage"):
        amend(root)
    assert manager.read_json(path) == record
    assert not (root / "revisions").exists()
