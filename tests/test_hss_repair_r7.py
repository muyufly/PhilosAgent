"""Incident recovery keeps paid work and cannot fabricate unknown usage."""

from copy import deepcopy

import pytest

from philoagent.hss_repair_r7 import authorize_task, active_records
from philoagent.hss_experiment import write_json


def record(**extra):
    return {"task_identity": "unchanged", "task_execution_id": "paid-1", "status": "judge_failed",
            "answer": "A complete paid report.", "agent_messages": [], "stage": "learning", **extra}


def test_judge_authorization_preserves_answer_identity_and_confirmed_usage():
    original = record()
    event = {"attempt": 1, "task_execution_id": "paid-1", "status": "success",
             "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}}
    fixed = authorize_task(original, [event])
    assert fixed["answer"] == original["answer"] and fixed["task_identity"] == "unchanged"
    assert fixed["recovery_authorized"]["next_phase"] == "judge"
    assert not fixed["recovery_authorized"]["allow_actor_retry"]
    assert fixed["effective_model_tokens"] == fixed["confirmed_model_tokens"] == 300
    assert original == record()


def test_reserved_actor_error_preserves_unknown_and_links_explicit_recovery():
    original = record(status="actor_failed", answer="")
    event = {"attempt": 3835, "task_execution_id": "paid-1", "status": "error",
             "usage": None, "reserved_unknown_tokens": 50000}
    fixed = authorize_task(original, [event])
    assert not fixed["api_usage"]["usage_complete"] and fixed["api_usage"]["usage_accounted"]
    assert fixed["confirmed_model_tokens"] == 0 and fixed["effective_model_tokens"] == 50000
    assert fixed["recovery_authorized"]["reserved_unknown_attempts"] == [3835]
    assert fixed["recovery_authorized"]["known_failed_attempts"] == [3835]
    assert fixed["recovery_authorized"]["allow_actor_retry"]


def test_unreserved_usage_and_ambiguous_ingestion_are_refused():
    with pytest.raises(ValueError, match="Unreserved"):
        authorize_task(record(), [{"attempt": 1, "task_execution_id": "paid-1", "usage": None}])
    with pytest.raises(ValueError, match="Ambiguous"):
        authorize_task(record(status="ingestion_failed", memory_ingestion_started=True), [])


def test_active_record_scan_excludes_backup_trees(tmp_path):
    active = tmp_path / "evolution/round_00/eval_batch/provider/1.json"
    write_json(active, record())
    write_json(tmp_path / "evolution/round_00/_backup/eval_batch/provider/1.json", record())
    write_json(tmp_path / "evolution/round_00/eval_batch/provider/_recovery/1.json", record())
    assert list(active_records(tmp_path)) == [active]


def test_incident_recovery_reserves_cost_under_hold_and_releases_without_protocol_migration(tmp_path, monkeypatch):
    from philoagent import hss_repair_r7 as repair
    from philoagent.hss_budget import HSSBudget
    from philoagent.hss_experiment import read_json
    root = tmp_path / "incident"
    budget = HSSBudget(root / "budget", max_inflight=6)
    with budget._ledger_lock():
        state = budget._read()
        state["attempt_count"] = 3834
        budget._save(state)
    number = budget.begin({"messages": [{"role": "user", "content": "offline"}], "max_tokens": 20},
                          job_id="humanities-native-r1", task_execution_id="paid-1")
    assert number == 3835
    budget.finish(number, status="error", error_type="APIConnectionError")
    protocol = {"max_calls": 60000, "max_tokens": 350000000, "revision": "repair-r4"}
    write_json(root / "manifest.json", {"adapter_revision": "repair-r7", "adapter_sha256": "frozen",
        "upstream_sha256": "frozen", "protocol": protocol})
    write_json(root / "supervisor.json", {"status": "needs_attention"})
    write_json(root / "report.json", {"status": "incomplete"})
    job = root / "jobs/humanities-native-r1"
    write_json(job / "job_status.json", {"status": "failed"})
    task = job / "evolution/round_00/eval_finalists/child/1.json"
    write_json(task, record(status="actor_failed", answer=""))
    monkeypatch.setattr(repair, "source_hash", lambda path: "frozen")
    monkeypatch.setattr(repair, "finalist_recovery_hashes", lambda *args: {})
    monkeypatch.setattr(repair, "enum_collision_input_hashes", lambda *args: {})

    def local_recovery(*args, **kwargs):
        assert budget.summary()["blocked_reason"] == repair.MAINTENANCE
        return {"offline": True}

    monkeypatch.setattr(repair, "restore_incomplete_finalists", local_recovery)
    monkeypatch.setattr(repair, "repair_enum_collision", local_recovery)
    receipt = repair.amend(root)
    state = budget.summary()
    assert state["blocked_reason"] == repair.MAINTENANCE
    assert state["actual_tokens"] == 0 and state["unknown_usage_attempts"] == 1
    assert state["reserved_unknown_tokens"] > 0 and state["usage_accounted"]
    assert read_json(task)["task_identity"] == "unchanged"
    assert read_json(root / "manifest.json")["protocol"] == protocol
    assert read_json(root / "manifest.json")["controller_policy"] == repair.CONTROLLER_POLICY
    assert receipt["paid_candidates_regenerated"] == 0
    assert repair.amend(root) == receipt
    repair.release(root)
    assert budget.summary()["blocked_reason"] is None
    assert read_json(root / "budget/attempts/003835.json")["usage"] is None
