"""Five real native Actor completions, replayed offline without new Actor calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from philoagent.hss_tasks_runtime import HSSResumeRequired, NativeHSSRunner

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "runs/hss-business-formal-20261029-r01"
spec = importlib.util.spec_from_file_location("actor_r17_helper", ROOT / "native_recovery_20261030/resume_prepare.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def recovered(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "native_recovery_20261030/test_fixture_r17", tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    for path in (tmp_path / "jobs").rglob("*.json"):
        rec = helper.read_json(path)
        if "runtime_artifact" not in rec:
            continue
        source = Path(rec["runtime_artifact"]["source_path"])
        job = path.parts[path.parts.index("jobs") + 1]
        rec["runtime_artifact"]["source_path"] = str(tmp_path / "jobs" / job / "workspace/EvolveLab/providers" / source.name)
        path.write_text(json.dumps(rec), encoding="utf8")
    result = helper.prepare(tmp_path)
    assert result["resume_allowed"], result.get("blockers")
    assert len(result["operations"]) == 5
    return tmp_path, result


def test_all_five_native_answers_restore_their_exact_final_trajectory(recovered):
    root, result = recovered
    for operation in result["operations"]:
        rec = helper.read_json(root / operation["path"])
        original = helper.read_json(ROOT / "native_recovery_20261030/test_fixture_r17" / operation["path"])
        assert rec["answer"] == original["answer"]
        assert rec["agent_messages"] == original["partial_agent_messages"]
        assert rec["agent_trajectory"] == original["partial_agent_trajectory"]
        assert rec["status"] == "answer_saved" and operation["next_phase"] == "judge"
        fallback = NativeHSSRunner._verified_actor_completion(rec)
        if operation["job_id"] in ("humanities-native-r2", "social-full-r3", "social-native-r1"):
            assert len(fallback) == 3
            assert rec["api_usage"]["unrecovered_failed_request_count"] == 3
            assert rec["api_usage"]["blocking_failed_request_count"] == 0
            linked = {i for x in rec.get("phase_recovery_receipts", []) for i in x["recovered_attempts"]}
            assert not (fallback & linked)
        else:
            assert not fallback
            assert rec["api_usage"]["unrecovered_failed_request_count"] == 0
        assert rec["api_usage"]["failed_request_count"] == original["api_usage"]["failed_request_count"]
        assert rec["api_usage"]["usage_complete"] is False


def test_judge_success_does_not_claim_failed_native_retrieval_was_recovered(recovered):
    root, result = recovered
    operation = next(x for x in result["operations"] if x["job_id"] == "social-full-r3")
    rec = helper.read_json(root / operation["path"])
    rec["prior_api_usage"] = dict(rec["api_usage"])
    before = list(rec["phase_recovery_receipts"])
    NativeHSSRunner._resolve_prior_phase_failures(rec, "judge")
    assert rec["phase_recovery_receipts"] == before
    assert rec["prior_api_usage"]["unrecovered_failed_request_count"] == 3
    callback = object.__new__(NativeHSSRunner)
    callback.usage_reader = lambda: []
    callback.data_first = False
    rec["task_execution_id"] = "new-judge-execution"
    callback._save(root / "after-judge.json", rec, __import__("time").monotonic())
    assert rec["api_usage"]["unrecovered_failed_request_count"] == 3
    assert rec["api_usage"]["blocking_failed_request_count"] == 0


@pytest.mark.parametrize("damage", ["answer", "messages", "receipt", "fallback_ids"])
def test_complete_actor_recovery_rejects_changed_evidence(recovered, damage):
    root, result = recovered
    rec = helper.read_json(root / result["operations"][0]["path"])
    if damage == "answer":
        rec["answer"] += "changed"
    elif damage == "messages":
        rec["agent_messages"].pop()
    elif damage == "receipt":
        rec["native_actor_completion_recovery"]["receipt_sha256"] = "0" * 64
    else:
        rec["native_actor_completion_recovery"]["native_fallback_failed_attempt_ids"] = [99999]
    with pytest.raises(HSSResumeRequired, match="Actor recovery evidence"):
        NativeHSSRunner._verified_actor_completion(rec)


def test_unsigned_actor_failed_record_cannot_use_judge_to_clear_memory_failure(tmp_path):
    path = tmp_path / "task.json"
    rec = {"task_identity": "identity", "status": "actor_failed", "answer": "Complete answer",
           "stage": "heldout", "task_execution_id": "old", "agent_messages": [],
           "recovery_authorized": {"revision": "r17", "known_failed_attempts": [9]},
           "api_usage": {"usage_complete": True, "unrecovered_failed_request_count": 1,
                         "unrecovered_failed_attempt_ids": [9]}}
    path.write_text(json.dumps(rec), encoding="utf8")
    callback = object.__new__(NativeHSSRunner)
    callback.protocol = {"resume_incomplete_tasks": True}
    callback.data_first = False
    with pytest.raises(HSSResumeRequired, match="explicit audited recovery"):
        callback._saved_task(path, "identity")


def _unlisted_execution_fixture(tmp_path, monkeypatch):
    fixture = ROOT / "native_recovery_20261030/test_fixture_r17"
    job = "humanities-native-r2"
    shutil.copytree(fixture / "jobs" / job, tmp_path / "jobs" / job)
    path = tmp_path / "jobs" / job / "evolution/round_00/eval_finalists/provenance_loom_memory/4.json"
    rec = helper.read_json(path)
    rec["task_execution_id"] = "a-completely-new-unlisted-native-execution"
    rec["task_execution_ids"] = [rec["task_execution_id"]]
    source = Path(rec["runtime_artifact"]["source_path"])
    rec["runtime_artifact"]["source_path"] = str(tmp_path / "jobs" / job / "workspace/EvolveLab/providers" / source.name)
    path.write_text(json.dumps(rec), encoding="utf8")
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": [{"job_id": job}]}), encoding="utf8")
    for identifier in [14877, 14668, 14708, 14737, 14741, 14765]:
        event = helper.read_json(fixture / f"budget/attempts/{identifier:06d}.json")
        event["task_execution_id"] = rec["task_execution_id"]
        dest = tmp_path / f"budget/attempts/{identifier:06d}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(event), encoding="utf8")
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    return path


def test_new_execution_recovers_only_from_paid_completion_and_native_retrieval_evidence(tmp_path, monkeypatch):
    path = _unlisted_execution_fixture(tmp_path, monkeypatch)
    result = helper.prepare(tmp_path)
    assert result["resume_allowed"], result.get("blockers")
    rec = helper.read_json(path)
    assert NativeHSSRunner._verified_actor_completion(rec) == {14668, 14708, 14737}
    assert rec["api_usage"]["unrecovered_failed_request_count"] == 3
    assert rec["api_usage"]["blocking_failed_request_count"] == 0


@pytest.mark.parametrize("damage", ["core_failure", "failed_response", "final_length", "different_replacement", "short_answer"])
def test_generic_completion_rejects_core_or_incomplete_evidence_without_mutation(tmp_path, monkeypatch, damage):
    path = _unlisted_execution_fixture(tmp_path, monkeypatch)
    target = tmp_path / "budget/attempts" / {
        "core_failure": "014668.json", "failed_response": "014668.json",
        "final_length": "014877.json", "different_replacement": "014765.json",
        "short_answer": "014877.json"}[damage]
    event = helper.read_json(target)
    if damage == "core_failure":
        event["role"] = "actor"
    elif damage == "failed_response":
        event["response"] = {"choices": []}
    elif damage == "final_length":
        event["response"]["choices"][0]["finish_reason"] = "length"
    elif damage == "different_replacement":
        event["request_sha256"] = "different request"
    else:
        rec = helper.read_json(path)
        rec["answer"] = rec["answer"][:100]
        path.write_text(json.dumps(rec), encoding="utf8")
    target.write_text(json.dumps(event), encoding="utf8")
    before = {str(p): helper.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = helper.prepare(tmp_path)
    assert not result["resume_allowed"]
    assert before == {str(p): helper.sha(p) for p in tmp_path.rglob("*") if p.is_file()}


def _aegis_paid_fixture(tmp_path, monkeypatch):
    relative = Path("jobs/social-native-r2/evolution/round_00/eval_batch/aegis_loom_memory/1.json")
    rec = helper.read_json(ROOT / relative)
    if rec.get("native_actor_completion_recovery"):
        receipt = helper.read_json(rec["native_actor_completion_recovery"]["receipt_path"])
        rec = helper.read_json(receipt["backed_up_record"])
    source = Path(rec["runtime_artifact"]["source_path"])
    copied_source = tmp_path / "jobs/social-native-r2/workspace/EvolveLab/providers" / source.name
    copied_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, copied_source)
    rec["runtime_artifact"]["source_path"] = str(copied_source.resolve())
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec), encoding="utf8")
    for identifier in [15825, 15828, 15845, 15847, 15848, 15859, 15994]:
        target = tmp_path / f"budget/attempts/{identifier:06d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f"budget/attempts/{identifier:06d}.json", target)
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": [{"job_id": "social-native-r2"}]}), encoding="utf8")
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    return path


def test_actual_aegis_429_memory_fallback_and_402_exact_replacement_preserve_complete_answer(tmp_path, monkeypatch):
    path = _aegis_paid_fixture(tmp_path, monkeypatch)
    original = helper.read_json(path)
    result = helper.prepare(tmp_path)
    assert result["resume_allowed"], result.get("blockers")
    rec = helper.read_json(path)
    assert len(rec["answer"]) == 3779 and rec["answer"] == original["answer"]
    assert NativeHSSRunner._verified_actor_completion(rec) == {15825, 15828, 15845}
    assert rec["api_usage"]["unrecovered_failed_request_count"] == 3
    assert rec["api_usage"]["blocking_failed_request_count"] == 0
    recovered = {i for x in rec["phase_recovery_receipts"] for i in x["recovered_attempts"]}
    assert {15847, 15848} <= recovered


def test_aegis_partial_memory_response_cannot_be_treated_as_empty_native_fallback(tmp_path, monkeypatch):
    _aegis_paid_fixture(tmp_path, monkeypatch)
    path = tmp_path / "budget/attempts/015825.json"
    rec = helper.read_json(path)
    rec["response"] = {"choices": [{"finish_reason": "length", "message": {"content": "partial memory output"}}]}
    path.write_text(json.dumps(rec), encoding="utf8")
    result = helper.prepare(tmp_path)
    assert not result["resume_allowed"]
