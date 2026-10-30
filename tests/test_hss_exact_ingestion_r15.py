"""Replay the two actual partial ingestions using copied, local-only evidence."""
from copy import deepcopy
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
spec = importlib.util.spec_from_file_location("exact_ingestion_helper", ROOT / "native_recovery_20261030/resume_prepare.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "native_recovery_20261030/test_fixture_r15", tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    result = helper.prepare(tmp_path)
    assert result["status"] == "prepared", result.get("blockers")
    return tmp_path, result


def test_exact_rollback_keeps_actor_usage_and_reuses_answer_grade(prepared):
    root, result = prepared
    assert len(result["operations"]) == 2
    for operation in result["operations"]:
        rec = helper.read_json(root / operation["path"])
        original = helper.read_json(ROOT / "native_recovery_20261030/test_fixture_r15" / operation["path"])
        assert rec["answer"] == original["answer"] and rec["grade"] == original["grade"]
        assert rec["status"] == "graded" and not rec.get("memory_ingestion_started")
        assert not rec["memory_ingestion_complete"]
        assert NativeHSSRunner._verified_ingestion_rollback(rec, rec["recovery_authorized"])
        assert rec["api_usage"]["unrecovered_failed_request_count"] in (2, 3)
    cl = helper.read_json(root / "jobs/humanities-full-r1/workspace/storage/credence_lattice/semantic_ledger.json")
    assert cl["meta"]["task_counter"] == 14
    assert all(x["stats"]["created_task"] != 14 for x in cl["items"])
    picked = next(x for x in cl["items"] if x["id"] == "sem-000003-64663c")
    assert picked["stats"]["usage"] == 1 and picked["stats"]["success"] == 0
    assert picked["stats"]["last_used"] == 1793294121.9252365
    lw = helper.read_json(root / "jobs/business-native-r1/workspace/storage/lightweight_memory/longterm_memory.json")
    assert lw["strategic"][11]["usage_count"] == 3
    assert lw["strategic"][11]["success_count"] == 2


def _resumed_lightweight_record(root, result):
    operation = next(x for x in result["operations"] if x["job_id"] == "business-native-r1")
    rec = helper.read_json(root / operation["path"])
    rec["recovery_history"].append({"prior_task_execution_id": rec["task_execution_id"]})
    rec["task_execution_id"] = "new-ingestion-execution"
    return rec


def test_runtime_restores_lost_native_ram_credit_snapshot(prepared):
    root, result = prepared
    record = _resumed_lightweight_record(root, result)
    db = helper.read_json(root / "jobs/business-native-r1/workspace/storage/lightweight_memory/longterm_memory.json")
    memory = SimpleNamespace(longterm_db=db, task_context={}, _task_snapshots={},
                             _compute_signature=lambda query: hashlib.md5(query.encode()).hexdigest())
    NativeHSSRunner._restore_ingestion_context(memory, record)
    expected = ["strategic_11", "strategic_13", "operational_16"]
    assert memory.task_context["used_memory_ids"] == expected
    assert memory._task_snapshots[memory._compute_signature(record["enhanced_question"])] == expected
    # Real LightweightMemory's successful ingestion credits these IDs once.
    for identity in memory.task_context["used_memory_ids"]:
        kind, index = identity.rsplit("_", 1)
        memory.longterm_db[kind][int(index)]["success_count"] += 1
    assert memory.longterm_db["strategic"][11]["success_count"] == 3


def test_unsigned_runtime_context_cannot_change_native_credit_targets(prepared):
    root, result = prepared
    record = _resumed_lightweight_record(root, result)
    record["native_ingestion_recovery"]["runtime_context"]["used_memory_ids"] = ["strategic_0"]
    memory = SimpleNamespace(task_context={})
    with pytest.raises(HSSResumeRequired):
        NativeHSSRunner._restore_ingestion_context(memory, record)
    assert memory.task_context == {}


def test_exact_plan_rejects_newer_mutation_without_writing(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "native_recovery_20261030/test_fixture_r15", tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    path = tmp_path / "jobs/humanities-full-r1/workspace/storage/credence_lattice/semantic_ledger.json"
    ledger = helper.read_json(path)
    ledger["meta"]["task_counter"] = 15
    path.write_text(json.dumps(ledger), encoding="utf8")
    before = {str(p): helper.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = helper.prepare(tmp_path)
    assert not result["resume_allowed"]
    assert before == {str(p): helper.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
