"""A committed ingestion with signed degraded memory needs only local finalization."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_hss_complete_actor_r17 import recovered
from philoagent.hss_evaluation import RUBRICS, validate_grade
from philoagent.hss_tasks_runtime import HSSResumeRequired, NativeHSSRunner


@pytest.fixture
def committed(recovered, monkeypatch):
    root, result = recovered
    operation = next(item for item in result["operations"] if item["job_id"] == "humanities-native-r2")
    path = root / operation["path"]
    record = json.loads(path.read_text(encoding="utf8"))
    domain = record["domain"]
    grade = validate_grade({"dimensions": {name: {"score": 4.0, "reason": "Retained fixture assessment."}
                                          for name in RUBRICS[domain]}, "overall_comment": "Retained fixture grade."}, domain)
    record.update(status="ingestion_failed", memory_ingestion_started=True, memory_ingestion_complete=True,
                  grade=grade, quality_score=grade["quality_score"], score=grade["quality_score"] / 100)
    record["recovery_authorized"]["next_phase"] = "integrity"
    path.write_text(json.dumps(record), encoding="utf8")
    callback = object.__new__(NativeHSSRunner)
    callback.protocol = {"resume_incomplete_tasks": True}
    callback.data_first = False
    callback.usage_reader = lambda: []
    callback._artifact = lambda _: deepcopy(record["runtime_artifact"])
    callback._task_identity = lambda *args: record["task_identity"]
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("external-work")
        raise AssertionError("Integrity finalization attempted Actor, Judge or ingestion work")
    callback._new_model = forbidden
    callback.native = SimpleNamespace(ToolCallingAgent=forbidden)
    callback.runner = SimpleNamespace(load_memory_provider=forbidden, TrajectoryData=forbidden)
    monkeypatch.setenv("PHILO_HSS_OBSERVE_ACCOUNTING", "1")
    return path, record, callback, calls


def execute(path, record, callback):
    task = {"task_id": record["task_id"], "domain": record["domain"], "question": record["question"]}
    return callback._run_task(task, int(path.stem) - 1, record["provider"], path.parent,
                              artifact=record["runtime_artifact"], stage="learning",
                              stage_id="fixture-integrity", dataset_name="writingbench")


def test_committed_ingestion_with_signed_fallback_only_saves_completed(committed):
    path, before, callback, calls = committed
    fallback = NativeHSSRunner._verified_actor_completion(before)
    assert len(fallback) == 3 and before["api_usage"]["blocking_failed_request_count"] == 0
    completed = execute(path, before, callback)
    assert completed["status"] == "completed" and calls == []
    assert completed["answer"] == before["answer"] and completed["grade"] == before["grade"]
    assert completed["memory_ingestion_complete"] is True
    assert completed["api_usage"]["failed_request_count"] == before["api_usage"]["failed_request_count"]
    assert set(completed["api_usage"]["unrecovered_failed_attempt_ids"]) == fallback
    assert completed["api_usage"]["blocking_failed_request_count"] == 0


@pytest.mark.parametrize("damage", ["unsigned", "real_core_failure"])
def test_unverified_or_genuinely_blocking_integrity_record_remains_refused(committed, damage):
    path, record, callback, calls = committed
    if damage == "unsigned":
        record.pop("native_actor_completion_recovery")
    else:
        record["api_usage"]["unrecovered_failed_attempt_ids"].append(999999)
        record["api_usage"]["unrecovered_failed_request_count"] += 1
        record["api_usage"]["failed_request_count"] += 1
        record["recovery_authorized"]["known_failed_attempts"].append(999999)
    path.write_text(json.dumps(record), encoding="utf8")
    before = path.read_bytes()
    with pytest.raises(HSSResumeRequired):
        execute(path, record, callback)
    assert path.read_bytes() == before and calls == []
