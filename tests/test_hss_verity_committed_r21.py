"""Real native wrapper proof for one exact offline transaction recovery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from philoagent.hss_tasks_runtime import HSSResumeRequired, NativeHSSRunner
from research.verity_committed_recovery_20261030 import COMMITTED, plan_verity_committed_ingestion
from test_hss_native_wrappers_r20 import ROOT, helper, native_provider, wrapped, Episode
from test_social_full_r2_cairn_rollback_plan import incident as cairn_incident

RELATIVE = Path("jobs/social-fixed-r2/evolution/round_00/eval_batch/verity_loom_memory/1.json")
SOURCE = Path("jobs/social-fixed-r2/workspace/EvolveLab/providers/verity_loom_memory_provider.py")
STORAGE = Path("jobs/social-fixed-r2/workspace/storage/verity_loom_memory")


@pytest.fixture
def incident(tmp_path):
    record = helper.read_json(ROOT / RELATIVE)
    log = ROOT / "jobs/social-fixed-r2/api_calls.jsonl"
    numbers = [e["attempt"] for e in map(json.loads, log.read_text(encoding="utf8").splitlines())
               if e.get("task_execution_id") in record["task_execution_ids"]]
    events = [helper.read_json(ROOT / f"budget/attempts/{n:06d}.json") for n in numbers]
    for relative in [RELATIVE, SOURCE, *[STORAGE / name for name in COMMITTED],
                     *[Path(f"budget/attempts/{e['attempt']:06d}.json") for e in events]]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    record["runtime_artifact"]["source_path"] = str((tmp_path / SOURCE).resolve())
    (tmp_path / RELATIVE).write_text(json.dumps(record), encoding="utf8")
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": [{"job_id": "social-fixed-r2"}]}), encoding="utf8")
    return tmp_path, record, events


def files(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def callback(root, record, monkeypatch):
    runner = object.__new__(NativeHSSRunner)
    runner.protocol = {"resume_incomplete_tasks": True}
    runner.data_first = False
    runner.usage_reader = lambda: []
    runner._artifact = lambda _: deepcopy(record["runtime_artifact"])
    runner._task_identity = lambda *args: record["task_identity"]
    calls = []
    def no_client():
        calls.append("client")
        raise AssertionError("Offline replay constructed a live API client")
    runner._new_model = no_client
    def load_provider(provider, model):
        inner = native_provider(root / SOURCE, record, root, model)
        outer = wrapped(inner, record, root / SOURCE, root, monkeypatch)
        assert outer.initialize()
        return outer
    runner.runner = SimpleNamespace(load_memory_provider=load_provider, TrajectoryData=Episode)
    monkeypatch.setenv("PHILO_HSS_OBSERVE_ACCOUNTING", "1")
    monkeypatch.setattr("philoagent.hss_tasks_runtime.task_prompt", lambda _: record["enhanced_question"])
    return runner, calls


def run(root, record, runner):
    return runner._run_task({"question": record["question"], "domain": record["domain"]}, 0, record["provider"],
                            (root / RELATIVE).parent, artifact=record["runtime_artifact"], stage="learning",
                            stage_id="r21-local", dataset_name="writingbench")


def test_exact_native_offline_recovery_preserves_cost_failures_and_saved_work(incident, monkeypatch):
    root, before, events = incident
    original = files(root)
    plan = plan_verity_committed_ingestion(root, before, events)
    assert files(root) == original
    assert plan["new_model_request"] is False
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    result = helper.prepare(root)
    assert result["resume_allowed"], result.get("blockers")
    record = helper.read_json(root / RELATIVE)
    evidence = helper.read_json(record["native_ingestion_recovery"]["receipt_path"])
    assert NativeHSSRunner._verified_ingestion_rollback(record, record["recovery_authorized"])
    assert len(evidence["storage_rollback_evidence"]) == 3
    assert not (root / STORAGE / "telemetry.jsonl").exists()
    backup = Path(result["receipt"]).parent
    assert all((backup / f"budget/attempts/{n:06d}.json").read_bytes() == (root / f"budget/attempts/{n:06d}.json").read_bytes()
               for n in (19025, 19035, 19052))
    runner, calls = callback(root, record, monkeypatch)
    completed = run(root, record, runner)
    assert completed["status"] == "completed" and completed["memory_ingestion_complete"] and calls == []
    assert completed["answer"] == before["answer"] and completed["grade"] == before["grade"]
    assert [x["original_attempt"] for x in completed["local_provider_replays"]] == [15359, 15391]
    assert completed["api_usage_current_execution"]["request_count"] == 0
    for key in ("request_count", "failed_request_count", "unknown_usage_attempts", "total_tokens"):
        assert completed["api_usage"][key] == before["api_usage"][key]
    assert completed["api_usage"]["usage_complete"] is False
    receipt = completed["phase_recovery_receipts"][-1]
    assert receipt["recovered_attempts"] == [19025, 19035, 19052]
    assert receipt["original_successful_attempts"] == [15359, 15391]
    assert receipt["new_model_request"] is receipt["new_sdk_success"] is False
    store = helper.read_json(root / STORAGE / "loom_store.json")
    ids = set(evidence["runtime_context"]["task"]["surfaced_ids"])
    assert len(ids) == 9 and all(x["appearances"] == x["successes"] == 1 for x in store["items"] if x["id"] in ids)
    assert len(store["items"]) == 11
    for n in (15359, 15391):
        p = Path(f"budget/attempts/{n:06d}.json")
        assert (root / p).read_bytes() == original[p]


def test_exact_mismatch_escapes_real_native_catches_without_model_or_storage_commit(incident, monkeypatch):
    root, record, _ = incident
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    assert helper.prepare(root)["resume_allowed"]
    record = helper.read_json(root / RELATIVE)
    runner, calls = callback(root, record, monkeypatch)
    # Change the native extraction input after verified restoration, not the
    # signed receipts. The provider would normally catch an Exception here.
    restore = runner._restore_ingestion_context
    def corrupt_input(memory, active):
        restore(memory, active)
        inner = runner._native_memory_target(memory)
        original = inner._call_llm
        inner._call_llm = lambda prompt, *args, **kwargs: original(prompt + " changed", *args, **kwargs)
    runner._restore_ingestion_context = corrupt_input
    before = {p: p.read_bytes() for p in (root / STORAGE).iterdir() if p.is_file()}
    with pytest.raises(HSSResumeRequired, match="offline ingestion prompt differs"):
        run(root, record, runner)
    failed = helper.read_json(root / RELATIVE)
    assert failed["status"] == "ingestion_failed" and not failed["memory_ingestion_complete"]
    assert failed["error_type"] == "_ExactPaidReplayMismatch" and calls == []
    assert {p: p.read_bytes() for p in (root / STORAGE).iterdir() if p.is_file()} == before
    assert failed["api_usage_current_execution"]["request_count"] == 0


def test_unmatched_later_commit_refuses_without_any_write(incident):
    root, record, events = incident
    path = root / STORAGE / "telemetry.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    before = files(root)
    with pytest.raises(AssertionError, match="Committed transaction changed"):
        plan_verity_committed_ingestion(root, record, events)
    assert files(root) == before


def test_offline_entrypoint_uses_real_framework_types_and_original_config(incident, monkeypatch):
    from research import replay_verity_ingestion_offline_20261030 as offline
    root, record, _ = incident
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    assert helper.prepare(root)["resume_allowed"]
    record = helper.read_json(root / RELATIVE)
    job = root / "jobs/social-fixed-r2"
    original_job = ROOT / "jobs/social-fixed-r2"
    for relative in [Path("workspace/EvolveLab") / name for name in ("__init__.py", "base_memory.py", "memory_types.py", "config.py")]:
        shutil.copy2(original_job / relative, job / relative)
    lineage = Path("lineage/verity_loom_memory/orientation.json")
    (job / lineage).parent.mkdir(parents=True)
    shutil.copy2(original_job / lineage, job / lineage)
    manifest = helper.read_json(ROOT / "manifest.json")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    monkeypatch.syspath_prepend(str(job / "workspace"))
    monkeypatch.chdir(job / "workspace")
    imported_before = set(sys.modules)
    try:
        with offline.no_network():
            runner = offline.make_runner(root, record)
            # Absolute source paths are relocated only for this temporary copy.
            # The live dataset/protocol identity was independently checked exact.
            runner._task_identity = lambda *args: record["task_identity"]
            tasks = [json.loads(line) for line in (original_job / "learning.jsonl").read_text(encoding="utf8").splitlines()]
            completed = runner._run_task(tasks[0], 0, record["provider"], (root / RELATIVE).parent,
                artifact=record["runtime_artifact"], stage=record["stage"], stage_id="r21-real-framework-temp", dataset_name="writingbench")
        assert completed["status"] == "completed"
        assert completed["api_usage_current_execution"]["request_count"] == 0
        assert [x["original_attempt"] for x in completed["local_provider_replays"]] == [15359, 15391]
    finally:
        for key in set(sys.modules) - imported_before:
            if key == "EvolveLab" or key.startswith("EvolveLab.") or key == "r21_exact_original_verity_provider":
                sys.modules.pop(key, None)


def test_cairn_v2_restores_real_inner_and_requires_credit_before_exact_extraction_cache(cairn_incident, monkeypatch):
    root, before, events = cairn_incident
    source = root / "jobs/social-full-r2/workspace/EvolveLab/providers/cairn_memory_provider.py"
    path = root / "jobs/social-full-r2/evolution/round_00/eval_finalists/cairn_memory/1.json"
    before["runtime_artifact"]["source_path"] = str(source.resolve())
    path.write_text(json.dumps(before), encoding="utf8")
    (root / "manifest.json").write_text(json.dumps({"jobs": [{"job_id": "social-full-r2"}]}), encoding="utf8")
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    result = helper.prepare(root)
    assert result["resume_allowed"], result.get("blockers")
    record = helper.read_json(path)
    assert record["native_ingestion_recovery"]["new_credit_request_required"] is True
    assert not record["native_ingestion_recovery"]["offline_ingestion_retry_allowed"]
    assert record["api_usage"]["unrecovered_failed_attempt_ids"] == [18970, 18974, 18985]
    assert NativeHSSRunner._verified_ingestion_rollback(record, record["recovery_authorized"])
    credit_event = next(x for x in events if x["attempt"] == 18970)
    generated = next(helper.read_json(p) for p in (root / "jobs/social-full-r2/evolution/round_00").glob("generated_system_*.json")
                     if helper.read_json(p)["memory_type_info"]["enum_value"] == "cairn_memory")
    calls = []
    def fake_credit_model(messages, **kwargs):
        # A unit-test response only: the live phase still needs a valid key.
        assert messages == credit_event["request"]["messages"]
        calls.append("real-credit-path")
        return SimpleNamespace(content=json.dumps({"followed": [], "violated": [], "not_applicable": [], "notes": "fixture"}))
    inner = native_provider(source, record, root, fake_credit_model,
        {**generated["config_updates"], "database_path": str(root / "jobs/social-full-r2/workspace/storage/cairn_memory/cairn_store.json")})
    outer = wrapped(inner, record, source, root, monkeypatch)
    assert outer.initialize()
    record.setdefault("recovery_history", []).append({"prior_task_execution_id": record["task_execution_id"]})
    record["task_execution_id"] = "cairn-v2-fixture-only"
    NativeHSSRunner._restore_ingestion_context(outer, record)
    assert "_db" not in outer.__dict__ and inner._task_counter == 2 and len(inner._task_states) == 1
    assert sum(len(inner._db[k]) for k in ("strategic", "operational", "boundaries")) == 30
    task = next(iter(inner._task_states.values()))
    selected = set(task["delivered_memory_ids"])
    originals = {x["id"]: deepcopy(x) for kind in ("strategic", "operational", "boundaries") for x in inner._db[kind]}
    success, description = outer.take_in_memory(Episode(record["question"], record["agent_messages"], record["agent_result"], {
        "item_index": record["global_task_index"], "status": "success", "is_correct": True,
        "quality_score": record["quality_score"], "quality_grade": record["grade"],
        "judge_comment": record["grade"]["overall_comment"], "domain": record["domain"],
        "full_query": record["enhanced_question"], "learning_success_threshold": record["learning_success_threshold"],
        "fitness_semantics": "continuous_quality_not_binary_correctness"}))
    assert success, description
    assert calls == ["real-credit-path"]
    assert [x["original_attempt"] for x in record["local_provider_replays"]] == [18996]
    assert all(not x["new_model_request"] for x in record["local_provider_replays"])
    for kind in ("strategic", "operational", "boundaries"):
        for item in inner._db[kind]:
            if item["id"] in originals:
                previous = originals[item["id"]]
                assert item["usage_count"] == previous["usage_count"]
                assert item["delivered_count"] == previous["delivered_count"] + int(item["id"] in selected)
    # Direct native execution in a test is not evidence of a live SDK success.
    assert record["api_usage"]["unrecovered_failed_attempt_ids"] == [18970, 18974, 18985]
