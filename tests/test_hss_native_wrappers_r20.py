"""Exercise native provider methods behind the actual OrientedProvider wrapper."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from philoagent.orientation import bootstrap, load
from philoagent.runtime import OrientedProvider
from philoagent.hss_tasks_runtime import NativeHSSRunner
from test_hss_verity_runtime_r19 import prepared_verity
from test_social_fixed_r2_ingestion_plan import incident

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "runs/hss-business-formal-20261029-r01"
spec = importlib.util.spec_from_file_location("wrapper_r20_helper", ROOT / "native_recovery_20261030/resume_prepare.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@dataclass
class Episode:
    query: str
    trajectory: list
    result: str
    metadata: dict


def native_provider(source, record, root, model, config=None):
    # Execute every original provider method; replace only the framework base
    # and annotation names so optional global registries are not imported.
    tree = ast.parse(source.read_text(encoding="utf8"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("EvolveLab"))]
    class Base:
        def __init__(self, memory_type, config):
            self.memory_type, self.config = memory_type, config
    scope = {"BaseMemoryProvider": Base,
             "MemoryType": SimpleNamespace(VERITY_LOOM_MEMORY="verity", CAIRN_MEMORY="cairn"),
             **{name: SimpleNamespace for name in ["MemoryItem", "MemoryItemType", "MemoryRequest", "MemoryResponse", "MemoryStatus", "TrajectoryData"]}}
    exec(compile(tree, str(source), "exec"), scope)
    class_name = "VerityLoomMemoryProvider" if record["provider"] == "verity_loom_memory" else "CairnMemoryProvider"
    inner = scope[class_name]({"storage_dir": str(source.parents[2] / "storage" / record["provider"]),
                              "model": model,
                              "philo_orientation_items": record["runtime_artifact"]["orientation_snapshot"]["items"],
                              **(config or {})})
    return inner


def wrapped(inner, record, source, root, monkeypatch):
    lineage = root / "wrapper-lineage"
    bootstrap(lineage, record["provider"], record["runtime_artifact"]["orientation_snapshot"]["items"])
    monkeypatch.setenv("PHILO_EVIDENCE_DIR", str(root / "wrapper-evidence"))
    return OrientedProvider(inner, load(lineage, record["provider"]), source, hashlib.sha256(source.read_bytes()).hexdigest())


def resumed(record):
    record.setdefault("recovery_history", []).append({"prior_task_execution_id": record["task_execution_id"]})
    record["task_execution_id"] = "new-ingestion-only-execution"


def trajectory(record):
    return Episode(record["question"], record["agent_messages"], record["agent_result"], {
        "is_correct": True, "status": "success", "quality_score": record["quality_score"],
        "quality_grade": record["grade"], "full_query": record["enhanced_question"],
        "domain": record["domain"], "judge_comment": record["grade"]["overall_comment"],
        "learning_success_threshold": 60, "fitness_semantics": "continuous_quality_not_binary_correctness"})


def test_real_verity_wrapper_patches_inner_and_credits_all_nine_ids_once(prepared_verity, monkeypatch):
    root, record, source = prepared_verity
    def model(messages, **kwargs):
        for attempt in (15359, 15391):
            event = helper.read_json(root / f"budget/attempts/{attempt:06d}.json")
            if event["request"]["messages"] == messages:
                return SimpleNamespace(content=event["response"]["choices"][0]["message"]["content"])
        raise AssertionError("Verity original paid prompt differs")
    inner = native_provider(source, record, root, model)
    wrapper = wrapped(inner, record, source, root, monkeypatch)
    assert wrapper.initialize()
    NativeHSSRunner._apply_provider_compatibility(wrapper, record, "ingestion")
    resumed(record)
    NativeHSSRunner._restore_ingestion_context(wrapper, record)
    original_ids = list(inner._task["surfaced_ids"])
    assert len(inner._store["items"]) == 9 and "_store" not in wrapper.__dict__
    success, description = wrapper.take_in_memory(trajectory(record))
    assert success, description
    originals = [item for item in inner._store["items"] if item["id"] in original_ids]
    assert len(originals) == 9 and all(item["appearances"] == item["successes"] == 1 for item in originals)
    assert "_make_item" not in wrapper.__dict__
    evidence = [json.loads(line) for path in (root / "wrapper-evidence").glob("*.jsonl") for line in path.read_text(encoding="utf8").splitlines()]
    assert any(x["operation"] == "take_in_memory" for x in evidence)


@pytest.fixture
def prepared_cairn(tmp_path, monkeypatch):
    relative = Path("jobs/social-fixed-r3/evolution/round_00/eval_batch/cairn_memory/1.json")
    raw_path = ROOT / relative
    record = helper.read_json(raw_path)
    if record.get("native_ingestion_recovery", {}).get("method") == "restore_cairn_volatile_preingestion_then_retry":
        receipt = helper.read_json(record["native_ingestion_recovery"]["receipt_path"])
        raw_path = Path(receipt["backed_up_record"])
        record = helper.read_json(raw_path)
    source_relative = Path("jobs/social-fixed-r3/workspace/EvolveLab/providers/cairn_memory_provider.py")
    store_relative = Path("jobs/social-fixed-r3/workspace/storage/cairn_memory/memory.json")
    from research.social_fixed_r3_recovery_plan_20261030 import ATTEMPTS
    for rel in [source_relative, store_relative, *[Path(f"budget/attempts/{n:06d}.json") for n in ATTEMPTS]]:
        out = tmp_path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out)
    if "receipt" in locals():
        db = deepcopy(receipt["runtime_context"]["db"])
        selected = receipt["runtime_context"]["task_context"]["delivered_memories"]
        for item in db["memories"]:
            if item["mem_id"] in selected:
                item["retrieval_trials"] -= 1
        raw = json.dumps(db, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf8")
        expected = next(v for k, v in receipt["pinned_files_before"].items() if k.endswith("memory.json"))
        assert hashlib.sha256(raw).hexdigest() == expected
        (tmp_path / store_relative).write_bytes(raw)
        judge = helper.read_json(tmp_path / "budget/attempts/016039.json")
        os.utime(tmp_path / store_relative, (judge["finished_unix"] - 10,) * 2)
    source = tmp_path / source_relative
    record["runtime_artifact"]["source_path"] = str(source.resolve())
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf8")
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": [{"job_id": "social-fixed-r3"}]}), encoding="utf8")
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    result = helper.prepare(tmp_path)
    assert result["resume_allowed"], result.get("blockers")
    return tmp_path, helper.read_json(path), source


def test_real_cairn_wrapper_replays_two_paid_jsons_without_a_new_model_call(prepared_cairn, monkeypatch):
    root, record, source = prepared_cairn
    calls = []
    def model(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Exact paid ingestion prompts must replay without network")
    inner = native_provider(source, record, root, model)
    wrapper = wrapped(inner, record, source, root, monkeypatch)
    assert wrapper.initialize()
    NativeHSSRunner._apply_provider_compatibility(wrapper, record, "ingestion")
    resumed(record)
    NativeHSSRunner._restore_ingestion_context(wrapper, record)
    assert all(inner._by_id[item["mem_id"]] is item for item in inner.db["memories"])
    selected = list(inner.task_context["delivered_memories"])
    assert len(selected) == 6
    success, description = wrapper.take_in_memory(trajectory(record))
    assert success, description
    assert not calls
    assert [x["original_attempt"] for x in record["local_provider_replays"]] == [16076, 16100]
    assert all(not x["new_usage_counted"] and not x["new_model_request"] for x in record["local_provider_replays"])
    assert all(inner._by_id[mid]["attempts"] == 1 and inner._by_id[mid]["retrieval_trials"] == 1 for mid in selected)
    assert inner.db["meta"]["total_ingestions"] == 1
    assert "db" not in wrapper.__dict__ and "_jaccard" not in wrapper.__dict__ and "_call_llm" not in wrapper.__dict__


def test_cairn_replay_mismatch_uses_original_model_and_outer_source_check_still_applies(prepared_cairn, monkeypatch):
    root, record, source = prepared_cairn
    calls = []
    def model(messages, **kwargs):
        calls.append(messages)
        return SimpleNamespace(content='{"new_response":true}')
    inner = native_provider(source, record, root, model)
    wrapper = wrapped(inner, record, source, root, monkeypatch)
    wrapper.initialize()
    NativeHSSRunner._apply_provider_compatibility(wrapper, record, "ingestion")
    resumed(record)
    NativeHSSRunner._restore_ingestion_context(wrapper, record)
    assert inner._call_llm("A distinct prompt without an existing paid response") == '{"new_response":true}'
    assert len(calls) == 1 and record.get("local_provider_replays", []) == []
    source.write_text(source.read_text(encoding="utf8") + "\n# source changed\n", encoding="utf8")
    with pytest.raises(RuntimeError, match="Provider source changed"):
        wrapper.take_in_memory(trajectory(record))
    assert len(calls) == 1


def test_actual_second_verity_failure_reauthorizes_current_execution_without_resetting_history(tmp_path, monkeypatch):
    relative = Path("jobs/social-fixed-r2/evolution/round_00/eval_batch/verity_loom_memory/1.json")
    record = helper.read_json(ROOT / relative)
    while record["task_execution_id"] != "0dcbbb86c7d9416285aa26a2ef445259":
        receipt = helper.read_json(record["native_ingestion_recovery"]["receipt_path"])
        record = helper.read_json(receipt["backed_up_record"])
    assert record["task_execution_id"] != "07e6272e1a1240bebd5122fafab94f8c"
    before_execution, before_ids = record["task_execution_id"], list(record["task_execution_ids"])
    ids = set(before_ids)
    events = [json.loads(line) for line in (ROOT / "jobs/social-fixed-r2/api_calls.jsonl").read_text(encoding="utf8").splitlines()]
    identifiers = {e["attempt"] for e in events if e.get("task_execution_id") in ids}
    source_relative = Path("jobs/social-fixed-r2/workspace/EvolveLab/providers/verity_loom_memory_provider.py")
    storage = Path("jobs/social-fixed-r2/workspace/storage/verity_loom_memory")
    relatives = [source_relative, storage / "loom_store.json", storage / "task_snapshots.json"]
    relatives += [Path(f"budget/attempts/{n:06d}.json") for n in identifiers]
    for rel in relatives:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, tmp_path / rel)
    prior_receipt = helper.read_json(record["native_ingestion_recovery"]["receipt_path"])
    context = prior_receipt["runtime_context"]
    seed_store = {**context["store"], "items": [x for x in context["store"]["items"] if x["seeded"]]}
    for name, value in (("loom_store.json", seed_store), ("task_snapshots.json", context["snapshots"])):
        raw = json.dumps(value, ensure_ascii=False, indent=1).replace("\n", "\r\n").encode("utf8")
        assert hashlib.sha256(raw).hexdigest() == next(v for k, v in prior_receipt["pinned_files_before"].items() if k.endswith(name))
        target_storage = tmp_path / storage / name
        target_storage.write_bytes(raw)
        timestamp = (max(x["created_ts"] for x in seed_store["items"]) if name == "loom_store.json"
                     else next(iter(context["snapshots"].values()))["ts"])
        os.utime(target_storage, (timestamp, timestamp))
    record["runtime_artifact"]["source_path"] = str((tmp_path / source_relative).resolve())
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record), encoding="utf8")
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": [{"job_id": "social-fixed-r2"}]}), encoding="utf8")
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    result = helper.prepare(tmp_path)
    assert result["resume_allowed"], result.get("blockers")
    prepared = helper.read_json(target)
    assert prepared["task_execution_id"] == before_execution and prepared["task_execution_ids"] == before_ids
    assert prepared["answer"] == record["answer"] and prepared["grade"] == record["grade"]
    assert prepared["native_ingestion_recovery"]["continued_failed_ingestion_successful_requests"]
    assert NativeHSSRunner._verified_ingestion_rollback(prepared, prepared["recovery_authorized"])
