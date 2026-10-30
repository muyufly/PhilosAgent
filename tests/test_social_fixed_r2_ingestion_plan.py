"""Temp-only state restoration and single-credit proof for VerityLoom."""
import ast
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import shutil
import time
import typing

import pytest

from research.social_fixed_r2_recovery_plan_20261030 import (
    EXECUTION, JOB, plan_verity_ingestion,
)

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def incident(tmp_path):
    original = PROJECT / "runs/hss-business-formal-20261029-r01"
    job = Path("jobs") / JOB
    relative_record = job / "evolution/round_00/eval_batch/verity_loom_memory/1.json"
    original_record_path = original / relative_record
    record = json.loads(original_record_path.read_text(encoding="utf-8"))
    historical = None
    while (record.get("task_execution_id") != EXECUTION
            and record.get("native_ingestion_recovery", {}).get("method") == "restore_verity_loom_volatile_preingestion_then_retry"):
        recovery = record["native_ingestion_recovery"]
        receipt = json.loads(Path(recovery["receipt_path"]).read_text(encoding="utf8"))
        historical = receipt
        original_record_path = Path(receipt["backed_up_record"])
        assert __import__("hashlib").sha256(original_record_path.read_bytes()).hexdigest() == receipt["source_record_before_sha256"]
        record = json.loads(original_record_path.read_text(encoding="utf8"))
    identifiers = [json.loads(line)["attempt"] for line in (original / job / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("task_execution_id") == EXECUTION]
    paths = [job / "workspace/EvolveLab/providers/verity_loom_memory_provider.py",
             job / "workspace/storage/verity_loom_memory/loom_store.json",
             job / "workspace/storage/verity_loom_memory/task_snapshots.json", relative_record]
    paths.extend(Path("budget/attempts") / f"{identifier:06d}.json" for identifier in identifiers)
    for relative in paths:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_record_path if relative == relative_record else original / relative, tmp_path / relative)
    if historical is not None:
        context = historical["runtime_context"]
        store = {**context["store"], "items": [x for x in context["store"]["items"] if x["seeded"]]}
        for name, value in (("loom_store.json", store), ("task_snapshots.json", context["snapshots"])):
            raw = json.dumps(value, ensure_ascii=False, indent=1).replace("\n", "\r\n").encode("utf8")
            expected = next(v for k, v in historical["pinned_files_before"].items() if k.endswith(name))
            assert hashlib.sha256(raw).hexdigest() == expected
            path = tmp_path / job / "workspace/storage/verity_loom_memory" / name
            path.write_bytes(raw)
            timestamp = (max(x["created_ts"] for x in store["items"]) if name == "loom_store.json"
                         else next(iter(context["snapshots"].values()))["ts"])
            os.utime(path, (timestamp, timestamp))
    events = [json.loads((tmp_path / "budget/attempts" / f"{identifier:06d}.json").read_text(encoding="utf-8")) for identifier in identifiers]
    return tmp_path, record, events


def test_plan_preserves_disk_and_reconstructs_all_nine_original_credit_targets(incident):
    root, record, events = incident
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original = deepcopy(record)
    plan = plan_verity_ingestion(root, record, events)
    context = plan["runtime_context"]
    assert len(context["store"]["items"]) == len(context["task"]["surfaced_ids"]) == 9
    assert context["id_counter"] == 11 and context["task_counter"] == 2
    assert context["task"]["step"] == 7 and len(context["task"]["steps"]) == 6
    assert len(context["actor_memory_degradation"]["in_attempts"]) == 5
    assert all(item["appearances"] == item["successes"] == 0 for item in context["store"]["items"])
    assert all(item["created_ts"] is None for item in context["store"]["items"][3:])
    assert plan["storage_rollbacks"] == [] and record == original
    assert all(path.read_bytes() == value for path, value in before.items())


def test_original_outcome_method_credits_each_restored_target_exactly_once(incident):
    root, record, events = incident
    context = plan_verity_ingestion(root, record, events)["runtime_context"]
    source = root / "jobs" / JOB / "workspace/EvolveLab/providers/verity_loom_memory_provider.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    provider = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VerityLoomMemoryProvider")
    method = next(node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_outcome")
    subset = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Isolated", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[]))
    scope = {**vars(typing), "time": time}
    exec(compile(subset, "isolated-verity-credit", "exec"), scope)
    memory = scope["Isolated"]()
    memory._store = deepcopy(context["store"])
    memory.utility_alpha, memory.utility_prior, memory.success_ratio = .32, .5, .7
    assert memory._apply_outcome(context["task"]["surfaced_ids"], {"overall_norm": None, "success": True}) == 9
    assert all(item["appearances"] == item["successes"] == 1 for item in memory._store["items"])
    assert [item["utility"] for item in memory._store["items"][:3]] == [.628] * 3
    assert [item["utility"] for item in memory._store["items"][3:]] == [.6792] * 6
