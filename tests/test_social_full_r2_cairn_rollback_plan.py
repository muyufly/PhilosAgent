"""Temporary-copy proof that only Cairn's incomplete ingestion is undone."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from research.social_full_r2_cairn_ingestion_plan_20261030 import (
    EXECUTION, JOB, KINDS, plan_cairn_full_ingestion,
)

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def incident(tmp_path):
    original = PROJECT / "runs/hss-business-formal-20261029-r01"
    job = Path("jobs") / JOB
    relative_record = job / "evolution/round_00/eval_finalists/cairn_memory/1.json"
    record = json.loads((original / relative_record).read_text(encoding="utf8"))
    assert record["task_execution_id"] == EXECUTION
    identifiers = [json.loads(line)["attempt"] for line in (original / job / "api_calls.jsonl").read_text(encoding="utf8").splitlines()
                   if json.loads(line).get("task_execution_id") == EXECUTION]
    generated_paths = [path.relative_to(original) for path in (original / job / "evolution/round_00").glob("generated_system_*.json")]
    paths = [relative_record, job / "workspace/EvolveLab/providers/cairn_memory_provider.py",
             job / "workspace/storage/cairn_memory/cairn_store.json", *generated_paths,
             *(Path("budget/attempts") / f"{identifier:06d}.json" for identifier in identifiers)]
    for relative in paths:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / relative, tmp_path / relative)
    events = [json.loads((tmp_path / "budget/attempts" / f"{identifier:06d}.json").read_text(encoding="utf8")) for identifier in identifiers]
    return tmp_path, record, events


def files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_exact_rollback_preserves_actor_usage_and_meta_but_requires_new_credit(incident):
    root, record, events = incident
    before = files(root)
    original_record, original_events = deepcopy(record), deepcopy(events)
    plan = plan_cairn_full_ingestion(root, record, events)
    assert plan["forward_transaction_proof"]["native_transaction_reproduced"] is True
    assert plan["new_credit_request_required"] and not plan["offline_ingestion_retry_allowed"]
    assert plan["failed_attempts"] == [18970, 18974, 18985]
    rollback = plan["storage_rollbacks"][0]
    old = json.loads(before[Path(rollback["path"]).as_posix()])
    restored = json.loads(rollback["restored_utf8"])
    assert restored["meta"] == {**old["meta"], "task_counter": 2}
    selected = set(next(iter(plan["runtime_context"]["task_states"].values()))["delivered_memory_ids"])
    removed = []
    for kind in KINDS:
        for original in old[kind]:
            if original["created_task_index"] == 2:
                removed.append(original["id"])
                continue
            item = next(item for item in restored[kind] if item["id"] == original["id"])
            expected = deepcopy(original)
            if item["id"] in selected:
                expected["delivered_count"] -= 1
            assert item == expected
            assert item["usage_count"] == original["usage_count"]
            assert item["last_selected_task_index"] == original["last_selected_task_index"]
    assert len(removed) == 8
    assert plan["runtime_context"]["paid_ingestion_replays"][0]["attempt"] == 18996
    assert record == original_record and events == original_events and files(root) == before


@pytest.mark.parametrize("damage", ["credit_counter_drift", "paid_extraction_drift"])
def test_unmatched_transaction_evidence_is_rejected_without_writes(incident, damage):
    root, record, events = incident
    if damage == "credit_counter_drift":
        path = root / "jobs" / JOB / "workspace/storage/cairn_memory/cairn_store.json"
        store = json.loads(path.read_text(encoding="utf8"))
        store["strategic"][0]["delivered_count"] += 1
        path.write_text(json.dumps(store), encoding="utf8")
    else:
        event = next(event for event in events if event["attempt"] == 18996)
        payload = json.loads(event["response"]["choices"][0]["message"]["content"])
        payload["strategic"][0]["text"] += " Altered paid extraction."
        event["response"]["choices"][0]["message"]["content"] = json.dumps(payload)
    before = files(root)
    with pytest.raises(AssertionError):
        plan_cairn_full_ingestion(root, record, events)
    assert files(root) == before
