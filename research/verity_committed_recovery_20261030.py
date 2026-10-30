"""Read-only plan for the single committed Verity fallback transaction.

The previous extraction was denied, but its native empty-candidate fallback
credited nine items and wrote one telemetry row. Restore the exact old durable
bytes before replaying the two original paid JSON responses, without network.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from research.social_fixed_r2_recovery_plan_20261030 import SOURCE_SHA256

EXECUTION = "dce9d876967b4bbea5ba58037a78c46c"
METHOD = "rollback_verity_committed_fallback_then_replay_paid_ingestion"
OLD_STORE = "2fe09100b211cd76c3d4087c69057ece8ecc8ea0165921cb01498d6de975cd4e"
OLD_SNAPSHOT = "c89f38139af661a7e7253c7638e94431b452aa4f458e0565f0240d61c1cedd07"
COMMITTED = {
    "loom_store.json": "6f9163a5403075c27bb6d7504b8f44ae718c54b652074ba88f061bb33890134c",
    "task_snapshots.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "telemetry.jsonl": "871578fa0915d655577c422b76d693ec04da3150e7b5df33c8a908bcfe285fb7",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_bytes(value):
    # Original _write_json_atomic uses indent=1 and Windows text newlines.
    return json.dumps(value, ensure_ascii=False, indent=1).replace("\n", "\r\n").encode("utf8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def predecessor(record):
    """Return the old signed RAM context, checking its entire archive chain."""
    evidence = record["native_ingestion_recovery"]
    path = Path(evidence["receipt_path"])
    assert path.is_absolute() and sha(path) == evidence["receipt_sha256"]
    receipt = read_json(path)
    assert receipt["method"] == "restore_verity_loom_volatile_preingestion_then_retry"
    original = record
    seen = set()
    pins = {path: sha(path)}
    while original["task_execution_id"] != "07e6272e1a1240bebd5122fafab94f8c":
        assert original["task_execution_id"] not in seen
        seen.add(original["task_execution_id"])
        proof = original["native_ingestion_recovery"]
        proof_path = Path(proof["receipt_path"])
        assert proof_path.is_absolute() and sha(proof_path) == proof["receipt_sha256"]
        proof_receipt = read_json(proof_path)
        archived = Path(proof_receipt["backed_up_record"])
        assert archived.is_absolute() and sha(archived) == proof_receipt["source_record_before_sha256"]
        pins.update({proof_path: sha(proof_path), archived: sha(archived)})
        original = read_json(archived)
        assert original["task_identity"] == record["task_identity"]
        assert original["answer"] == record["answer"] and original["grade"] == record["grade"]
    return receipt, pins


def plan_verity_committed_ingestion(root, record, events):
    root = Path(root).resolve()
    assert record["task_execution_id"] == EXECUTION and record["provider"] == "verity_loom_memory"
    assert record["status"] == "ingestion_failed" and record["memory_ingestion_started"]
    assert not record["memory_ingestion_complete"]
    assert record["grade"]["quality_score"] == record["quality_score"] == 75.4 and len(record["answer"]) == 3796
    job = root / "jobs/social-fixed-r2"
    source = job / "workspace/EvolveLab/providers/verity_loom_memory_provider.py"
    assert sha(source) == record["runtime_artifact"]["code_sha256"] == SOURCE_SHA256
    storage = job / "workspace/storage/verity_loom_memory"
    assert {p.name for p in storage.iterdir() if p.is_file()} == set(COMMITTED), "Unexpected later storage file"
    assert all(sha(storage / name) == digest for name, digest in COMMITTED.items()), "Committed transaction changed"
    receipt, old_pins = predecessor(record)
    context = deepcopy(receipt["runtime_context"])
    assert context["kind"] == "verity_loom_preingestion" and context["query"] == record["enhanced_question"]
    old_store = {**context["store"], "items": [deepcopy(x) for x in context["store"]["items"] if x["seeded"]]}
    originals = {"loom_store.json": native_bytes(old_store), "task_snapshots.json": native_bytes(context["snapshots"])}
    assert hashlib.sha256(originals["loom_store.json"]).hexdigest() == OLD_STORE
    assert hashlib.sha256(originals["task_snapshots.json"]).hexdigest() == OLD_SNAPSHOT
    assert next(v for k, v in receipt["pinned_files_before"].items() if k.endswith("loom_store.json")) == OLD_STORE
    assert next(v for k, v in receipt["pinned_files_before"].items() if k.endswith("task_snapshots.json")) == OLD_SNAPSHOT
    current = [e for e in events if e.get("task_execution_id") == EXECUTION]
    assert [e["attempt"] for e in sorted(current, key=lambda e: e["attempt"])] == [19025, 19035, 19052]
    assert all(e["role"] == "memory_ingestion" and e["status"] == "error" and e["response"] is None for e in current)
    assert [e["http_status"] for e in sorted(current, key=lambda e: e["attempt"])] == [429, 429, 402]
    assert record["api_usage"]["unrecovered_failed_attempt_ids"] == [19025, 19035, 19052]
    by_id = {e["attempt"]: e for e in events}
    references = []
    for attempt in (15359, 15391):
        path = root / f"budget/attempts/{attempt:06d}.json"
        event = read_json(path)
        assert event == by_id[attempt] and event["status"] == "success" and event["role"] == "memory_ingestion"
        assert len(event["response"]["choices"]) == 1 and event["response"]["choices"][0]["finish_reason"] == "stop"
        assert event["task_execution_id"] == "07e6272e1a1240bebd5122fafab94f8c"
        references.append({"attempt": attempt, "receipt_path": str(path), "receipt_sha256": sha(path)})
    assert all(e["request"]["messages"] == by_id[15359]["request"]["messages"] for e in current)
    context["paid_ingestion_replays"] = references
    context["paid_ingestion_replay_policy"] = "exact_sequence_offline_only"
    pins = {source: sha(source), **old_pins}
    for event in events:
        path = root / f"budget/attempts/{event['attempt']:06d}.json"
        assert read_json(path) == event
        pins[path] = sha(path)
    rollbacks = [{"path": str((storage / name).relative_to(root)), "before_sha256": COMMITTED[name],
                  "after_sha256": hashlib.sha256(raw).hexdigest(), "restored_utf8": raw.decode("utf8")}
                 for name, raw in originals.items()]
    rollbacks.append({"path": str((storage / "telemetry.jsonl").relative_to(root)),
                      "before_sha256": COMMITTED["telemetry.jsonl"], "after_sha256": None,
                      "operation": "remove_file", "reason": "Only failed transaction telemetry; original file was absent"})
    return {"method": METHOD, "reason": "Exact single committed fallback transaction; restore historical hash-pinned precredit state and replay original paid responses with no network",
            "provider_sha256": SOURCE_SHA256, "runtime_context": context, "storage_rollbacks": rollbacks,
            "pinned_files": {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): digest for p, digest in pins.items()},
            "backup_files": [f"budget/attempts/{n:06d}.json" for n in (19025, 19035, 19052)],
            "saved_answer_and_grade_reused": True, "new_model_request": False,
            "paid_ingestion_attempts_preserved": [15359, 15391], "historical_failed_attempts_preserved": [19025, 19035, 19052]}
