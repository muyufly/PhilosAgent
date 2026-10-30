"""Read-only, hash-pinned plan for the stopped VerityLoom first ingestion.

No live writes, API calls or process control. A later runtime revision must
explicitly support the returned context before a recovery helper applies it.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

JOB = "social-fixed-r2"
EXECUTION = "07e6272e1a1240bebd5122fafab94f8c"
PROVIDER = "verity_loom_memory"
SOURCE_SHA256 = "fd2cfd885e682654a8e34b85b1813c5c89b9c20dac324460f8c62c3acb3bcc7e"
STEP_ATTEMPTS = [15057, 15084, 15145, 15204, 15249, 15289]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response(event):
    choices = event["response"]["choices"]
    assert event["status"] == "success" and len(choices) == 1 and choices[0]["finish_reason"] == "stop"
    assert isinstance(event["usage"], dict) and event["usage"]["total_tokens"] > 0
    return json.loads(choices[0]["message"]["content"])


def format_context(messages):
    parts = []
    for message in messages:
        value = message.get("content", [])
        text = " ".join(p.get("text", "") for p in value if p.get("type") == "text") if isinstance(value, list) else str(value)
        if text:
            parts.append(f"{message.get('role', '')}: {text}")
    return "\n".join(parts)


def plan_verity_ingestion(root, record, events):
    root = Path(root).resolve()
    job = root / "jobs" / JOB
    assert record["task_execution_id"] == EXECUTION and record["provider"] == PROVIDER
    assert record["status"] == "ingestion_failed" and record["memory_ingestion_started"]
    assert not record["memory_ingestion_complete"] and record["quality_score"] == 75.4
    assert record["grade"]["quality_score"] == 75.4 and len(record["answer"]) == 3796
    source = job / "workspace/EvolveLab/providers/verity_loom_memory_provider.py"
    assert sha(source) == record["runtime_artifact"]["code_sha256"] == SOURCE_SHA256
    current = [e for e in events if e.get("task_execution_id") == EXECUTION]
    assert len(current) == 21 and all(e["status"] == "success" for e in current)
    by_id = {e["attempt"]: e for e in current}
    pin = response(by_id[15011])
    selected = response(by_id[15034])["selected_ids"]
    extraction = response(by_id[15359])
    decisions = response(by_id[15391])
    assert len(pin["constraints"]) == 29 and len(extraction["items"]) == len(decisions["decisions"]) == 4
    assert [e["attempt"] for e in current if e["role"] == "memory_ingestion"] == [15359, 15391]
    store_path = job / "workspace/storage/verity_loom_memory/loom_store.json"
    snapshot_path = store_path.with_name("task_snapshots.json")
    telemetry_path = store_path.with_name("telemetry.jsonl")
    assert not telemetry_path.exists(), "Ingestion telemetry would imply later durable writes"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    snapshots = json.loads(snapshot_path.read_text(encoding="utf-8"))
    query = record["enhanced_question"]
    key = hashlib.sha256(query.strip().lower().encode("utf8")).hexdigest()
    assert set(snapshots) == {key} and snapshots[key]["query"] == query
    ids = snapshots[key]["ids"]
    seed_ids = [item["id"] for item in store["items"]]
    assert len(ids) == 9 and len(store["items"]) == 3 and set(ids[6:]) == set(seed_ids)
    assert ids[6:8] == selected and store["meta"] == {}
    assert all(item["seeded"] and item["appearances"] == item["successes"] == item["reinforcements"] == 0
               and item["utility"] == .5 and item["last_surfaced"] is None for item in store["items"])
    judge = by_id[15332]
    assert store_path.stat().st_mtime < snapshot_path.stat().st_mtime < judge["finished_unix"]
    domain = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\- ]", "", pin["domain"]).strip().lower()
    domain = re.sub(r"\s+", "-", domain)[:40]
    pre_store = deepcopy(store)
    for identifier, content in zip(ids[:6], pin["constraints"][:6]):
        content = str(content).strip()
        assert len(content) >= 8
        item = deepcopy(store["items"][0])
        item.update(id=identifier, content=content, kind="CONSTRAINT", polarity="do", binding=True,
                    trigger="always in force for this task", scope=f"task:{key}", domain=domain, tier="instance",
                    provenance="task-source", tags=[], seeded=False, created_ts=None,
                    signature=hashlib.sha256(content.lower().encode("utf8")).hexdigest())
        pre_store["items"].append(item)
    updates = [response(by_id[number]) for number in STEP_ATTEMPTS]
    # Five IN updates failed at keyword binding before _make_item's body. Their
    # summaries/stage survived in RAM, but no item or counter increment occurred.
    assert [len(update["items"]) for update in updates] == [3, 2, 3, 3, 3, 0]
    messages = record["agent_messages"]
    assert len(messages) == 17 and messages[14]["role"] == "user" and messages[15]["role"] == "assistant"
    last_context = format_context(messages[:14])
    last_delta = format_context(messages[12:14]).strip()
    compact = last_delta if len(last_delta) <= 5000 else last_delta[:4997].rstrip() + "..."
    prompt = by_id[15289]["request"]["messages"][-1]["content"]
    if isinstance(prompt, list):
        prompt = " ".join(part.get("text", "") for part in prompt if part.get("type") == "text")
    assert "NEW CONTENT SINCE LAST STEP:\n" + compact + "\n\nPROVISIONAL WORKING PRINCIPLES" in prompt
    task = {"task_key": key, "query": query, "task_no": 2, "start_ts": None, "step": 7,
            "stage": updates[-1]["stage"], "domain": domain, "intent": str(pin["intent"]).strip()[:300],
            "pinned": ids[:6], "active_longterm": selected, "live_items": [], "live_new": [],
            "audit_ids": [], "last_audit": -999, "last_provision": 7, "last_context": last_context,
            "steps": [{"step": step, "summary": str(update["summary"]).strip()[:400]}
                      for step, update in enumerate(updates, 2)],
            "surfaced_ids": ids, "surfaced_events": 17, "cap_violations": 0}
    context = {"kind": "verity_loom_preingestion", "query": query, "task_key": key,
               "store": pre_store, "task": task, "snapshots": snapshots, "id_counter": 11,
               "task_counter": 2, "ingestions_since_maintenance": 0,
               "unknown_nonbehavioral_fields": ["task.start_ts", "store.items[3:9].created_ts"],
               "actor_memory_degradation": {"reason": "Original provider caught applies_now TypeError in five IN calls and returned empty memory; do not backfill guidance", "in_attempts": STEP_ATTEMPTS[:5]}}
    pinned_paths = [source, store_path, snapshot_path]
    pinned_paths.extend(root / "budget/attempts" / f"{e['attempt']:06d}.json" for e in current)
    return {"method": "restore_verity_loom_volatile_preingestion_then_retry",
            "reason": "No ingestion store/telemetry write occurred. Restore six original BEGIN constraints and all nine IDs before crediting the saved episode once.",
            "provider_sha256": SOURCE_SHA256, "failed_attempts": [], "judge_finished_unix": judge["finished_unix"],
            "runtime_context": context, "storage_rollbacks": [],
            "pinned_files": {str(path.relative_to(root)): sha(path) for path in pinned_paths},
            "saved_answer_and_grade_reused": True, "paid_ingestion_attempts_preserved": [15359, 15391],
            "adapter_compatibility": "Hash-pinned _make_item instance wrapper removes only applies_now; no provider bytes change and all other unexpected keywords still fail",
            "native_completion_required": True}
