"""Pure, hash-pinned Cairn first-ingestion recovery plan; no model calls or writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

JOB = "social-fixed-r3"
EXECUTION = "eda565fd801c47ddb4e1be07c2773990"
PROVIDER = "cairn_memory"
SOURCE_SHA256 = "7843293e36224af76e99e613aa91c63d8c56b95cd33a7fae15e15a10fbfc91fc"
STORE_SHA256 = "316c780569b5bd8671873dcf836aefcfce8a91152ecd9400dabbd8b4f71fd866"
ATTEMPTS = [15854, 15862, 15871, 15887, 15916, 15934, 15951, 15959, 15967, 15984,
            16039, 16076, 16100]
STEP_ATTEMPTS = [15916, 15951, 15967]
INGESTION_ATTEMPTS = [16076, 16100]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_text(event):
    choices = event["response"]["choices"]
    assert event["status"] == "success" and len(choices) == 1 and choices[0]["finish_reason"] == "stop"
    assert isinstance(event["usage"], dict) and event["usage"]["total_tokens"] > 0
    return choices[0]["message"]["content"]


def response(event):
    return json.loads(response_text(event))


def prompt_text(event):
    messages = event["request"]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list) and len(content) == 1 and content[0]["type"] == "text"
    return content[0]["text"]


def credit_prompt(delivered, summaries, success):
    """Exact provider prompt; matching the paid receipt proves restored RAM inputs."""
    memories = [f"- mem_id={mid} | delivered_at_step={info.get('step')}\n"
                f"  Content: {info.get('content', '')[:260]}\n"
                f"  Trigger: {info.get('trigger', '')[:180]}" for mid, info in delivered.items()]
    steps = [f"Step {item.get('step')}: {str(item.get('summary', ''))[:200]}" for item in summaries]
    return (
        "Audit whether the agent actually USED the delivered memories below.\n\n"
        "**Memories delivered during the run:**\n" + "\n".join(memories)
        + "\n\n**Step summaries recorded during the run:**\n"
        + ("\n".join(steps) if steps else "(no step summaries recorded)")
        + "\n\n" + f"**Outcome:** {'SUCCESS' if success else 'FAILURE'}\n\n"
        "**For each memory, output:**\n"
        '- used: true only if a subsequent step shows the agent acting in a way '
        'consistent with this memory (taking the recommended action, applying the '
        'technique, respecting the constraint). Delivery alone is NOT usage.\n'
        '- confidence: your confidence in the used assessment (0.0-1.0).\n'
        '- evidence: short reason (<=120 chars).\n\n'
        "**JSON output only:**\n"
        '{\n'
        '  "credits": [{"mem_id": "...", "used": true, "confidence": 0.8, "evidence": "..."}]\n'
        '}\n'
    )


def plan_cairn_ingestion(root, record, events):
    root = Path(root).resolve()
    job = root / "jobs" / JOB
    assert record["task_execution_id"] == EXECUTION and record["provider"] == PROVIDER
    assert record["status"] == "ingestion_failed" and record["memory_ingestion_started"]
    assert not record["memory_ingestion_complete"] and record["quality_score"] == 92.0
    assert record["grade"]["quality_score"] == 92.0 and len(record["answer"]) == 7582
    assert record["question"] == record["enhanced_question"]
    source = job / "workspace/EvolveLab/providers/cairn_memory_provider.py"
    store_path = job / "workspace/storage/cairn_memory/memory.json"
    assert sha(source) == record["runtime_artifact"]["code_sha256"] == SOURCE_SHA256
    assert sha(store_path) == STORE_SHA256
    assert [p.name for p in store_path.parent.iterdir() if p.is_file()] == ["memory.json"]
    current = sorted((e for e in events if e.get("task_execution_id") == EXECUTION), key=lambda e: e["attempt"])
    assert [e["attempt"] for e in current] == ATTEMPTS
    by_id = {event["attempt"]: event for event in current}
    assert by_id[15854]["status"] == "error" and by_id[15862]["recovered_attempts"] == [15854]
    assert by_id[15862]["recovery_reason"] == "CredentialQuotaExhausted"
    assert all(e["status"] == "success" for e in current[1:])
    assert [e["attempt"] for e in current if e["role"] == "memory_ingestion"] == INGESTION_ATTEMPTS
    assert set(response(by_id[16076])) == {"credits"}
    assert set(response(by_id[16100])) == {"techniques", "constraints", "failure_modes"}
    judge = by_id[16039]
    assert judge["role"] == "judge" and store_path.stat().st_mtime < judge["finished_unix"]
    # Only initialize() and the final successful take_in_memory() save this DB;
    # the first consolidation raises before that final commit. No rollback.
    stored = json.loads(store_path.read_text(encoding="utf-8"))
    assert len(stored["memories"]) == 9 and stored["meta"]["total_ingestions"] == 0
    assert all(m["is_coldstart"] and m["attempts"] == m["retrieval_trials"] == m["attributed_successes"] == 0
               and isinstance(m["token_set"], list) for m in stored["memories"])
    db = deepcopy(stored)
    selected = response(by_id[15871])["selected_mem_ids"]
    assert len(selected) == len(set(selected)) == 6
    memories = {m["mem_id"]: m for m in db["memories"]}
    assert set(selected) <= set(memories)
    task_id = "cairn_task_1"  # new provider; __init__=0, first BEGIN resets once
    delivered = {}
    for mid in selected:
        mem = memories[mid]
        mem["retrieval_trials"] += 1
        delivered[mid] = {"step": 1, "content": mem["content"][:300],
                          "trigger": mem["trigger"][:200], "task_id": task_id}
    summaries = [{"step": step, "summary": str(response(by_id[number])["step_summary"]).strip()}
                 for step, number in enumerate(STEP_ATTEMPTS, 2)]
    assert credit_prompt(delivered, summaries, True) == prompt_text(by_id[16076])
    query = record["question"]
    task_context = {"task_id": task_id, "query": query,
                    "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    "delivered_memories": delivered, "step_summaries": summaries}
    # Only fields consumed by ingestion are reconstructed. No claim is made
    # about discarded timestamps or working items unused by take_in_memory.
    replays = [{"attempt": number,
                "receipt_path": str(root / "budget/attempts" / f"{number:06d}.json"),
                "receipt_sha256": sha(root / "budget/attempts" / f"{number:06d}.json")}
               for number in INGESTION_ATTEMPTS]
    context = {"kind": "cairn_preingestion", "query": query, "db": db,
               "task_context": task_context, "task_snapshots": {}, "task_id_counter": 1,
               "paid_ingestion_replays": replays}
    # The helper backs up and hashes the original task record separately; its
    # active path changes when recovery authorization is written.
    paths = [source, store_path]
    paths.extend(root / "budget/attempts" / f"{e['attempt']:06d}.json" for e in current)
    return {"method": "restore_cairn_volatile_preingestion_then_retry",
            "reason": "Only initial nine cold-start memories reached disk; restore BEGIN retrieval trials and exact credit inputs before completing the saved episode once.",
            "provider_sha256": SOURCE_SHA256, "failed_attempts": [],
            "judge_finished_unix": judge["finished_unix"], "runtime_context": context,
            "storage_rollbacks": [], "pinned_files": {str(p.relative_to(root)): sha(p) for p in paths},
            "saved_answer_and_grade_reused": True, "paid_ingestion_attempts_preserved": INGESTION_ATTEMPTS,
            "adapter_compatibility": "Hash-pinned Cairn _jaccard instance wrapper converts both operands to sets; original provider bytes remain unchanged.",
            "restoration_scope": "Only RAM fields consumed by take_in_memory; _by_id must reference db memories directly.",
            "native_completion_required": True}
