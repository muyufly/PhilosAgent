"""Auditable, offline migration of the interrupted HSS run to repair-r2.

The shared ledger must be drained before migration. Original sources, manifests,
answers and usage are preserved; the same whole-experiment budget is resumed.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil

from .ablation_runtime import digest
from .cli import PROJECT, FLASH, source_hash
from .hss_budget import HSSBudget
from .hss_experiment import now, process_alive, read_json, write_json


REVISION = "repair-r2"
MAINTENANCE = "maintenance_repair_20261029"


def repaired_protocol(protocol):
    result = deepcopy(protocol)
    result.update(
        revision=REVISION, actor_max_output_tokens=32768,
        analysis_max_output_tokens=16384, judge_max_output_tokens=16384,
        resume_incomplete_tasks=True, actor_recovery_max_attempts=1,
        judge_recovery_max_attempts=1,
        transport_recovery={"max_additional_attempts": 2,
                            "min_output_tokens": 32768, "max_output_tokens": 60000},
        role_min_output_tokens={"actor": 32768, "analysis": 16384,
                                "judge": 16384, "trajectory_summary": 8192,
                                "memory_retrieval": 16384, "memory_ingestion": 16384},
    )
    return result


def _records(job):
    # Native comparison logs are nested under eval_batch/<provider>, rather
    # than base_logs. Recovery archives use non-numeric names and are excluded.
    return sorted(path for path in (job / "evolution").glob("round_*/**/[0-9]*.json")
                  if path.stem.isdigit() and "task_execution_id" in read_json(path))


def _truncated_actor(record, attempts):
    events = [event for event in attempts
              if event.get("task_execution_id") == record.get("task_execution_id")
              and event.get("role") == "actor"]
    if not events:
        return False
    last = max(events, key=lambda item: item["attempt"])
    return any(choice.get("finish_reason") == "length"
               for choice in (last.get("response") or {}).get("choices", []))


def _prepare_record(record, task_rows, protocol, attempts, backup):
    record = deepcopy(record)
    old_identity = record["task_identity"]
    record["task_identity"] = digest({
        "task": task_rows[record["task_id"]], "index": record["global_task_index"],
        "provider": record["provider"], "artifact": record["runtime_artifact"],
        "stage": record["stage"], "dataset": "writingbench",
        "protocol": protocol, "model": protocol["model"],
    })
    truncated = _truncated_actor(record, attempts)
    if truncated:
        record.update(actor_output_issue="audited_finish_reason_length", response_truncated=True)
    if record.get("status") != "completed" or truncated:
        record["recovery_authorized"] = {"revision": REVISION,
            "reason": "User authorized saved-phase recovery and bounded technical retries",
            "allow_actor_retry": truncated or not bool(record.get("answer", "").strip())}
        failed = [event for event in attempts
                  if event.get("task_execution_id") == record.get("task_execution_id")
                  and event.get("status") == "error"]
        if any(not isinstance(event.get("usage"), dict)
               or any(type(event["usage"].get(key)) is not int for key in
                      ("prompt_tokens", "completion_tokens", "total_tokens")) for event in failed):
            raise ValueError("Unknown prior task usage cannot be authorized for recovery")
        identifiers = [event["attempt"] for event in failed]
        record.setdefault("api_usage", {}).update(
            unrecovered_failed_attempt_ids=identifiers, unrecovered_failed_request_count=len(identifiers))
        record["recovery_authorized"]["known_failed_attempts"] = identifiers
    record["protocol_migration"] = {"revision": REVISION, "previous_task_identity": old_identity,
                                     "original_record": str(backup)}
    return record, truncated


def amend(root, *, unfreeze=False):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if manifest["protocol"].get("revision") == REVISION:
        if manifest["adapter_sha256"] != source_hash(PROJECT / "philoagent"):
            raise ValueError("Repaired source changed after its freeze")
        return read_json(root / "revisions" / REVISION / "amendment.json")
    budget = HSSBudget(root / "budget", max_calls=manifest["protocol"]["max_calls"],
                       max_tokens=manifest["protocol"]["max_tokens"], max_inflight=6)
    state = budget.summary()
    if state["blocked_reason"] != MAINTENANCE or state["pending_count"] or state["unknown_usage_attempts"]:
        raise ValueError("Repair requires maintenance mode and zero pending requests")
    if manifest["upstream_sha256"] != source_hash(FLASH):
        raise ValueError("Native source unexpectedly changed during repair")
    supervisor = read_json(root / "supervisor.json")
    if process_alive(supervisor.get("pid")):
        raise ValueError("Supervisor must exit before repair")
    for status in (root / "jobs").glob("*/job_status.json"):
        if process_alive(read_json(status).get("pid")):
            raise ValueError("Worker must exit before repair")
    revision = root / "revisions" / REVISION
    if revision.exists():
        raise ValueError("Partial amendment exists; inspect it before further writes")
    attempts = [read_json(path) for path in sorted((root / "budget/attempts").glob("*.json"))]
    protocol = repaired_protocol(manifest["protocol"])
    plans = []
    for spec in manifest["jobs"]:
        job = root / "jobs" / spec["job_id"]
        if not job.exists():
            continue
        records = [(path, read_json(path)) for path in _records(job)]
        reset = any((_truncated_actor(record, attempts) and record.get("memory_ingestion_complete"))
                    or record.get("memory_ingestion_started") and not record.get("memory_ingestion_complete")
                    or record.get("status") == "ingestion_failed"
                    for _, record in records)
        destination = revision / "restarted_jobs" / job.name
        if not job.resolve().is_relative_to(root / "jobs") or not destination.resolve().is_relative_to(revision):
            raise ValueError("Repair archive escaped the experiment directory")
        prepared = []
        if not reset:
            task_rows = {row["task_id"]: row for row in
                         (json.loads(line) for line in (job / "learning.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
            for path, record in records:
                backup = revision / "task_records.before" / path.relative_to(root)
                transformed, truncated = _prepare_record(record, task_rows, protocol, attempts, backup)
                prepared.append((path, transformed, truncated, backup))
        plans.append((spec, job, prepared, reset))
    revision.mkdir(parents=True)
    write_json(revision / "manifest.before.json", manifest)
    write_json(revision / "budget.before.json", state)
    shutil.copytree(root / "code_snapshot", revision / "code_snapshot.before")
    changed, restarted = [], []
    for spec, job, records, reset in plans:
        if reset:
            destination = revision / "restarted_jobs" / job.name
            # Both resolved targets are verified before a recursive move.
            if not job.resolve().is_relative_to(root / "jobs") or not destination.resolve().is_relative_to(revision):
                raise ValueError("Repair archive escaped the experiment directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(job), str(destination))
            restarted.append({"job_id": job.name,
                              "reason": "Truncated work was already ingested or ingestion was interrupted; restore a consistent native lineage",
                              "archive": str(destination)})
            continue
        for path, record, truncated, backup in records:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            write_json(path, record)
            changed.append({"job_id": job.name, "path": str(path), "status": record.get("status"),
                            "truncated": truncated, "actor_retry": record.get("recovery_authorized", {}).get("allow_actor_retry", False)})
    manifest["protocol"] = protocol
    manifest["adapter_sha256"] = source_hash(PROJECT / "philoagent")
    manifest.setdefault("amendments", []).append({"revision": REVISION, "user_message": "尝试修复。",
                                                 "recorded_utc": now(), "evidence": str(revision / "amendment.json")})
    # Keep the original code snapshot intact; repaired workers pin this revision.
    shutil.copytree(PROJECT / "philoagent", revision / "code_snapshot.after/philoagent",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_json(root / "manifest.json", manifest)
    evidence = {"revision": REVISION, "recorded_utc": now(), "user_message": "尝试修复。",
                "recovered_task_records": changed, "restarted_lineages": restarted,
                "budget_before": {key: state[key] for key in ("attempt_count", "actual_tokens", "unknown_usage_attempts")},
                "same_budget_ledger": str(root / "budget/budget.json"),
                "adapter_sha256": manifest["adapter_sha256"], "protocol": protocol,
                "status": "ready_for_resume", "prior_results_are_not_final_ablation_evidence": True}
    write_json(revision / "amendment.json", evidence)
    if unfreeze:
        release(root)
    return evidence


def release(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if manifest["protocol"].get("revision") != REVISION:
        raise ValueError("No audited repair amendment is installed")
    if manifest["adapter_sha256"] != source_hash(PROJECT / "philoagent"):
        raise ValueError("Repaired source no longer matches its freeze")
    budget = HSSBudget(root / "budget", max_calls=60000, max_tokens=350000000, max_inflight=6)
    with budget._ledger_lock():
        state = budget._read()
        if state["blocked_reason"] != MAINTENANCE or state["pending"] or state["unknown_usage_attempts"]:
            raise ValueError("Maintenance release refused: unsettled or unknown usage")
        state["blocked_reason"] = None
        state["maintenance"]["released_utc"] = now()
        state["maintenance"]["revision"] = REVISION
        budget._save(state)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    result = amend(args.root, unfreeze=args.release)
    print(json.dumps({"revision": result["revision"], "tasks": len(result["recovered_task_records"]),
                      "restarted_lineages": result["restarted_lineages"], "status": result["status"]}, ensure_ascii=False))
