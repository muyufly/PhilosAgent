"""Repair native compatibility and resume an externally interrupted matrix."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

from .ablation_runtime import digest
from .cli import PROJECT, FLASH, source_hash
from .hss_budget import HSSBudget
from .hss_experiment import read_json, write_json, process_owned, now
from .hss_repair import repaired_protocol as r2_protocol, _records


REVISION = "repair-r3"
MAINTENANCE = "maintenance_repair_20261029"


def repaired_protocol(protocol):
    result = r2_protocol(protocol)
    result.update(revision=REVISION, actor_recovery_max_attempts=2,
                  generated_code_policy="native_local", tool_decoding="native_json_openai_dsml",
                  process_identity="pid_birth_executable", background_launch="windows_wmi_detached")
    return result


def task_usage(record, attempts):
    identifiers = set(record.get("task_execution_ids", [])) | {record["task_execution_id"]}
    events = [event for event in attempts if event.get("task_execution_id") in identifiers]
    total = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    reserved = [event for event in events if event.get("reserved_unknown_tokens", 0) > 0]
    unknown = [event for event in events if not isinstance(event.get("usage"), dict)]
    if any(event not in reserved for event in unknown):
        raise ValueError("Unreserved unknown task usage cannot be resumed")
    for event in events:
        usage = event.get("usage")
        if usage is not None:
            if any(type(usage.get(key)) is not int for key in total):
                raise ValueError("Invalid prior task usage")
            for key in total:
                total[key] += usage[key]
    recovered = {number for event in events if event.get("status") == "success"
                 for number in event.get("recovered_attempts", [])}
    recovered.update(number for receipt in record.get("phase_recovery_receipts", [])
                     for number in receipt.get("recovered_attempts", []))
    failures = [event for event in events if event.get("status") != "success"]
    unresolved = [event["attempt"] for event in failures if event["attempt"] not in recovered]
    return {**total, "request_count": len(events), "failed_request_count": len(failures),
            "usage_complete": not unknown, "usage_accounted": True,
            "unknown_usage_attempts": len(unknown),
            "reserved_unknown_tokens": sum(event["reserved_unknown_tokens"] for event in reserved),
            "reserved_unknown_attempt_ids": [event["attempt"] for event in reserved],
            "unrecovered_failed_attempt_ids": unresolved,
            "unrecovered_failed_request_count": len(unresolved)}


def amend(root, *, revision_name=REVISION, maintenance_reason=MAINTENANCE,
          protocol_builder=repaired_protocol):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    revision = root / "revisions" / revision_name
    if manifest["protocol"].get("revision") == revision_name:
        if manifest["adapter_sha256"] != source_hash(PROJECT / "philoagent"):
            raise ValueError("Repaired source changed after freeze")
        return read_json(revision / "amendment.json")
    if revision.exists():
        raise ValueError("Partial revision exists; inspect before proceeding")
    budget = HSSBudget(root / "budget", max_inflight=6)
    state = budget.summary()
    if state["blocked_reason"] != maintenance_reason or state["pending_count"]:
        raise ValueError("Migration requires maintenance and no unsettled requests")
    if state["unknown_usage_attempts"] != len(state.get("reserved_unknown_attempts", [])):
        raise ValueError("Unknown unreserved budget usage blocks migration")
    if process_owned(read_json(root / "supervisor.json")):
        raise ValueError("Supervisor is still active")
    if manifest["upstream_sha256"] != source_hash(FLASH):
        raise ValueError("Native source changed unexpectedly")
    attempts = [read_json(path) for path in sorted((root / "budget/attempts").glob("*.json"))]
    protocol = protocol_builder(manifest["protocol"])
    plans = []
    statuses = []
    for job in (root / "jobs").iterdir():
        state_path = job / "job_status.json"
        if state_path.exists() and read_json(state_path).get("status") == "running" and process_owned(read_json(state_path)):
            raise ValueError("A worker is still active")
        rows = {}
        if (job / "learning.jsonl").exists():
            rows = {row["task_id"]: row for row in
                    (json.loads(line) for line in (job / "learning.jsonl").read_text(encoding="utf8").splitlines() if line.strip())}
        for path in _records(job):
            old = read_json(path)
            record = deepcopy(old)
            if (record.get("memory_ingestion_started") or record.get("status") == "ingestion_failed") and not record.get("memory_ingestion_complete"):
                raise ValueError("Ambiguous native ingestion needs storage restoration")
            identity_fields = {"task": rows[record["task_id"]],
                "index": record["global_task_index"], "provider": record["provider"],
                "artifact": record["runtime_artifact"], "stage": record["stage"],
                "dataset": "writingbench", "model": protocol["model"]}
            if digest({**identity_fields, "protocol": manifest["protocol"]}) != record["task_identity"]:
                raise ValueError("Saved task identity differs from the frozen protocol")
            record["task_identity"] = digest({**identity_fields, "protocol": protocol})
            usage = task_usage(record, attempts)
            if record["status"] != "completed":
                record["api_usage"] = usage
                record["total_model_tokens"] = usage["total_tokens"]
                record["recovery_authorized"] = {"revision": revision_name,
                    "reason": "User authorized repair of native decoder and externally terminated workers",
                    "allow_actor_retry": not bool(record.get("answer", "").strip()),
                    "known_failed_attempts": usage["unrecovered_failed_attempt_ids"],
                    "reserved_unknown_attempts": usage["reserved_unknown_attempt_ids"]}
            record["protocol_migration"] = {"revision": revision_name,
                "previous_task_identity": old["task_identity"],
                "original_record": str(revision / "task_records.before" / path.relative_to(root))}
            plans.append((path, record))
        if state_path.exists():
            statuses.append((state_path, {"status": "queued", "revision": revision_name,
                                          "reason": "Retry after audited technical repair"}))
    revision.mkdir(parents=True)
    write_json(revision / "manifest.before.json", manifest)
    write_json(revision / "budget.before.json", state)
    shutil.copy2(root / "supervisor.json", revision / "supervisor.before.json")
    for path, record in plans:
        backup = revision / "task_records.before" / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        write_json(path, record)
    for path, status in statuses:
        backup = revision / "job_status.before" / path.parent.name / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        write_json(path, status)
    manifest.update(protocol=protocol, adapter_sha256=source_hash(PROJECT / "philoagent"))
    manifest.setdefault("amendments", []).append({"revision": revision_name,
        "user_message": "实验是否正常运行，若无则修复", "recorded_utc": now(),
        "evidence": str(revision / "amendment.json")})
    shutil.copytree(PROJECT / "philoagent", revision / "code_snapshot.after/philoagent",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    evidence = {"revision": revision_name, "recorded_utc": now(), "user_message": "实验是否正常运行，若无则修复",
        "migrated_tasks": [{"path": str(path), "status": record["status"]} for path, record in plans],
        "same_budget_ledger": str(root / "budget/budget.json"), "budget_before": state,
        "protocol": protocol, "adapter_sha256": manifest["adapter_sha256"], "status": "ready_for_resume"}
    write_json(root / "manifest.json", manifest)
    write_json(revision / "amendment.json", evidence)
    return evidence


def release(root, *, revision_name=REVISION, maintenance_reason=MAINTENANCE):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if manifest["protocol"].get("revision") != revision_name or manifest["adapter_sha256"] != source_hash(PROJECT / "philoagent"):
        raise ValueError("Repaired sources do not match the r3 freeze")
    budget = HSSBudget(root / "budget", max_inflight=6)
    with budget._ledger_lock():
        state = budget._read()
        if (state["blocked_reason"] != maintenance_reason or state["pending"]
                or state["unknown_usage_attempts"] != len(state.get("reserved_unknown_attempts", []))):
            raise ValueError("Unsettled or unreserved usage prevents release")
        state["blocked_reason"] = None
        state["maintenance"].update(released_utc=now(), revision=revision_name)
        budget._save(state)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    result = amend(args.root)
    if args.release:
        release(args.root)
    print(json.dumps({"revision": REVISION, "tasks": len(result["migrated_tasks"])}, ensure_ascii=False))
