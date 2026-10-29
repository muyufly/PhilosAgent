"""Restore interrupted comparisons and reserve, rather than erase, unknown cost."""

from copy import deepcopy
from pathlib import Path
import shutil

from .ablation_runtime import digest
from .cli import PROJECT, FLASH, source_hash
from .hss_budget import HSSBudget, reconcile_settled_unknown_attempts
from .hss_experiment import now, read_json, write_json, process_owned
from .hss_finalist_recovery import finalist_recovery_hashes, restore_incomplete_finalists
from .hss_enum_collision_recovery import enum_collision_input_hashes, repair_enum_collision
from .hss_repair_r3 import task_usage
from .hss_tasks_runtime import classify_recovery_record


REVISION = "repair-r7"
MAINTENANCE = "maintenance_repair_r7_20261029"
CONTROLLER_POLICY = {"connection_recovery": {
    "max_additional_attempts": 2, "reserve_unknown_usage": True, "backoff_seconds": [1, 3]}}


def active_records(directory):
    for round_dir in sorted((directory / "evolution").glob("round_[0-9][0-9]")):
        for pattern in ("base_logs/[0-9]*.json", "eval_batch/*/[0-9]*.json", "eval_finalists/*/[0-9]*.json"):
            for path in sorted(round_dir.glob(pattern)):
                if path.stem.isdigit() and "task_execution_id" in read_json(path):
                    yield path


def authorize_task(record, attempts):
    """Keep task identity, paid work and costs; authorize only the missing phase."""
    result = deepcopy(record)
    phase = classify_recovery_record(result)
    if phase not in ("actor", "judge", "ingestion", "integrity"):
        raise ValueError("Ambiguous memory mutation or missing trajectory requires separate restoration")
    usage = task_usage(result, attempts)
    usage["effective_tokens"] = usage["total_tokens"] + usage["reserved_unknown_tokens"]
    result.update(api_usage=usage, total_model_tokens=usage["total_tokens"],
                  confirmed_model_tokens=usage["total_tokens"], effective_model_tokens=usage["effective_tokens"])
    result["recovery_authorized"] = {"revision": REVISION,
        "reason": "User authorized recovery after connection failure and incomplete comparisons",
        "allow_actor_retry": phase == "actor" and not result.get("memory_ingestion_complete"),
        "known_failed_attempts": usage["unrecovered_failed_attempt_ids"],
        "reserved_unknown_attempts": usage["reserved_unknown_attempt_ids"], "next_phase": phase}
    return result


def _stopped(root):
    if process_owned(read_json(root / "supervisor.json")):
        raise ValueError("Supervisor is still alive")
    for path in (root / "jobs").glob("*/job_status.json"):
        state = read_json(path)
        if process_owned(state.get("worker_source") or state):
            raise ValueError("An experiment worker is still alive")


def amend(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("adapter_revision") != REVISION
            or manifest["adapter_sha256"] != source_hash(PROJECT / "philoagent")
            or manifest["upstream_sha256"] != source_hash(FLASH)):
        raise ValueError("Publish and verify the r7 controller snapshot before recovery")
    _stopped(root)
    recovery = root / "revisions" / REVISION / "recovery"
    receipt_path = recovery / "receipt.json"
    if receipt_path.exists():
        return read_json(receipt_path)
    if recovery.exists():
        raise ValueError("Partial r7 recovery exists; inspect its archived phase before proceeding")
    budget = HSSBudget(root / "budget", max_calls=manifest["protocol"]["max_calls"],
        max_tokens=manifest["protocol"]["max_tokens"], max_inflight=6)
    state = budget.summary()
    if state["pending_count"] or state["blocked_reason"] != "unknown_usage":
        raise ValueError("Recovery expects the settled usage-unknown incident")
    unknown = []
    for path in sorted((root / "budget/attempts").glob("*.json")):
        event = read_json(path)
        if not isinstance(event.get("usage"), dict) and not event.get("reserved_unknown_tokens"):
            unknown.append((path, event))
    if (len(unknown) != 1 or unknown[0][1]["attempt"] != 3835
            or unknown[0][1].get("error_type") != "APIConnectionError"
            or unknown[0][1].get("job_id") != "humanities-native-r1"):
        raise ValueError("Incident differs from the explicitly audited request 3835")
    unknown_path = unknown[0][0]
    finalized_pins = {job: finalist_recovery_hashes(root, job)
                      for job in ("humanities-native-r1", "humanities-full-r1")}
    enum_pins = enum_collision_input_hashes(root, "business-fixed-r1")
    recovery.mkdir(parents=True)
    for name in ("manifest.json", "supervisor.json", "report.json"):
        shutil.copy2(root / name, recovery / (Path(name).stem + ".before.json"))
    write_json(recovery / "budget.before.json", state)
    shutil.copy2(unknown_path, recovery / "attempt-003835.before.json")
    write_json(recovery / "plan.json", {"recorded_utc": now(), "finalist_pins": finalized_pins,
        "enum_pins": enum_pins, "unknown_attempt_sha256": digest(unknown_path.read_bytes())})
    with budget._ledger_lock():
        current = budget._read()
        if current != {key: value for key, value in state.items() if key in current}:
            raise ValueError("Ledger changed during offline repair planning")
        current["blocked_reason"] = MAINTENANCE
        current["maintenance"] = {"reason": "User authorized unknown-cost and incomplete-comparison recovery",
            "revision": REVISION, "authorized_by_user": "进行修复并继续推进实验", "started_utc": now()}
        budget._save(current)
    reservation = reconcile_settled_unknown_attempts(root / "budget",
        "User authorized conservative reservation of connection-failed request 3835; actual usage remains unknown",
        expected_attempts=[3835], expected_hashes={3835: digest(unknown_path.read_bytes())})
    finalist_receipts = [restore_incomplete_finalists(root, job, pins)
                         for job, pins in finalized_pins.items()]
    enum_receipt = repair_enum_collision(root, "business-fixed-r1", expected_hashes=enum_pins)
    attempts = [read_json(path) for path in sorted((root / "budget/attempts").glob("*.json"))]
    plans = []
    for directory in sorted((root / "jobs").iterdir()):
        for path in active_records(directory):
            record = read_json(path)
            if record.get("status") != "completed":
                plans.append((path, authorize_task(record, attempts)))
    # Validate every task before altering any recovery authorization.
    for path, record in plans:
        archive = recovery / "task_records.before" / path.relative_to(root)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive)
        write_json(path, record)
    statuses = []
    for path in sorted((root / "jobs").glob("*/job_status.json")):
        previous = read_json(path)
        if previous.get("status") in ("failed", "interrupted", "queued"):
            write_json(recovery / "job_status.before" / path.parent.name / path.name, previous)
            write_json(path, {"status": "queued", "revision": REVISION,
                             "reason": "Resume audited missing phases and complete all scheduled comparisons"})
            statuses.append(path.parent.name)
    manifest["controller_policy"] = deepcopy(CONTROLLER_POLICY)
    manifest.setdefault("amendments", []).append({"revision": REVISION, "kind": "audited_incident_recovery",
        "recorded_utc": now(), "user_message": "进行修复并继续推进实验", "protocol_unchanged": True,
        "evidence": str(receipt_path)})
    write_json(root / "manifest.json", manifest)
    receipt = {"revision": REVISION, "recorded_utc": now(), "status": "prepared_under_maintenance",
        "reservation": reservation, "finalist_recovery": finalist_receipts, "enum_recovery": enum_receipt,
        "authorized_tasks": [{"path": str(path), "phase": value["recovery_authorized"]["next_phase"]}
                             for path, value in plans], "queued_job_statuses": statuses,
        "controller_policy": CONTROLLER_POLICY, "protocol_unchanged": True,
        "task_identity_unchanged": True, "paid_candidates_regenerated": 0}
    write_json(receipt_path, receipt)
    return receipt


def release(root):
    root = Path(root).resolve(strict=True)
    _stopped(root)
    manifest = read_json(root / "manifest.json")
    recovery = root / "revisions" / REVISION / "recovery"
    receipt = read_json(recovery / "receipt.json")
    if (manifest.get("adapter_revision") != REVISION
            or source_hash(PROJECT / "philoagent") != manifest["adapter_sha256"]
            or source_hash(FLASH) != manifest["upstream_sha256"]
            or manifest.get("controller_policy") != CONTROLLER_POLICY):
        raise ValueError("Recovery source or controller policy differs from its freeze")
    budget = HSSBudget(root / "budget", max_calls=manifest["protocol"]["max_calls"],
        max_tokens=manifest["protocol"]["max_tokens"], max_inflight=6)
    with budget._ledger_lock():
        state = budget._read()
        if (state["pending"] or state["blocked_reason"] != MAINTENANCE
                or state["unknown_usage_attempts"] != len(state.get("reserved_unknown_attempts", []))):
            raise ValueError("Unsettled or unreserved requests prevent resumption")
        state["blocked_reason"] = None
        state["maintenance"]["released_utc"] = now()
        budget._save(state)
    receipt.update(status="released_for_native_resumption", released_utc=now())
    write_json(recovery / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    result = amend(args.root)
    if args.release:
        result = release(args.root)
    print({"revision": result["revision"], "status": result["status"],
           "authorized_tasks": len(result["authorized_tasks"]), "protocol_unchanged": True})
