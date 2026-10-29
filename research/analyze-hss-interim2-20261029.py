"""Capture active formal results and settled usage without changing the run."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from philoagent.hss_experiment import process_owned

ROOT = PROJECT / "runs/hss-business-formal-20261029-r01"
OUTPUT = PROJECT / "research/hss-business-interim2-20261029.snapshot.json"


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(.02)


def utc():
    return datetime.now(timezone.utc).isoformat()


def capture():
    start = utc()
    manifest = read(ROOT / "manifest.json")
    # Settled receipts cannot be completed after this accounting boundary.
    budget = read(ROOT / "budget/budget.json")
    pending = {int(number) for number in budget["pending"]}
    attempt_keys = ("attempt", "status", "role", "job_id", "stage", "provider", "task_id",
        "task_execution_id", "round_dir", "usage", "reserved_unknown_tokens", "error_type",
        "started_utc", "finished_utc", "duration_seconds", "response_finish_reasons",
        "recovery_index", "retry_of_attempt", "recovery_reason", "transport_recovered",
        "recovered_attempts", "max_tokens", "orientation_items_sha256")
    attempts = []
    for number in range(1, budget["attempt_count"] + 1):
        if number in pending:
            continue
        event = read(ROOT / "budget/attempts" / f"{number:06d}.json")
        attempts.append({key: event[key] for key in attempt_keys if key in event})
    confirmed = sum(x.get("usage", {}).get("total_tokens", 0) for x in attempts if isinstance(x.get("usage"), dict))
    assert confirmed == budget["actual_tokens"], (confirmed, budget["actual_tokens"])
    assert sum(x.get("reserved_unknown_tokens", 0) for x in attempts) == budget["reserved_unknown_tokens"]
    rows, rounds, jobs, orientations, generated = [], [], [], [], []
    keys = ("task_id", "question", "domain", "provider", "stage", "global_task_index", "item_index",
        "task_identity", "task_execution_id", "task_execution_ids", "status", "answer", "grade",
        "judge_output", "judge_normalization", "quality_score", "score", "judgement",
        "memory_ingestion_complete", "memory_ingestion_started", "api_usage", "total_model_tokens",
        "effective_model_tokens", "metrics", "error_type", "actor_recovery_attempts",
        "actor_output_issue", "response_truncated", "recovery_history", "phase_recovery_receipts")
    for job in manifest["jobs"]:
        directory = ROOT / "jobs" / job["job_id"]
        status_path = directory / "job_status.json"
        status = read(status_path) if status_path.exists() else {"status": "queued"}
        jobs.append({**job, "job_status": status, "owned_process": process_owned(status) if status.get("pid") else False,
                     "complete_result": read(directory / "result.json") if (directory / "result.json").exists() else None})
        for rdir in sorted((directory / "evolution").glob("round_[0-9][0-9]")):
            summary = rdir / "round_summary.json"
            if summary.exists():
                rounds.append({"job_id": job["job_id"], "path": str(summary.relative_to(ROOT)), "summary": read(summary)})
            for phase, pattern in (("base", "base_logs/[0-9]*.json"), ("batch", "eval_batch/*/[0-9]*.json"),
                                   ("finalist", "eval_finalists/*/[0-9]*.json")):
                for path in sorted(rdir.glob(pattern)):
                    raw = path.read_bytes()
                    record = json.loads(raw)
                    if not path.stem.isdigit() or "task_execution_id" not in record:
                        continue
                    artifact = record.get("runtime_artifact", {})
                    rows.append({"job_id": job["job_id"], "group": job["group"], "repetition": job["repetition"],
                        "round": int(rdir.name.split("_")[1]), "phase": phase,
                        "path": str(path.relative_to(ROOT)), "record_sha256": hashlib.sha256(raw).hexdigest(),
                        "record": {k: record[k] for k in keys if k in record},
                        "runtime_artifact": {k: artifact[k] for k in ("provider", "code_sha256", "orientation_items_sha256", "orientation_snapshot") if k in artifact}})
            for path in sorted(rdir.glob("generated_system_[0-9]*.json")):
                config = read(path)
                generated.append({"job_id": job["job_id"], "round": int(rdir.name.split("_")[1]),
                    "path": str(path.relative_to(ROOT)),
                    "config": {k: v for k, v in config.items() if "code" not in k.lower() and k != "_raw_llm_output"}})
        for path in sorted((directory / "terminal/task_logs").glob("*/[0-9]*.json")):
            record = read(path)
            if "task_execution_id" in record:
                rows.append({"job_id": job["job_id"], "group": job["group"], "repetition": job["repetition"],
                    "round": None, "phase": "heldout", "path": str(path.relative_to(ROOT)),
                    "record": {k: record[k] for k in keys if k in record}})
        for path in sorted((directory / "lineage").glob("*/orientation.json")):
            orientations.append({"job_id": job["job_id"], "path": str(path.relative_to(ROOT)), "snapshot": read(path)})
    queue = read(ROOT / "recovery_supervisor.r9.json")
    snapshot = {"captured_start_utc": start, "captured_end_utc": utc(), "root": str(ROOT),
        "read_only_run_capture": True, "manifest": manifest, "budget": budget,
        "budget_receipts_confirmed_tokens_reconciled": True,
        "attempt_min": attempts, "records": rows, "round_summaries": rounds, "jobs": jobs,
        "orientations": orientations, "generated_candidates": generated, "queue": queue,
        "sampling_note": "Active task records only, excluding revision/backups/recovery archives. Tasks observed during capture interval; budget excludes every request pending at its earlier accounting boundary. Multiple candidate evaluations are repeated observations, not independent tasks."}
    OUTPUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "captured_start_utc": start, "captured_end_utc": snapshot["captured_end_utc"],
        "records": len(rows), "record_statuses": dict(Counter(x["record"]["status"] for x in rows)),
        "round_summaries": len(rounds), "settled_attempts": len(attempts), "confirmed_tokens": confirmed,
        "bytes": OUTPUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    capture()
