"""Authorize reuse of a paid, now-parseable judge response for one stopped job."""

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

from .ablation_runtime import digest, items_digest
from .hss_evaluation import normalize_grade_output, validate_grade
from .hss_experiment import now, process_owned, read_json, write_json
from .hss_repair_r3 import task_usage
from .hss_tasks_runtime import classify_recovery_record
from .orientation import load


def _target(root, job_id, relative_task_path):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if not any(job["job_id"] == job_id for job in manifest["jobs"]):
        raise ValueError("Unknown experiment job")
    directory = (root / "jobs" / job_id).resolve(strict=True)
    relative = Path(relative_task_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Judge recovery requires a relative active task path")
    task_path = (directory / relative).resolve(strict=True)
    pattern = r"evolution/round_\d{2}/(?:base_logs|eval_batch/[A-Za-z0-9_]+|eval_finalists/[A-Za-z0-9_]+)/[1-9]\d*\.json"
    if (not task_path.is_relative_to(directory) or not re.fullmatch(pattern, task_path.relative_to(directory).as_posix())):
        raise ValueError("Judge recovery target is outside an active native learning stage")
    return root, directory, task_path, manifest


def saved_judge_recovery_hashes(root, job_id, relative_task_path):
    """Pin the task, stopped-worker evidence, controller state and code/Phi."""
    _, directory, task_path, _ = _target(root, job_id, relative_task_path)
    record = read_json(task_path)
    source = Path(record["runtime_artifact"]["source_path"]).resolve(strict=True)
    if not source.is_relative_to(directory / "workspace/EvolveLab/providers"):
        raise ValueError("Saved provider source escaped the job workspace")
    round_dir = directory / "evolution" / task_path.relative_to(directory).parts[1]
    paths = [task_path, directory / "job_status.json", directory / "runtime_summary.json",
             round_dir / "checkpoint.json", directory / "learning.jsonl", source,
             directory / "workspace/EvolveLab/memory_types.py",
             directory / "lineage" / record["provider"] / "orientation.json"]
    for path in (directory / "workspace/storage").rglob("*"):
        if path.is_file():
            paths.append(path)
    if any(not path.is_file() for path in paths):
        raise ValueError("Saved judge recovery evidence is incomplete")
    return {path.relative_to(directory).as_posix(): digest(path.read_bytes()) for path in sorted(set(paths))}


def recover_saved_judge(root, job_id, relative_task_path, *, expected_hashes, revision="repair-r8"):
    """Authorize only native judge/ingestion continuation; make no model calls."""
    root, directory, task_path, manifest = _target(root, job_id, relative_task_path)
    archive = (root / "revisions" / revision / "judge_recovery" / job_id / task_path.relative_to(directory).with_suffix("")).resolve()
    if not archive.is_relative_to(root / "revisions"):
        raise ValueError("Judge recovery archive escaped the revisions directory")
    receipt_path = archive / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt["input_hashes"] != expected_hashes or digest(task_path.read_bytes()) != receipt["authorized_task_sha256"]
                or {key: digest((directory / key).read_bytes()) for key in receipt["preserved_hashes"]} != receipt["preserved_hashes"]):
            raise ValueError("Saved judge recovery artifacts changed after authorization")
        return receipt
    if archive.exists():
        raise ValueError("Partial saved-judge recovery exists; inspect its archived evidence")
    status = read_json(directory / "job_status.json")
    if status.get("status") != "failed" or process_owned(status.get("worker_source") or status):
        raise ValueError("Saved judge recovery requires a stopped failed target job")
    budget = read_json(root / "budget/budget.json")
    for number, pending in budget.get("pending", {}).items():
        attempt = read_json(root / "budget/attempts" / f"{int(number):06d}.json")
        if attempt.get("job_id", pending.get("job_id")) == job_id:
            raise ValueError("Target job has an unsettled model request")
    pins = saved_judge_recovery_hashes(root, job_id, relative_task_path)
    if pins != expected_hashes:
        raise ValueError("Saved judge recovery inputs differ from their explicit hashes")
    record = read_json(task_path)
    if (record.get("status") != "judge_failed" or classify_recovery_record(record) != "judge"
            or record.get("memory_ingestion_started") or record.get("memory_ingestion_complete")
            or not isinstance(record.get("agent_messages"), list) or not record["agent_messages"]
            or not isinstance(record.get("answer"), str) or not record["answer"].strip()
            or record.get("response_truncated") or record.get("stage") != "learning"):
        raise ValueError("Saved task is not an unconsumed complete answer awaiting judge parsing")
    index = record["global_task_index"]
    tasks = [json.loads(line) for line in (directory / "learning.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if type(index) is not int or not 0 <= index < len(tasks) or task_path.stem != str(index + 1):
        raise ValueError("Saved task index differs from the locked dataset")
    task, artifact, protocol = tasks[index], record["runtime_artifact"], manifest["protocol"]
    identity = digest({"task": task, "index": index, "provider": record["provider"], "artifact": artifact,
                       "stage": record["stage"], "dataset": "writingbench", "protocol": protocol, "model": protocol["model"]})
    source = Path(artifact["source_path"])
    snapshot = load(directory / "lineage", record["provider"])
    registry = (directory / "workspace/EvolveLab/memory_types.py").read_text(encoding="utf-8")
    mapping = re.search(r'''MemoryType\.[A-Za-z0-9_]+:\s*\(["']''' + re.escape(artifact["class_name"])
                        + r'''["'],\s*["']''' + re.escape(artifact["module"]) + r'''["']\)''', registry)
    if (identity != record["task_identity"] or task.get("task_id") != record["task_id"]
            or digest(source.read_bytes()) != artifact["code_sha256"] or not mapping
            or snapshot.to_dict() != artifact["orientation_snapshot"]
            or items_digest(snapshot.items) != artifact["orientation_items_sha256"]):
        raise ValueError("Saved judge task identity, provider code or philosophy differs from the paid task")
    normalized, normalization = normalize_grade_output(record["judge_output"], task["domain"])
    grade = validate_grade(normalized, task["domain"])
    attempts = [read_json(path) for path in sorted((root / "budget/attempts").glob("*.json"))]
    matching = []
    for attempt in attempts:
        choices = (attempt.get("response") or {}).get("choices", [])
        usage = attempt.get("usage")
        if (attempt.get("job_id") == job_id and attempt.get("role") == "judge"
                and attempt.get("task_execution_id") == record["task_execution_id"]
                and attempt.get("task_id") == record["task_id"] and attempt.get("provider") == record["provider"]
                and attempt.get("code_sha256") == artifact["code_sha256"]
                and attempt.get("orientation_items_sha256") == artifact["orientation_items_sha256"]
                and attempt.get("status") == "success" and len(choices) == 1
                and choices[0].get("finish_reason") == "stop"
                and choices[0].get("message", {}).get("content") == record["judge_output"]
                and isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"))):
            matching.append(attempt)
    if len(matching) != 1:
        raise ValueError("Saved judge output must match exactly one successful settled judge response")
    usage = task_usage(record, attempts)
    usage["effective_tokens"] = usage["total_tokens"] + usage["reserved_unknown_tokens"]
    authorized = deepcopy(record)
    authorized.update(api_usage=usage, total_model_tokens=usage["total_tokens"], confirmed_model_tokens=usage["total_tokens"],
                      effective_model_tokens=usage["effective_tokens"])
    authorized["recovery_authorized"] = {"revision": revision, "reason": "Reuse the original settled judge response after deterministic parser repair",
        "allow_actor_retry": False, "known_failed_attempts": usage["unrecovered_failed_attempt_ids"],
        "reserved_unknown_attempts": usage["reserved_unknown_attempt_ids"], "next_phase": "judge",
        "reuse_saved_judge_attempt": matching[0]["attempt"], "additional_judge_requests_authorized": 0}
    if pins != saved_judge_recovery_hashes(root, job_id, relative_task_path):
        raise ValueError("Stopped job artifacts changed while preparing judge recovery")
    archive.mkdir(parents=True)
    round_dir = directory / "evolution" / task_path.relative_to(directory).parts[1]
    for path in (task_path, directory / "job_status.json", directory / "runtime_summary.json", round_dir / "checkpoint.json"):
        shutil.copy2(path, archive / (path.name + ".before"))
    write_json(archive / "judge_attempt.before.json", matching[0])
    write_json(task_path, authorized)
    preserved = {key: value for key, value in pins.items() if key != task_path.relative_to(directory).as_posix()}
    if {key: digest((directory / key).read_bytes()) for key in preserved} != preserved:
        raise RuntimeError("Saved judge recovery changed protected code, memory or controller state")
    receipt = {"revision": revision, "job_id": job_id, "task_path": str(task_path), "recorded_utc": now(),
               "input_hashes": pins, "preserved_hashes": preserved, "authorized_task_sha256": digest(task_path.read_bytes()),
               "paid_judge_attempt": matching[0]["attempt"], "original_judge_usage": matching[0]["usage"],
               "normalization": normalization, "validated_quality_score": grade["quality_score"],
               "answer_sha256": digest(record["answer"].encode("utf-8")),
               "original_judge_output_sha256": digest(record["judge_output"].encode("utf-8")),
               "archive": str(archive), "new_model_requests": 0, "native_completion_required": True,
               "task_identity_unchanged": True, "native_memory_unchanged": True}
    write_json(receipt_path, receipt)
    return receipt
