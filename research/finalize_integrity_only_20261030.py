"""Review and finish one already-ingested native task with networking disabled.

Default: emit a read-only plan. --temp-check exercises the original NativeRunner
integrity path on a temporary task copy. Only --apply --plan FILE writes the
reviewed live task. It never launches a whole lineage or publishes a partial
batch summary, and model/provider construction and socket calls are forbidden.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from philoagent.ablation_runtime import digest, items_digest
from philoagent.cli import source_hash
from philoagent.hss_evaluation import native_quality_summary, validate_grade
from philoagent.hss_experiment import process_owned, read_json, write_json
from philoagent.hss_tasks_runtime import NativeHSSRunner, classify_recovery_record
from philoagent.orientation import load

JOB = "social-full-r1"
PROVIDER = "aspect_cascade_memory"
RELATIVE_TASK = Path("jobs") / JOB / "evolution/round_01/eval_finalists" / PROVIDER / "5.json"
SOURCE_SHA256 = "550e2eca0f4501d47d684b4f0f38b2ba08b0b5bdaba2ef949e9115a366603270"


class ForbiddenNetworkOperation(RuntimeError):
    pass


def forbidden(*args, **kwargs):
    raise ForbiddenNetworkOperation("Offline integrity finalization forbids model, provider, and network operations")


@contextmanager
def no_network():
    with patch.object(socket.socket, "connect", forbidden), \
            patch.object(socket.socket, "connect_ex", forbidden), \
            patch.object(socket, "create_connection", forbidden), \
            patch.object(socket, "getaddrinfo", forbidden), \
            patch.dict(os.environ, {"PHILO_HSS_OBSERVE_ACCOUNTING": "1", "PHILO_HSS_DATA_FIRST": "0"}):
        yield


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def storage_manifest(directory):
    return {str(p.resolve()): sha(p) for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def backup_original_record(root, task, expected_hash):
    """Save exact latest bytes once, before NativeRunner may rewrite the task."""
    raw = Path(task).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_hash
    directory = Path(root) / "native_recovery_20261030/integrity_only" / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    backup = directory / "record.before.json"
    with backup.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    backup.chmod(0o444)
    assert sha(backup) == expected_hash
    return {"original_record_backup_path": str(backup.resolve()),
            "original_record_backup_sha256": expected_hash}


def current_artifact(root, record):
    """Native artifact fields verified without importing or constructing a provider."""
    job = Path(root) / "jobs" / JOB
    old = record["runtime_artifact"]
    source = job / "workspace/EvolveLab/providers/aspect_cascade_memory_provider.py"
    assert source.resolve() == Path(old["source_path"]).resolve()
    assert sha(source) == old["code_sha256"] == SOURCE_SHA256
    snapshot = load(job / "lineage", PROVIDER)
    artifact = {"provider": PROVIDER, "module": old["module"], "class_name": old["class_name"],
                "source_path": str(source.resolve()), "code_sha256": sha(source),
                "orientation_snapshot": snapshot.to_dict(), "orientation_items_sha256": items_digest(snapshot.items)}
    assert artifact == old, "Provider code or orientation differs from the saved task"
    return artifact


def plan(root):
    root = Path(root).resolve(strict=True)
    path = root / RELATIVE_TASK
    record = read_json(path)
    assert record["provider"] == PROVIDER and record["task_id"] == "WB0704"
    assert classify_recovery_record(record) == "integrity" and record["memory_ingestion_complete"] is True
    assert record.get("memory_ingestion_started") is True and record.get("recovery_authorized", {}).get("next_phase") == "integrity"
    grade = validate_grade(record["grade"], record["domain"])
    assert grade["quality_score"] == record["quality_score"] == 90.6
    assert isinstance(record.get("answer"), str) and record["answer"].strip()
    fallback = NativeHSSRunner._verified_actor_completion(record)
    assert fallback == {14687, 14704, 14728}
    usage = record["api_usage"]
    assert usage["request_count"] > 0 and usage["blocking_failed_request_count"] == 0
    assert set(usage["unrecovered_failed_attempt_ids"]) <= fallback
    job = root / "jobs" / JOB
    for name in ("job_status.json", "worker_source.json"):
        if (job / name).exists():
            assert not process_owned(read_json(job / name)), "The target lineage still has an owned worker"
    artifact = current_artifact(root, record)
    dataset = job / "learning.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = record["global_task_index"]
    assert index == 4 and rows[index]["task_id"] == record["task_id"]
    manifest = read_json(root / "manifest.json")
    assert source_hash(PROJECT / "philoagent") == manifest["adapter_sha256"], "Publish the reviewed adapter before offline finalization"
    pins = {str(path): sha(path), str(dataset): sha(dataset),
            str(root / "manifest.json"): sha(root / "manifest.json"),
            str(Path(artifact["source_path"])): sha(Path(artifact["source_path"])),
            str(job / "lineage" / PROVIDER / "orientation.json"): sha(job / "lineage" / PROVIDER / "orientation.json")}
    pins.update(storage_manifest(job / "workspace/storage" / PROVIDER))
    proof = record["native_actor_completion_recovery"]
    pins[str(Path(proof["receipt_path"]))] = sha(proof["receipt_path"])
    pins.update(proof["pinned_files"])
    return {"kind": "native_single_task_integrity_only_plan", "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
            "job_id": JOB, "provider": PROVIDER, "task_id": record["task_id"], "index": index,
            "record_path": str(path), "record_sha256": sha(path), "dataset_path": str(dataset),
            "adapter_sha256": manifest["adapter_sha256"], "protocol_sha256": digest(manifest["protocol"]),
            "quality_score": grade["quality_score"], "answer_sha256": digest(record["answer"].encode("utf-8")),
            "saved_api_usage": deepcopy(usage), "signed_fallback_attempts": sorted(fallback),
            "pinned_files": pins, "model_provider_network_forbidden": True,
            "memory_ingestion_repeated": False, "batch_summary_published": False}


def verify_plan(value):
    assert value["kind"] == "native_single_task_integrity_only_plan" and value["version"] == 1
    root = Path(value["root"]).resolve(strict=True)
    assert Path(value["record_path"]).resolve() == (root / RELATIVE_TASK).resolve()
    for filename, expected in value["pinned_files"].items():
        assert sha(filename) == expected, "Reviewed integrity evidence changed"
    current = plan(root)
    for key in ("record_sha256", "adapter_sha256", "protocol_sha256", "quality_score", "answer_sha256",
                "saved_api_usage", "signed_fallback_attempts", "pinned_files"):
        assert current[key] == value[key], "Reviewed integrity plan no longer matches"
    return root


def make_runner(root, work_root, record):
    root, work_root = Path(root).resolve(), Path(work_root).resolve()
    manifest = read_json(root / "manifest.json")
    runner = NativeHSSRunner(work_root, root / "jobs" / JOB / "workspace", manifest["protocol"],
                             {"model_id": manifest["protocol"]["model"]},
                             runner=SimpleNamespace(load_memory_provider=forbidden),
                             native=SimpleNamespace(OpenAIServerModel=forbidden, ToolCallingAgent=forbidden,
                                                    WebSearchTool=forbidden, CrawlPageTool=forbidden),
                             base_agent_class=forbidden, usage_reader=lambda: [])
    runner._new_model = forbidden
    runner._artifact = lambda provider: current_artifact(root, record) if provider == PROVIDER else forbidden()
    return runner


def finalize_copy(root, output_dir, *, work_root=None):
    """Run the unmodified native per-task path; no partial batch cache is emitted."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    record = read_json(output_dir / "5.json")
    assert classify_recovery_record(record) == "integrity"
    rows = [json.loads(line) for line in (root / "jobs" / JOB / "learning.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with no_network():
        runner = make_runner(root, work_root or root / "jobs" / JOB, record)
        artifact = runner._artifact(PROVIDER)
        completed = runner._run_task(rows[4], 4, PROVIDER, output_dir, artifact=artifact,
                                     stage="learning", stage_id=digest(str(output_dir))[:16], dataset_name="writingbench")
    assert completed["status"] == "completed" and completed["memory_ingestion_complete"]
    assert completed["answer"] == record["answer"] and completed["grade"] == record["grade"]
    assert completed["api_usage_current_execution"]["request_count"] == 0
    assert completed["api_usage"] == record["api_usage"]
    assert native_quality_summary([completed])["success"] is True
    return completed


def temp_check(value):
    root = verify_plan(value)
    before = {name: sha(name) for name in value["pinned_files"]}
    before_budget = sha(root / "budget/budget.json")
    with TemporaryDirectory(prefix="philo-integrity-only-") as temporary:
        sandbox = Path(temporary)
        output = sandbox / "task"
        output.mkdir()
        shutil.copy2(value["record_path"], output / "5.json")
        backup = backup_original_record(sandbox, output / "5.json", value["record_sha256"])
        result = finalize_copy(root, output, work_root=sandbox)
        assert sha(backup["original_record_backup_path"]) == value["record_sha256"]
        assert Path(backup["original_record_backup_path"]).read_bytes() == Path(value["record_path"]).read_bytes()
        assert not (output / "_quality/summary.json").exists()
        positive = {"status": result["status"], "quality_score": result["quality_score"],
                    "new_request_count": result["api_usage_current_execution"]["request_count"],
                    "new_tokens": result["api_usage_current_execution"]["total_tokens"]}
        # A task with unfinished ingestion must never pass this offline route.
        negative = deepcopy(read_json(value["record_path"]))
        negative["memory_ingestion_complete"] = False
        write_json(output / "5.json", negative)
        rejected = False
        try:
            finalize_copy(root, output, work_root=sandbox)
        except AssertionError:
            rejected = True
        assert rejected
        # Even if a wrong phase is accidentally invoked, the normal native
        # _new_model entry point is disabled before it can construct a client.
        with no_network():
            callback = make_runner(root, sandbox, read_json(value["record_path"]))
            blocked = False
            try:
                callback._new_model()
            except ForbiddenNetworkOperation:
                blocked = True
            assert blocked
            provider_blocked = False
            try:
                callback.runner.load_memory_provider(PROVIDER, None)
            except ForbiddenNetworkOperation:
                provider_blocked = True
            assert provider_blocked
    assert before == {name: sha(name) for name in value["pinned_files"]}
    assert sha(root / "budget/budget.json") == before_budget
    return {"positive": positive, "unfinished_ingestion_rejected": True,
            "new_model_construction_forbidden": True, "provider_loading_forbidden": True,
            "live_pins_unchanged": True, "budget_unchanged": True,
            "original_record_backup_hash_matches": True,
            "partial_batch_summary_published": False}


def apply(value):
    root = verify_plan(value)
    supervisor = read_json(root / "native_recovery_20261030/supervisor.json")
    assert supervisor.get("operational_status") == "waiting_for_credentials"
    assert supervisor.get("running") == supervisor.get("preparing") == 0
    assert not read_json(root / "budget/budget.json").get("pending")
    task = Path(value["record_path"])
    before_storage = storage_manifest(root / "jobs" / JOB / "workspace/storage" / PROVIDER)
    budget_before = sha(root / "budget/budget.json")
    backup = backup_original_record(root, task, value["record_sha256"])
    attempt_directory = Path(backup["original_record_backup_path"]).parent
    write_json(attempt_directory / "attempt.json", {"status": "started", "task_path": str(task), **backup})
    try:
        result = finalize_copy(root, task.parent)
        assert storage_manifest(root / "jobs" / JOB / "workspace/storage" / PROVIDER) == before_storage
        assert sha(root / "budget/budget.json") == budget_before
        assert sha(backup["original_record_backup_path"]) == value["record_sha256"]
    except BaseException as exc:
        write_json(attempt_directory / "outcome.json", {"status": "failed", "error_type": type(exc).__name__,
                   "task_path": str(task), "task_record_after_sha256": sha(task) if task.exists() else None, **backup})
        raise
    receipt = {"kind": "native_single_task_integrity_only_completion", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "task_path": str(task), "task_record_before_sha256": value["record_sha256"],
               "task_record_after_sha256": sha(task), "status": result["status"],
               "quality_score": result["quality_score"], "saved_api_usage_preserved": result["api_usage"] == value["saved_api_usage"],
               "new_api_usage": result["api_usage_current_execution"], "storage_changed": False,
               "budget_changed": False, "partial_batch_summary_published": False,
               "whole_lineage_launched": False, **backup}
    write_json(attempt_directory / "outcome.json", receipt)
    write_json(root / "native_recovery_20261030/aspect_integrity_only_completion.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "runs/hss-business-formal-20261029-r01")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--temp-check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    value = read_json(args.plan) if args.plan else plan(args.root)
    if args.apply:
        if not args.plan or args.temp_check:
            parser.error("--apply requires an existing --plan and cannot combine with --temp-check")
        result = apply(value)
    elif args.temp_check:
        result = temp_check(value)
    else:
        result = value
    if args.plan_out:
        write_json(args.plan_out, value)
    print(json.dumps(result, ensure_ascii=True))
