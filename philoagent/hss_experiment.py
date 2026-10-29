"""Prepare and supervise the authorized v0.0.6 humanities/social/business run.

The supervisor dispatches fresh native worker processes. Task solving, native
evolution and judging live in hss_native/hss_tasks_runtime; this manager does not
invent another learning algorithm. Commands never print credential values.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

from .cli import PROJECT, FLASH, PROVIDER, environment, init_run, source_hash
from .orientation import bootstrap


AUTHORIZATION = {
    "date": "2026-10-29", "timezone": "Asia/Hong_Kong",
    "user_message": "批准开始正式运行",
    "plan": "planv0.0.6.md", "scope": "three_domains_three_groups_three_repetitions",
    "jobs": 27, "rounds_per_job": 2,
    "budget_reference": "research/hss-business-budget-20261029.md",
    "max_calls": 60000, "max_tokens": 350000000,
    "token_scope": "whole_experiment_soft_stop_with_response_settlement",
    "new_approval_each_stage": False, "unlimited_spending": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    for attempt in range(7):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows may briefly hold the destination during another writer's
            # atomic replacement. Keep this writer's complete temporary file.
            if attempt == 6:
                raise
            time.sleep(.01 * 2 ** attempt)


def now():
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid):
    """Read process state without Windows' destructive os.kill(pid, 0)."""
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied means it exists.
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_owned(record):
    """A reused PID does not keep a dead experiment worker looking alive."""
    from .hss_budget import process_matches
    identity = record.get("process_identity", {})
    started = identity.get("started_unix")
    observed = next((record.get(key) for key in
                     ("started_utc", "failed_utc", "completed_utc", "finished_utc") if record.get(key)), None)
    if started is None and observed:
        started = datetime.fromisoformat(observed).timestamp()
    return process_matches(record.get("pid"), started_unix=started,
                           executable=identity.get("executable") or "python.exe")


def _own_process():
    from .hss_budget import process_identity
    return {"pid": os.getpid(), "process_identity": process_identity(os.getpid())}


def record_worker_source(root, target, job_id, manifest):
    """Pin the source admitted at worker startup without changing task identity.

    Compatibility hotfixes may keep the protocol while moving the adapter to a
    new audited snapshot. Earlier running processes retain their original source
    evidence; only this worker appends its own startup to the job's history.
    """
    root, target = Path(root).resolve(), Path(target).resolve()
    protocol_revision = manifest["protocol"].get("revision", "initial")
    adapter_revision = manifest.get("adapter_revision", protocol_revision)
    snapshot = manifest.get("adapter_snapshot")
    if snapshot is None:
        candidate = root / "revisions" / adapter_revision / "code_snapshot.after/philoagent"
        snapshot = candidate if candidate.is_dir() else root / "code_snapshot/philoagent"
    snapshot = Path(snapshot)
    if not snapshot.is_absolute():
        snapshot = root / snapshot
    evidence = {"job_id": job_id, "recorded_utc": now(), **_own_process(),
                "adapter_sha256": manifest["adapter_sha256"], "adapter_revision": adapter_revision,
                "protocol_revision": protocol_revision, "snapshot_path": str(snapshot.resolve()),
                "controller_policy": manifest.get("controller_policy", {})}
    with (target / "source_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(target / "worker_source.json", evidence)
    return evidence


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(values):
    """Reproducible public settings; private endpoint/key values stay private."""
    endpoint = values.get("OPENAI_API_BASE") or values.get("OPENAI_BASE_URL") or ""
    return {
        "model": values.get("DEFAULT_MODEL"),
        "analysis_model": values.get("ANALYSIS_MODEL") or values.get("DEFAULT_MODEL"),
        "generation_model": values.get("GENERATION_MODEL") or values.get("DEFAULT_MODEL"),
        "judge_model": values.get("DEFAULT_JUDGE_MODEL") or values.get("DEFAULT_MODEL"),
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
        "web_search_provider": values.get("WEB_SEARCH_PROVIDER", "deepseek"),
        "web_access_provider": values.get("WEB_ACCESS_PROVIDER", "deepseek"),
    }


def protocol(values):
    result = {
        **configuration(values), "rounds": 2, "task_batch_x": 2, "candidates": 2,
        "top_t": 2, "extra_sample_y": 1, "max_workers": 1,
        "clear_storage_per_round": False, "memory_mode": "native_persistent",
        "actor_max_steps": 32, "actor_max_output_tokens": 8192,
        "analysis_max_steps": 20, "analysis_max_output_tokens": 8192,
        "generation_max_output_tokens": 60000, "validation_max_output_tokens": 32768,
        "judge_max_output_tokens": 8192, "repair_max_output_tokens": 32768,
        "request_timeout_seconds": 300, "thinking": "provider-default",
        "temperature": 0, "creativity_index": .5, "compact_provider": False,
        "max_calls": AUTHORIZATION["max_calls"], "max_tokens": AUTHORIZATION["max_tokens"],
        "max_fix_attempts": 3, "enable_auto_fix": True, "max_inflight_requests": 6,
        "quality_success_threshold": 60, "heldout_task_count": 4,
        "development_task_count": 8, "judge_count": 1,
        "judge_policy": "domain_rubric_and_original_task_criteria_anonymous",
        "per_task_spend_stop": False, "shadow_reflection": False,
        "technical_task_auto_retries": 0, "sdk_max_retries": 0,
        "web_requests": "native_search_fetch_evidence_separately_counted",
    }
    # Future runs use the same audited repair defaults as the migrated matrix.
    from .hss_repair_r4 import repaired_protocol
    return repaired_protocol(result)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(root, input_root=None, *, values=None):
    root = Path(root).resolve()
    input_root = Path(input_root or PROJECT / "data/hss_business/writingbench-20261029-r3").resolve(strict=True)
    values = environment() if values is None else values
    if not values.get("DEFAULT_MODEL") or not values.get("OPENAI_API_KEY"):
        raise ValueError("Configured model and API key are required for the authorized run")
    source = read_json(input_root / "manifest.json")
    if source.get("repetitions") != 3 or len(source.get("jobs", [])) != 27:
        raise ValueError("The authorization covers the prepared 27-job v0.0.6 matrix")
    for relative, expected in source["artifact_sha256"].items():
        if file_sha(input_root / relative) != expected:
            raise ValueError("Prepared input artifact differs from its manifest")
    job_specs = []
    artifacts = {}
    seed_source = PROJECT / "seeds/systems/dialectical-v1.json"
    artifacts["inputs/phi0.json"] = seed_source.read_bytes()
    for source_job in source["jobs"]:
        job = dict(source_job)
        identifier = job["job_id"]
        lookup = {row["task_id"]: row for row in _rows(input_root / job["learning_file"])}
        learning = [lookup[key] for key in job["learning_order"]]
        evaluation = {row["task_id"]: row for row in _rows(input_root / job["evaluation_file"])}
        heldout = [evaluation[key] for key in job["evaluation_order"]]
        if len(learning) != 8 or len(heldout) != 4 or set(lookup) & set(evaluation):
            raise ValueError("Incorrect eight/four learning/evaluation split")
        for split, rows in (("learning", learning), ("evaluation", heldout)):
            relative = f"inputs/{split}/{identifier}.jsonl"
            artifacts[relative] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
            job[f"{split}_file"] = relative
        job["heldout_file"] = job["evaluation_file"]
        job["replicate"] = job["repetition"] - 1
        job_specs.append(job)
    # Interleave all three groups and domains before advancing repetition.
    group_order = ("native", "fixed", "full")
    domain_order = ("humanities", "social", "business")
    job_specs.sort(key=lambda job: (job["repetition"],
                                   domain_order.index(job["domain"]), group_order.index(job["group"])))
    pinned = {
        "schema_version": 1, "kind": "authorized_hss_business_formal_experiment",
        "authorization": AUTHORIZATION, "protocol": protocol(values), "jobs": job_specs,
        "input_manifest_sha256": file_sha(input_root / "manifest.json"),
        "adapter_sha256": source_hash(PROJECT / "philoagent"),
        "upstream_sha256": source_hash(FLASH),
        "input_hashes": {relative: hashlib.sha256(content).hexdigest() for relative, content in artifacts.items()},
        "holdout_quality_metric": "0_to_100_continuous_not_binary_accuracy",
        "independence": "36_distinct_questions; repetitions_are_not_new_questions",
    }
    manifest = root / "manifest.json"
    if manifest.exists() and read_json(manifest) != pinned:
        raise ValueError("Formal run configuration changed; do not mix versions in one run")
    for relative, content in artifacts.items():
        target = root / relative
        if target.exists() and target.read_bytes() != content:
            raise ValueError("An initialized formal input differs")
    for relative, content in artifacts.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
    if not manifest.exists():
        write_json(manifest, pinned)
        write_json(root / "authorization.json", {**AUTHORIZATION, "recorded_utc": now()})
        shutil.copytree(PROJECT / "philoagent", root / "code_snapshot/philoagent",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    from .hss_budget import HSSBudget
    HSSBudget(root / "budget", max_calls=AUTHORIZATION["max_calls"], max_tokens=AUTHORIZATION["max_tokens"], max_inflight=6)
    return pinned


def initialize_job(root, job):
    """Create one isolated workspace lazily, so all workers use frozen sources."""
    root = Path(root)
    target = root / "jobs" / job["job_id"]
    manifest = read_json(root / "manifest.json")
    if source_hash(PROJECT / "philoagent") != manifest["adapter_sha256"]:
        raise ValueError("Adapter changed after experiment freeze")
    if source_hash(FLASH) != manifest["upstream_sha256"]:
        raise ValueError("Native source changed after experiment freeze")
    for relative in (job["heldout_file"], "inputs/phi0.json"):
        if file_sha(root / relative) != manifest["input_hashes"][relative]:
            raise ValueError("Frozen evaluation input or Phi seed changed")
    if not target.exists():
        if job["group"] == "native":
            init_run(job["job_id"], "none", output_root=root / "jobs")
        else:
            init_run(job["job_id"], "dialectical", output_root=root / "jobs", seed_file=root / "inputs/phi0.json")
        shutil.copyfile(root / job["learning_file"], target / "learning.jsonl")
        write_json(target / "condition.json", job)
    if read_json(target / "condition.json") != job:
        raise ValueError("Existing job condition differs")
    if file_sha(target / "learning.jsonl") != manifest["input_hashes"][job["learning_file"]]:
        raise ValueError("Job learning input changed")
    return target


def run_one(root, job_id):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    job = dict(next(item for item in manifest["jobs"] if item["job_id"] == job_id))
    target = initialize_job(root, job)
    result_path = target / "result.json"
    if result_path.exists() and read_json(result_path).get("status") == "complete":
        return read_json(result_path)
    values = environment()
    if configuration(values) != {key: manifest["protocol"][key] for key in configuration(values)}:
        raise ValueError("Model/tool configuration changed since authorization freeze")
    for key, value in values.items():
        if isinstance(value, str):
            os.environ[key] = value
    job["learning_path"] = str(target / "learning.jsonl")
    job["heldout_path"] = str(root / job["heldout_file"])
    worker_source = record_worker_source(root, target, job_id, manifest)
    source_fields = {"worker_source": worker_source, "worker_adapter_sha256": worker_source["adapter_sha256"]}
    write_json(target / "job_status.json", {"status": "running", "started_utc": now(), **_own_process(),
               "job_id": job_id, **source_fields})
    started = time.monotonic()
    try:
        from .hss_native import run_job
        policy = manifest.get("controller_policy")
        result = run_job(target, job, manifest["protocol"], root / "budget",
                         **({"controller_policy": policy} if policy else {}))
        if result.get("status") != "complete":
            raise RuntimeError("Native worker did not complete the formal lineage")
        result.update(job=job, completed_utc=now(), elapsed_seconds=round(time.monotonic() - started, 2), **source_fields)
        write_json(result_path, result)
        write_json(target / "job_status.json", {"status": "complete", "completed_utc": now(),
                   "pid": os.getpid(), **source_fields})
        return result
    except Exception as exc:
        write_json(target / "job_status.json", {"status": "failed", "failed_utc": now(),
                   "error_type": type(exc).__name__, "pid": os.getpid(), "job_id": job_id, **source_fields})
        raise


def summarize(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    from .hss_budget import HSSBudget
    budget = HSSBudget(root / "budget", max_calls=manifest["protocol"]["max_calls"],
                       max_tokens=manifest["protocol"]["max_tokens"],
                       max_inflight=manifest["protocol"].get("max_inflight_requests", 6)).summary()
    entries = []
    for job in manifest["jobs"]:
        directory = root / "jobs" / job["job_id"]
        state = read_json(directory / "job_status.json") if (directory / "job_status.json").exists() else {}
        result = read_json(directory / "result.json") if (directory / "result.json").exists() else {}
        grade = result.get("heldout_quality")
        if grade is None:
            grade = result.get("heldout", {}).get("quality_score") if isinstance(result.get("heldout"), dict) else None
        status = result.get("status") or state.get("status", "queued")
        if status == "running" and not process_owned(state):
            status = "interrupted"
        events = list((directory / "evidence/stages").glob("*.json"))
        entries.append({"job_id": job["job_id"], "domain": job["domain"], "group": job["group"],
                        "repetition": job["repetition"], "status": status, "heldout_quality": grade,
                        "stage_artifacts": len(events), "result_path": str(directory / "result.json")})
    counts = {name: sum(item["status"] == name for item in entries) for name in ("queued", "running", "complete", "failed", "interrupted")}
    cells = {}
    for domain in ("humanities", "social", "business"):
        for group in ("native", "fixed", "full"):
            scores = [item["heldout_quality"] for item in entries if item["domain"] == domain
                      and item["group"] == group and item["status"] == "complete" and item["heldout_quality"] is not None]
            if scores:
                cells[f"{domain}:{group}"] = {"mean_quality": statistics.mean(scores), "replicates": len(scores), "scores": scores}
    report = {"kind": manifest["kind"], "updated_utc": now(), "job_count": len(entries), "counts": counts,
              "status": "complete" if counts["complete"] == len(entries) else "incomplete",
              "budget": budget, "quality_by_domain_group": cells, "jobs": entries,
              "authorization": manifest["authorization"], "quality_metric": "continuous_0_to_100",
              "model_effectiveness_established": False}
    write_json(root / "report.json", report)
    return report


def supervise(root, workers=6):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    status_path = root / "supervisor.json"
    previous = read_json(status_path) if status_path.exists() else {}
    previous_pid = previous.get("pid")
    if previous.get("status") == "running" and previous_pid and previous_pid != os.getpid():
        if process_owned(previous):
            raise ValueError("A formal supervisor is already running")
    write_json(status_path, {"status": "running", **_own_process(), "started_utc": now(), "workers": workers})
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")

    def execute(job):
        target = initialize_job(root, job)
        result = target / "result.json"
        if result.exists() and read_json(result).get("status") == "complete":
            return job["job_id"], 0
        # One task process per lineage; native checkpoints own legal resumption.
        with (target / "worker.log").open("ab") as log:
            process = subprocess.run([sys.executable, "-u", "-m", "philoagent.hss_experiment", "run-job",
                                      str(root), job["job_id"]], cwd=PROJECT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
        return job["job_id"], process.returncode

    failures = []
    try:
        # Submit only a small window. Budget refusal stops new dispatch while
        # already executing workers preserve their completed state.
        jobs = iter(manifest["jobs"])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            for _ in range(workers):
                job = next(jobs, None)
                if job:
                    active[pool.submit(execute, job)] = job
            while active:
                future = next(as_completed(active))
                job = active.pop(future)
                try:
                    job_id, code = future.result()
                except Exception as exc:
                    job_id, code = job["job_id"], 1
                    failures.append({"job_id": job_id, "error_type": type(exc).__name__})
                report = summarize(root)
                print(json.dumps({"event": "job_finished", "job_id": job_id, "exit_code": code,
                                  "counts": report["counts"]}, ensure_ascii=False), flush=True)
                if code:
                    failures.append({"job_id": job_id, "exit_code": code})
                if not report["budget"].get("blocked_reason"):
                    job = next(jobs, None)
                    if job:
                        active[pool.submit(execute, job)] = job
        report = summarize(root)
        write_json(status_path, {"status": "complete" if report["status"] == "complete" else "needs_attention",
                                "pid": os.getpid(), "finished_utc": now(), "failures": failures,
                                "counts": report["counts"]})
        return report
    except Exception as exc:
        write_json(status_path, {"status": "failed", "pid": os.getpid(), "finished_utc": now(), "error_type": type(exc).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "status", "run-job"):
        command = commands.add_parser(name)
        command.add_argument("experiment_root", type=Path)
        if name == "prepare":
            command.add_argument("--input-root", type=Path)
        elif name == "run":
            command.add_argument("--workers", type=int, choices=range(1, 28), default=6)
        elif name == "run-job":
            command.add_argument("job_id")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.experiment_root, args.input_root)
            print(json.dumps({"prepared": True, "jobs": len(result["jobs"]), "authorization": AUTHORIZATION}, ensure_ascii=False))
        elif args.command == "run-job":
            result = run_one(args.experiment_root, args.job_id)
            print(json.dumps({"job_id": args.job_id, "status": result["status"]}, ensure_ascii=False))
        elif args.command == "run":
            supervise(args.experiment_root, args.workers)
        else:
            result = summarize(args.experiment_root)
            print(json.dumps({key: result[key] for key in ("status", "counts", "budget", "quality_by_domain_group")}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "message": str(exc) if isinstance(exc, ValueError) else "Inspect saved worker and budget evidence"}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
