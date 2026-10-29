"""Prepare, execute and summarize paired real-model factorial experiments."""

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
import time

from .cli import PROJECT, FLASH, environment, init_run, source_hash
from .ablation_design import build_design, write_design


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8"))
    os.replace(temp, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(values):
    fields = ("DEFAULT_MODEL", "ANALYSIS_MODEL", "GENERATION_MODEL", "DEFAULT_JUDGE_MODEL",
              "OPENAI_BASE_URL", "WEB_SEARCH_PROVIDER", "WEB_ACCESS_PROVIDER")
    # Credentials are deliberately absent from both the report and fingerprint.
    return {name: values.get(name) for name in fields}


def prepare(root):
    root = Path(root).resolve()
    design = build_design(PROJECT)
    write_design(root, design, project=PROJECT)
    values = environment()
    pinned = {
        "schema_version": 1,
        "design_sha256": design["design_sha256"],
        "adapter_sha256": source_hash(PROJECT / "philoagent"),
        "upstream_sha256": source_hash(FLASH),
        "model_configuration": fingerprint(values),
    }
    state_path = root / "execution.json"
    if state_path.exists():
        if read_json(state_path) != pinned:
            raise ValueError("Experiment code or model configuration changed; use a new experiment root")
    else:
        write_json(state_path, pinned)
        shutil.copytree(PROJECT / "philoagent", root / "code_snapshot" / "philoagent",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for job in design["jobs"]:
        run_dir = root / "jobs" / job["job_id"]
        if not run_dir.exists():
            init_run(job["job_id"], job["seed"], root / job["dev_file"], output_root=root / "jobs")
        manifest = read_json(run_dir / "run.json")
        if manifest["adapter_sha256"] != pinned["adapter_sha256"]:
            raise ValueError("Job adapter differs from the locked experiment")
        if manifest["initial_workspace_sha256"] != pinned["upstream_sha256"]:
            raise ValueError("Job initial source differs from the locked experiment")
        write_json(run_dir / "condition.json", job)
    return design


def run_one(root, job_id):
    root = Path(root).resolve()
    design = read_json(root / "manifest.json")
    execution = read_json(root / "execution.json")
    if source_hash(PROJECT / "philoagent") != execution["adapter_sha256"]:
        raise ValueError("Adapter changed after experiment initialization")
    if fingerprint(environment()) != execution["model_configuration"]:
        raise ValueError("Model configuration changed after experiment initialization")
    job = next(x for x in design["jobs"] if x["job_id"] == job_id)
    run_dir = root / "jobs" / job_id
    result_path = run_dir / "result.json"
    if result_path.exists() and read_json(result_path).get("status") == "complete":
        return read_json(result_path)
    from .ablation_runtime import run_native_job
    started = time.monotonic()
    write_json(run_dir / "job_status.json", {"status": "running", "started_utc": utc_now(), "job": job})
    try:
        result = run_native_job(run_dir, job, design["protocol"], root / job["dev_file"],
                                root / job["heldout_file"], api_env=environment())
        validate_job_result(result, design["protocol"])
        result.update(status="complete", job=job, completed_utc=utc_now(),
                      elapsed_seconds=round(time.monotonic() - started, 3),
                      design_sha256=design["design_sha256"])
        write_json(result_path, result)
        write_json(run_dir / "job_status.json", {"status": "complete", "completed_utc": utc_now()})
        return result
    except Exception as exc:
        # SDK exceptions may include credential-bearing request details.
        write_json(run_dir / "job_status.json", {"status": "failed", "failed_utc": utc_now(),
                   "error_type": type(exc).__name__, "safe_message": str(exc) if isinstance(exc, ValueError) else None})
        raise


def score_from_result(result):
    for key in ("heldout_accuracy", "accuracy"):
        if isinstance(result.get(key), (int, float)):
            return float(result[key])
    heldout = result.get("heldout", {})
    if isinstance(heldout, dict) and isinstance(heldout.get("accuracy"), (int, float)):
        return float(heldout["accuracy"])
    return None


def validate_job_result(result, protocol):
    if result.get("status") in ("failed", "incomplete", "running"):
        raise ValueError("Runtime did not complete the requested job")
    if result.get("rounds_completed") != protocol["rounds"]:
        raise ValueError("Runtime did not complete every requested evolution round")
    if result.get("heldout_count") != protocol["heldout_task_count"]:
        raise ValueError("Runtime did not complete the locked heldout task set")
    score = score_from_result(result)
    if score is None or not 0 <= score <= 1:
        raise ValueError("Runtime has no valid heldout accuracy")


def summarize(root):
    root = Path(root).resolve()
    design = read_json(root / "manifest.json")
    jobs = []
    cells = {}
    for job in design["jobs"]:
        run_dir = root / "jobs" / job["job_id"]
        path = run_dir / "result.json"
        result = read_json(path) if path.exists() else {}
        status_path = run_dir / "job_status.json"
        status = result.get("status") or (read_json(status_path).get("status") if status_path.exists() else "queued")
        score = score_from_result(result)
        entry = {**job, "status": status, "heldout_accuracy": score, "result_path": str(path)}
        jobs.append(entry)
        if status == "complete" and score is not None:
            key = f"{job['seed']}:p{int(job['code_evolve'])}:phi{int(job['orientation_evolve'])}"
            cells.setdefault(key, []).append((job["replicate"], score))
    cell_summary = {}
    for key, rows in sorted(cells.items()):
        values = [x[1] for x in rows]
        cell_summary[key] = {"replicates": len(rows), "mean_accuracy": statistics.mean(values),
                             "replicate_stdev": statistics.stdev(values) if len(values) > 1 else None,
                             "per_replicate": dict(rows)}
    paired = []
    for seed in design["seeds"]:
        for factor in ("code", "orientation"):
            for fixed in (False, True):
                differences = []
                for rep in range(len(design["repetitions"])):
                    def lookup(p, phi):
                        return next((x["heldout_accuracy"] for x in jobs if x["seed"] == seed
                                    and x["replicate"] == rep and x["code_evolve"] == p
                                    and x["orientation_evolve"] == phi and x["status"] == "complete"), None)
                    off = lookup(False, fixed) if factor == "code" else lookup(fixed, False)
                    on = lookup(True, fixed) if factor == "code" else lookup(fixed, True)
                    if off is not None and on is not None:
                        differences.append(on - off)
                if differences:
                    paired.append({"seed": seed, "factor": factor, "other_factor_enabled": fixed,
                                   "paired_replicates": len(differences), "accuracy_differences": differences,
                                   "mean_difference": statistics.mean(differences)})
    counts = {name: sum(x["status"] == name for x in jobs) for name in ("complete", "failed", "running", "queued")}
    report = {"schema_version": 1, "updated_utc": utc_now(), "design_sha256": design["design_sha256"],
              "status": "complete" if counts["complete"] == len(jobs) else "incomplete",
              "job_count": len(jobs), "counts": counts, "conditions": cell_summary,
              "paired_effects": paired, "jobs": jobs,
              "inference_limit": design["inference_limit"],
              "causal_effect_established": False}
    write_json(root / "report.json", report)
    return report


def run_parallel(root, workers=8, only=None, attempts=2):
    root = Path(root).resolve()
    design = prepare(root)
    selected = [x for x in design["jobs"] if not only or x["job_id"] in only]
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")

    def worker(job):
        job_id = job["job_id"]
        path = root / "jobs" / job_id
        result_file = path / "result.json"
        if result_file.exists() and read_json(result_file).get("status") == "complete":
            return job_id, 0
        for attempt in range(1, attempts + 1):
            log_path = path / f"worker-attempt-{attempt:02d}.log"
            with log_path.open("ab") as log:
                completed = subprocess.run([sys.executable, "-m", "philoagent.ablation", "run-job",
                    str(root), job_id], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode == 0:
                return job_id, 0
        return job_id, completed.returncode

    print(json.dumps({"event": "experiment_started", "jobs": len(selected), "workers": workers,
                      "root": str(root)}, ensure_ascii=False), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, job): job for job in selected}
        for future in as_completed(futures):
            job_id, code = future.result()
            report = summarize(root)
            print(json.dumps({"event": "job_finished", "job_id": job_id, "exit_code": code,
                              "counts": report["counts"]}, ensure_ascii=False), flush=True)
    return summarize(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "status", "run-job"):
        command = commands.add_parser(name)
        command.add_argument("experiment_root")
        if name == "run":
            command.add_argument("--workers", type=int, default=8)
            command.add_argument("--only", nargs="*")
            command.add_argument("--attempts", type=int, default=2)
        if name == "run-job":
            command.add_argument("job_id")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.experiment_root)
            result = {"prepared": True, "jobs": result["job_count"], "design_sha256": result["design_sha256"]}
        elif args.command == "run-job":
            result = run_one(args.experiment_root, args.job_id)
        elif args.command == "status":
            result = summarize(args.experiment_root)
        else:
            if not 1 <= args.workers <= 16 or not 1 <= args.attempts <= 3:
                raise ValueError("Workers must be 1..16 and attempts 1..3")
            result = run_parallel(args.experiment_root, args.workers, args.only, args.attempts)
        if args.command == "status":
            result = {key: result[key] for key in ("status", "counts", "job_count", "updated_utc")}
        elif args.command == "run-job":
            result = {"status": result["status"], "job_id": args.job_id}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") != "incomplete" else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "message": str(exc) if isinstance(exc, ValueError) else "See per-job audit artifacts"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
