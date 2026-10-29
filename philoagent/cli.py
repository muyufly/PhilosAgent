"""Small launcher around the upstream entrypoints; no experimental scheduler."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from . import UPSTREAM_COMMIT
from .orientation import bootstrap, load

PROJECT = Path(__file__).resolve().parent.parent
UPSTREAM = PROJECT / "vendor" / "MemEvolve"
FLASH = UPSTREAM / "Flash-Searcher-main"
PROVIDER = "lightweight_memory"
SEEDS = ("engineering", "dialectical", "critical", "none")


def source_hash(root):
    """Pin the copied code, including local adapters, without hashing secrets/data."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "storage", "data"} for part in path.relative_to(root).parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(env_file=None):
    values = os.environ.copy()
    path = Path(env_file) if env_file else PROJECT / ".env"
    if path.exists():
        from dotenv.parser import parse_stream
        with path.open(encoding="utf-8-sig") as stream:
            bindings = list(parse_stream(stream))
        invalid_lines = [binding.original.line for binding in bindings if binding.error]
        if invalid_lines:
            raise ValueError(f"Invalid .env syntax at lines {invalid_lines}; use KEY=value as in .env.example")
        from dotenv import dotenv_values
        values.update({k: v for k, v in dotenv_values(path, encoding="utf-8-sig").items() if v is not None})
    base = values.get("OPENAI_BASE_URL") or values.get("OPENAI_API_BASE")
    if values.get("OPENAI_BASE_URL") and values.get("OPENAI_API_BASE") and values["OPENAI_BASE_URL"] != values["OPENAI_API_BASE"]:
        raise ValueError("OPENAI_BASE_URL and OPENAI_API_BASE must identify the same endpoint")
    if base:
        values["OPENAI_BASE_URL"] = values["OPENAI_API_BASE"] = base
    for name, default, allowed in (
        ("WEB_SEARCH_PROVIDER", "serper", {"serper", "deepseek"}),
        ("WEB_ACCESS_PROVIDER", "jina", {"jina", "crawl4ai", "deepseek"}),
    ):
        provider = values.get(name, default).strip().lower()
        if provider not in allowed:
            raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
        values[name] = provider
    return values


def missing_configuration(values, *, evolving=False):
    names = ["DEFAULT_MODEL", "OPENAI_API_KEY"]
    if values.get("WEB_SEARCH_PROVIDER", "serper") == "serper":
        names.append("SERPER_API_KEY")
    if values.get("WEB_ACCESS_PROVIDER", "jina") == "jina":
        names.append("JINA_API_KEY")
    if evolving:
        names.extend(["ANALYSIS_MODEL", "GENERATION_MODEL"])
    return [name for name in names if not values.get(name) or values[name].startswith("your_")]


def doctor(env_file=None):
    required = ["dotenv", "openai", "yaml", "rich", "json_repair", "huggingface_hub", "pandas", "PIL", "litellm"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    configuration_error = None
    try:
        values = environment(env_file)
    except ValueError as exc:
        values = {}
        configuration_error = str(exc)
    result = {
        "python": sys.version.split()[0],
        "upstream_expected_commit": UPSTREAM_COMMIT,
        "upstream_present": (FLASH / "evolve_cli.py").is_file(),
        "missing_modules": missing,
        "missing_live_configuration": missing_configuration(values, evolving=True),
        "configuration_error": configuration_error,
        "web_backends": {
            "search": values.get("WEB_SEARCH_PROVIDER", "serper"),
            "fetch": values.get("WEB_ACCESS_PROVIDER", "jina"),
        },
        "live_model_calls_performed": False,
    }
    if (UPSTREAM / ".git").exists():
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, capture_output=True, text=True, check=True).stdout.strip()
        result["upstream_actual_commit"] = revision
        result["upstream_commit_matches"] = revision == UPSTREAM_COMMIT
    return result


def init_run(name, seed, data=None, *, output_root=None, seed_file=None):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("Run name must contain only letters, digits, underscores or hyphens")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    if seed == "none" and seed_file is not None:
        raise ValueError("--seed none cannot be combined with --seed-file")
    destination = (Path(output_root) if output_root else PROJECT / "runs") / name
    if destination.exists():
        raise FileExistsError(f"Run already exists: {destination}; reuse it with run, or choose a new name")
    source = Path(data).resolve(strict=True) if data else None
    if source:
        from .data import read_jsonl
        rows = read_jsonl(source)
        if not rows or any(not all(isinstance(row.get(field), str) and row[field].strip()
                                  for field in ("question", "answer", "root_url")) for row in rows):
            raise ValueError("Development data must be nonempty WebWalkerQA JSONL")
    seed_bytes = None
    if seed_file is not None:
        seed_path = Path(seed_file).resolve(strict=True)
        seed_bytes = seed_path.read_bytes()
        document = json.loads(seed_bytes.decode("utf-8-sig"))
        items = document.get("items") if isinstance(document, dict) else None
        if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
                or document["schema_version"] != 1 or not isinstance(items, list) or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)):
            raise ValueError("Custom seed requires schema_version 1 and a nonempty items list of nonempty strings")
    else:
        seed_path = PROJECT / "seeds" / f"{seed}.json"
        items = [] if seed == "none" else json.loads(seed_path.read_text(encoding="utf-8"))["items"]
    if not FLASH.is_dir():
        raise FileNotFoundError(f"Pinned upstream source missing: {FLASH}")
    workspace = destination / "workspace"
    # Native creation edits provider registration/source. Each run needs its own
    # checkout, not only its own JSON memory directory. Never copy local secrets.
    shutil.copytree(FLASH, workspace, ignore=shutil.ignore_patterns(
        ".git", ".env", ".env.*", "__pycache__", "*.pyc", "storage", "data", ".DS_Store"
    ))
    if seed_bytes is not None:
        (destination / "seed_source.json").write_bytes(seed_bytes)
    phi_path = bootstrap(destination / "lineage", PROVIDER, items)
    data_hash = None
    if source:
        dev = workspace / "data" / "webwalkerqa" / "webwalkerqa_subset_170.jsonl"
        dev.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dev)
        data_hash = hashlib.sha256(dev.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "initial_provider": PROVIDER,
        "orientation_sha256": load(destination / "lineage", PROVIDER).sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "initial_workspace_sha256": source_hash(workspace),
        "adapter_sha256": source_hash(PROJECT / "philoagent"),
        "development_data_sha256": data_hash,
        "development_task_count": len(rows) if source else 0,
    }
    if seed_bytes is not None:
        manifest["seed_source"] = {
            "path": str(seed_path),
            "sha256": hashlib.sha256(seed_bytes).hexdigest(),
            "snapshot": "seed_source.json",
        }
    (destination / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run_dir": str(destination.resolve()), "orientation_path": str(phi_path), **manifest}


def launch(run_dir, *, mode="task", env_file=None, tasks=2, rounds=2, candidates=1, batch_size=2, max_steps=40, dry_run=False):
    run_dir = Path(run_dir).resolve(strict=True)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    workspace = run_dir / "workspace"
    data_path = workspace / "data" / "webwalkerqa" / "webwalkerqa_subset_170.jsonl"
    if not data_path.is_file():
        raise ValueError("No development dataset: create a new run using init --data <dev.jsonl>")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["development_data_sha256"]:
        raise ValueError("Development dataset differs from this run's manifest")
    if source_hash(PROJECT / "philoagent") != manifest["adapter_sha256"]:
        raise ValueError("Adapter changed since initialization; create a new run")
    values = environment(env_file)
    values["PHILO_LINEAGE_ROOT"] = str(run_dir / "lineage")
    values["PHILO_EVIDENCE_DIR"] = str(run_dir / "evidence")
    values["PYTHONPATH"] = os.pathsep.join([str(workspace), str(PROJECT), values.get("PYTHONPATH", "")])
    values["PYTHONUTF8"] = "1"
    values["PYTHONDONTWRITEBYTECODE"] = "1"
    values["DEFAULT_JUDGE_MODEL"] = values.get("DEFAULT_JUDGE_MODEL") or values.get("DEFAULT_MODEL", "")
    state_path = run_dir / "evolution" / "evolve_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    current_provider = state.get("best_provider", PROVIDER)
    if min(tasks, rounds, candidates, batch_size, max_steps) < 1:
        raise ValueError("Task, round, candidate and batch counts must be positive")
    if mode == "task":
        if tasks > manifest["development_task_count"]:
            raise ValueError("Requested more tasks than the development dataset contains")
        cmd = [sys.executable, "run_flash_searcher_webwalkerqa.py", "--infile", str(data_path),
               "--outfile", str(run_dir / "task_results.jsonl"), "--memory_provider", current_provider,
               "--sample_num", str(tasks), "--concurrency", "1", "--max_steps", str(max_steps),
               "--direct_output_dir", str(run_dir / "tasks")]
    else:
        cursor = state.get("dataset_cursor", 0)
        remaining_rounds = rounds - state.get("round", 0)
        if remaining_rounds <= 0:
            return {"executed": False, "completed_rounds": state.get("round", 0), "reason": "Requested total rounds already completed"}
        if cursor + 2 * batch_size * remaining_rounds > manifest["development_task_count"]:
            raise ValueError("Insufficient development tasks for the requested native evolution rounds")
        cmd = [sys.executable, "evolve_cli.py", "auto-evolve", "webwalkerqa", "--yes",
               "--work-dir", str(run_dir / "evolution"), "--provider", PROVIDER,
               "--num-rounds", str(remaining_rounds), "--num-systems", str(candidates),
               "--task-batch-x", str(batch_size), "--top-t", str(min(2, candidates + 1)), "--extra-sample-y", "1"]
    missing = missing_configuration(values, evolving=mode == "evolve")
    if dry_run:
        return {"command": cmd, "cwd": str(workspace), "missing_configuration": missing, "executed": False}
    if missing:
        raise ValueError("Missing configuration: " + ", ".join(missing) + "; fill .env first")
    load(run_dir / "lineage", current_provider)
    completed = subprocess.run(cmd, cwd=workspace, env=values, check=True)
    return {"exit_code": completed.returncode, "run_dir": str(run_dir), "executed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Inspect setup without making API calls")
    check.add_argument("--env-file")
    prepare = commands.add_parser("prepare-data", help="Pin and split an existing WebWalkerQA JSONL file")
    prepare.add_argument("source")
    prepare.add_argument("output_dir")
    create = commands.add_parser("init", help="Create an isolated native workspace and initial orientation")
    create.add_argument("name")
    create.add_argument("--seed", choices=SEEDS, default="engineering")
    create.add_argument("--seed-file", help="Custom schema_version 1 seed JSON, frozen into the new run; incompatible with --seed none")
    create.add_argument("--data", help="Development split only; holdout must stay outside the run")
    run = commands.add_parser("run", help="Launch a native task batch or native outer evolution")
    run.add_argument("run_dir")
    run.add_argument("--mode", choices=["task", "evolve"], default="task")
    run.add_argument("--env-file")
    run.add_argument("--tasks", type=int, default=2)
    run.add_argument("--max-steps", type=int, default=40, help="Maximum actor steps per task (task mode)")
    run.add_argument("--rounds", type=int, default=2, help="Total target rounds, including already completed rounds")
    run.add_argument("--candidates", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.env_file)
        elif args.command == "init":
            result = init_run(args.name, args.seed, args.data, seed_file=args.seed_file)
        elif args.command == "prepare-data":
            from .data import prepare_dataset
            result = prepare_dataset(args.source, args.output_dir)
        else:
            result = launch(args.run_dir, mode=args.mode, env_file=args.env_file, tasks=args.tasks,
                            rounds=args.rounds, candidates=args.candidates, batch_size=args.batch_size,
                            max_steps=args.max_steps, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")
