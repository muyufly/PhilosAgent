"""The user-facing persistent agent; all native execution happens in a child."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from .storage import (PROJECT, assert_product_path, copy_runtime, digest, file_lock,
                      now, read_json, write_json)
from .systems import normalize_task, parse_system, read_tasks


class AgentBusyError(RuntimeError):
    """Another process owns this agent's mutable state."""


class AgentStateError(RuntimeError):
    """Saved state needs attention before additional learning can proceed."""


class Agent:
    def __init__(self, workspace, *, env_file=None):
        self.workspace = assert_product_path(workspace)
        self.env_file = Path(env_file).resolve() if env_file else None

    @classmethod
    def create(cls, workspace, system_path):
        source = Path(system_path).resolve(strict=True)
        return cls._create(workspace, source.name, source.read_bytes())

    @classmethod
    def create_from_text(cls, workspace, filename, content):
        return cls._create(workspace, filename, content)

    @classmethod
    def _create(cls, workspace, filename, content):
        system, raw = parse_system(filename, content)
        root = assert_product_path(workspace)
        if root.exists():
            raise FileExistsError("Agent workspace already exists; choose a new directory to import another system")
        from philoagent.cli import FLASH, PROVIDER, source_hash
        from philoagent.orientation import bootstrap, load

        if not FLASH.is_dir():
            raise FileNotFoundError("The pinned native runtime is missing")
        root.mkdir(parents=True, exist_ok=False)
        ignore = shutil.ignore_patterns(".git", ".env", ".env.*", "__pycache__", "*.pyc", "storage", "data", ".DS_Store")
        shutil.copytree(FLASH, root / "workspace", ignore=ignore)
        # Pin read-only adapters as well: future research edits must not silently
        # change how a previously initialized application agent behaves.
        shutil.copytree(PROJECT / "philoagent", root / "adapter_snapshot" / "philoagent",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        bootstrap(root / "lineage", PROVIDER, system["items"])
        source_dir = root / "system_source"
        source_dir.mkdir()
        (source_dir / system["filename"]).write_bytes(raw)
        write_json(root / "seed_source.json", {"schema_version": 1, "items": system["items"]})
        write_json(root / "run.json", {"schema_version": 1, "seed": "imported_system",
                   "initial_provider": PROVIDER, "orientation_sha256": load(root / "lineage", PROVIDER).sha256,
                   "initial_workspace_sha256": source_hash(root / "workspace"),
                   "adapter_sha256": source_hash(PROJECT / "philoagent"),
                   "development_data_sha256": None, "development_task_count": 0})
        write_json(root / "state.json", {"provider": PROVIDER, "updated_utc": now()})
        copy_runtime(root, root / "initial")
        write_json(root / "agent.json", {"schema_version": 1, "created_utc": now(),
                   "initial_provider": PROVIDER, "system": system,
                   "adapter_sha256": source_hash(root / "adapter_snapshot" / "philoagent")})
        return cls(root)

    def _require_initialized(self):
        if not (self.workspace / "agent.json").is_file():
            raise AgentStateError("Agent is not initialized; import a philosophy system first")
        manifest = read_json(self.workspace / "agent.json")
        if manifest.get("schema_version") != 1:
            raise AgentStateError("Unsupported agent state schema")
        return manifest

    @contextmanager
    def _exclusive(self):
        self._require_initialized()
        try:
            with file_lock(self.workspace / ".agent.lock"):
                # A child may have outlived its CLI/server parent. Its own lock
                # prevents a replacement parent from competing for the same M.
                with file_lock(self.workspace / ".worker.lock"):
                    pass
                yield
        except BlockingIOError:
            raise AgentBusyError("This agent is already executing a task or evolution") from None

    def _busy(self):
        try:
            for name in (".agent.lock", ".worker.lock"):
                with file_lock(self.workspace / name):
                    pass
        except BlockingIOError:
            return True
        return False

    @staticmethod
    def _saved_record(directory):
        path = Path(directory) / "output" / "1.json"
        return read_json(path) if path.is_file() else None

    def info(self):
        if not (self.workspace / "agent.json").is_file():
            return {"initialized": False, "workspace": str(self.workspace), "busy": False, "tasks": []}
        manifest = self._require_initialized()
        state = read_json(self.workspace / "state.json")
        from philoagent.orientation import load
        snapshot = load(self.workspace / "lineage", state["provider"])
        tasks, last = [], None
        directories = sorted((self.workspace / "operations").glob("*"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        busy = self._busy()
        for directory in directories:
            if not (directory / "request.json").is_file():
                continue
            request = read_json(directory / "request.json")
            result = read_json(directory / "result.json") if (directory / "result.json").exists() else None
            record = self._saved_record(directory)
            status = (result or {}).get("status", "running" if busy else "interrupted")
            operation = {"operation_id": directory.name, "operation": request["operation"],
                         "status": status, "result": result, "record": record}
            if last is None:
                last = operation
            if request["operation"] == "run" and len(tasks) < 20:
                record = record or (result or {}).get("record") or {}
                tasks.append({"id": directory.name, **record, **operation,
                              "record_path": str(directory / "output" / "1.json")})
            if last is not None and len(tasks) >= 20:
                break
        return {"initialized": True, "workspace": str(self.workspace), "provider": state["provider"],
                "system": {**manifest["system"], "items": list(snapshot.items)},
                "initial_system": manifest["system"], "busy": busy,
                "last_operation": last, "tasks": tasks}

    def _check_learning_integrity(self):
        for path in (self.workspace / "operations").glob("*/output/1.json"):
            record = read_json(path)
            if record.get("memory_ingestion_started") and (
                    not record.get("memory_ingestion_complete") or record.get("status") != "completed"):
                raise AgentStateError("A previous memory ingestion was interrupted; preserve its records and restore memory or create a new agent before further learning")
        for path in (self.workspace / "operations").glob("*/request.json"):
            request = read_json(path)
            if request.get("operation") == "evolve":
                result = read_json(path.parent / "result.json") if (path.parent / "result.json").exists() else {}
                if result.get("status") != "completed" and (path.parent / "worker_started.json").exists():
                    raise AgentStateError("A previous evolution did not complete; inspect its native checkpoint and memory before further learning")

    def _perform(self, operation, payload, request_id=None):
        identifier = request_id or uuid.uuid4().hex
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", identifier):
            raise ValueError("request_id must contain 1 to 96 ASCII letters, digits, underscores or hyphens")
        directory = self.workspace / "operations" / identifier
        identity = digest({"operation": operation, "payload": payload})
        with self._exclusive():
            if (directory / "request.json").exists():
                old = read_json(directory / "request.json")
                if old.get("identity") != identity:
                    raise ValueError("This request_id already belongs to a different operation")
                if (directory / "result.json").exists():
                    return read_json(directory / "result.json")
                # An interrupted invocation is evidence, never an invitation to
                # repeat paid work or a possibly partial memory mutation.
                return {"status": "interrupted", "operation": operation, "operation_id": identifier,
                        "record": self._saved_record(directory), "replayed": False}
            if operation in {"run", "evolve"}:
                self._check_learning_integrity()
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "request.json", {"operation": operation, "payload": payload,
                       "operation_id": identifier, "identity": identity, "created_utc": now()})
            write_json(directory / "status.json", {"status": "running", "started_utc": now(), "pid": os.getpid()})
            try:
                if operation == "evaluate":
                    from .evaluation import evaluate
                    result = evaluate(self, payload["tasks"], directory, compare_initial=payload["compare_initial"])
                    result.update(operation=operation, operation_id=identifier)
                else:
                    result = self._invoke(operation, payload, operation_dir=directory)
            except Exception as exc:
                # Keep operator-friendly validation errors local; do not store
                # arbitrary exception strings, endpoints or credentials.
                result = {"status": "failed", "operation": operation, "operation_id": identifier,
                          "error_type": type(exc).__name__, "record": self._saved_record(directory)}
                write_json(directory / "result.json", result)
                write_json(directory / "status.json", {"status": "failed", "finished_utc": now()})
                if isinstance(exc, (ValueError, AgentStateError, AgentBusyError)):
                    raise
                return result
            write_json(directory / "result.json", result)
            write_json(directory / "status.json", {"status": result["status"], "finished_utc": now()})
            return result

    def _invoke(self, operation, payload, operation_dir=None, runtime_root=None):
        root = Path(runtime_root or self.workspace).resolve()
        directory = Path(operation_dir) if operation_dir else root / "operations" / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=True)
        request = directory / "request.json"
        if not request.exists():
            write_json(request, {"operation": operation, "operation_id": directory.name, "payload": payload})
        from philoagent.cli import environment, missing_configuration, source_hash
        values = environment(self.env_file)
        missing = missing_configuration(values, evolving=operation == "evolve")
        if missing:
            raise ValueError("Missing configuration: " + ", ".join(missing))
        manifest = self._require_initialized()
        adapters = self.workspace / "adapter_snapshot"
        if source_hash(adapters / "philoagent") != manifest["adapter_sha256"]:
            raise AgentStateError("This agent's frozen runtime adapters changed")
        # Explicitly overwrite any inherited formal-experiment runtime routing.
        for name in list(values):
            if name.startswith("PHILO_"):
                values.pop(name)
        values.update(PHILO_LINEAGE_ROOT=str(root / "lineage"),
                      PHILO_EVIDENCE_DIR=str(root / "evidence" / "runtime"),
                      PHILOAGENT_APP_BUDGET_ROOT=str(self.workspace / "budget"),
                      PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                      PYTHONPATH=os.pathsep.join((str(adapters), str(PROJECT))))
        command = [sys.executable, "-u", "-m", "philoagent_app.worker", "--workspace", str(root), "--request", str(request)]
        with (directory / "worker.log").open("ab") as stream:
            # Python prepends cwd ahead of PYTHONPATH for ``-m``. Using PROJECT
            # here would silently load the live research adapters first.
            completed = subprocess.run(command, cwd=adapters, env=values, stdout=stream, stderr=subprocess.STDOUT)
        result_path = directory / "result.json"
        if result_path.exists():
            return read_json(result_path)
        return {"status": "interrupted", "operation": operation, "operation_id": directory.name,
                "error_type": "WorkerExitedWithoutResult", "exit_code": completed.returncode,
                "record": self._saved_record(directory)}

    def run(self, task, *, request_id=None):
        task = normalize_task(task)
        if task.get("split") in {"evaluation", "heldout", "holdout", "test"}:
            raise ValueError("Evaluation tasks must not enter persistent learning")
        return self._perform("run", {"task": task}, request_id)

    def evolve(self, tasks, *, rounds=1, request_id=None):
        tasks = read_tasks(tasks)
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if len(tasks) < 4 * rounds:
            raise ValueError("Each native evolution round needs at least four development tasks")
        if any(task.get("split") in {"evaluation", "heldout", "holdout", "test"} for task in tasks):
            raise ValueError("Evaluation tasks must not enter evolution")
        return self._perform("evolve", {"tasks": tasks, "rounds": rounds}, request_id)

    def evaluate(self, tasks, *, compare_initial=True, request_id=None):
        if type(compare_initial) is not bool:
            raise ValueError("compare_initial must be a boolean")
        return self._perform("evaluate", {"tasks": read_tasks(tasks), "compare_initial": compare_initial}, request_id)
