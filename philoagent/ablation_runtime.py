"""Native WebWalkerQA execution and SDK-boundary evidence for isolated jobs.

The experiment manager must call ``run_native_job`` in a fresh process. This
module deliberately does not implement candidate selection or an evolution loop:
both remain the upstream AutoEvolver's responsibility.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from functools import partial, wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from .orientation import assert_unchanged, load


_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("philo_ablation_call", default={})


def digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def items_digest(items) -> str:
    """Same semantic identity as ablation_design; independent of lineage IDs."""
    return hashlib.sha256(json.dumps(list(items), ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def audit_scope(**details):
    token = _CONTEXT.set({**_CONTEXT.get(), **details})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class AuditedRequestError(RuntimeError):
    """A provider request failed; exception text containing credentials is omitted."""


class EmptyModelContentError(AuditedRequestError):
    pass


class AuditSink:
    def __init__(self, path, lineage_root, *, model, max_tokens=8192,
                 generation_tokens=16000, timeout=120, thinking="disabled"):
        self.path = Path(path)
        self.lineage_root = Path(lineage_root)
        self.model = model
        self.max_tokens = max_tokens
        self.generation_tokens = generation_tokens
        self.timeout = timeout
        self.thinking = thinking
        self.lock = threading.Lock()
        self.events = []

    def append(self, event):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self.events.append(event)

    def orientation(self, context):
        provider = context.get("provider")
        if not provider:
            return ()
        return load(self.lineage_root, provider).items

    def summary(self):
        # Include previous attempts when resuming the same immutable job.
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()] if self.path.exists() else []
        roles = {}
        for event in events:
            role = roles.setdefault(event.get("role", "unscoped"), {
                "requests": 0, "successful_requests": 0, "failed_requests": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "orientation_present_requests": 0,
            })
            role["requests"] += 1
            role["successful_requests" if event["status"] == "success" else "failed_requests"] += 1
            role["orientation_present_requests"] += int(event.get("orientation_present_in_request", False))
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                role[name] += (event.get("usage") or {}).get(name, 0) or 0
        return {"request_count": len(events), "roles": roles,
                "failed_request_count": sum(event["status"] != "success" for event in events)}


class AuditedClient:
    """Drop-in chat client preserving audit state through SDK with_options calls."""

    def __init__(self, client, sink: AuditSink, context=None):
        self.client = client
        self.sink = sink
        self.context = dict(_CONTEXT.get() if context is None else context)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __getattr__(self, name):
        return getattr(self.client, name)

    def with_options(self, **kwargs):
        kwargs["timeout"] = self.sink.timeout
        kwargs["max_retries"] = 0
        return type(self)(self.client.with_options(**kwargs), self.sink, self.context)

    copy = with_options

    def create(self, **kwargs):
        context = {**self.context, **_CONTEXT.get()}
        context.setdefault("role", "unscoped")
        items = self.sink.orientation(context)
        messages = kwargs.get("messages", [])
        # Search JSON's decoded message strings, not escaped JSON (Chinese and
        # quotes inside a principle must not create false negative evidence).
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for part in value.values():
                    yield from strings(part)
            elif isinstance(value, (tuple, list)):
                for part in value:
                    yield from strings(part)
        prompt = "\n".join(strings(messages))
        budget = self.sink.generation_tokens if context["role"] == "generation" else self.sink.max_tokens
        requested = [kwargs.pop(name) for name in ("max_tokens", "max_completion_tokens") if name in kwargs]
        limits = [value for value in requested if isinstance(value, int) and not isinstance(value, bool) and value > 0]
        kwargs["max_tokens"] = min([budget, *limits])
        requested_model = kwargs.get("model")
        kwargs["model"] = self.sink.model
        extra = dict(kwargs.get("extra_body") or {})
        if self.sink.model.lower().startswith("deepseek"):
            extra["thinking"] = {"type": self.sink.thinking}
            kwargs["extra_body"] = extra
        # A caller cannot raise the frozen job's per-request timeout or retries.
        kwargs.pop("max_retries", None)
        kwargs["timeout"] = self.sink.timeout
        event = {**context, "model": self.sink.model, "requested_model": requested_model,
                 "prompt_sha256": digest(messages), "request_sha256": digest(kwargs),
                 "orientation_items_sha256": items_digest(items), "orientation_item_count": len(items),
                 "orientation_present_in_request": bool(items) and all(item in prompt for item in items),
                 "max_tokens": kwargs["max_tokens"], "thinking": self.sink.thinking,
                 "timeout_seconds": self.sink.timeout, "sdk_max_retries": 0,
                 "status": "started", "status_scope": "sdk_request_and_output"}
        started = time.monotonic()
        try:
            result = self.client.chat.completions.create(**kwargs)
            choices = getattr(result, "choices", [])
            has_content = any(bool(getattr(choice.message, "content", None)
                                   and str(choice.message.content).strip()) for choice in choices)
            has_tools = any(bool(getattr(choice.message, "tool_calls", None)) for choice in choices)
            usage = getattr(result, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif usage is not None and not isinstance(usage, dict):
                usage = {name: getattr(usage, name, 0) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
            event.update(usage=usage, response_id=getattr(result, "id", None),
                         response_has_content=has_content, response_has_tool_calls=has_tools,
                         finish_reason=[getattr(choice, "finish_reason", None) for choice in choices],
                         content_sha256=digest([choice.message.model_dump() if hasattr(choice.message, "model_dump")
                                                else vars(choice.message) for choice in choices]))
            if not has_content and not has_tools:
                raise EmptyModelContentError("Model response contains no assistant text or tool calls")
            event["status"] = "success"
            return result
        except Exception as exc:
            event.update(status="error", error_type=type(exc).__name__)
            if isinstance(exc, AuditedRequestError):
                raise
            raise AuditedRequestError(f"SDK request failed ({type(exc).__name__}); inspect API audit") from None
        finally:
            event["duration_seconds"] = round(time.monotonic() - started, 3)
            self.sink.append(event)


def _provider_artifact(run_dir: Path, workspace: Path, provider: str) -> dict:
    from EvolveLab.memory_types import MemoryType, PROVIDER_MAPPING
    memory_type = MemoryType(provider)
    class_name, module_name = PROVIDER_MAPPING[memory_type]
    module = importlib.import_module("EvolveLab.providers." + module_name)
    source = Path(module.__file__).resolve(strict=True)
    if workspace not in source.parents:
        raise ValueError("Provider was imported from outside this job workspace")
    actual = digest(source.read_bytes())
    loaded = getattr(module, "_philo_loaded_code_sha256", actual)
    if actual != loaded:
        raise ValueError("Already imported provider source changed")
    snapshot = load(run_dir / "lineage", provider)
    return {"provider": provider, "module": module_name, "class_name": class_name,
            "source_path": str(source), "code_sha256": actual,
            "orientation_snapshot": snapshot.to_dict(), "orientation_items_sha256": items_digest(snapshot.items)}


def reset_storage(workspace: Path, provider: str, config: dict) -> dict:
    """Verify the exact effective target before recursively resetting a provider."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", provider):
        raise ValueError("Unsafe provider identifier")
    if not isinstance(config.get("storage_dir"), str) or not config["storage_dir"]:
        raise ValueError("Fair memory reset requires an explicit effective storage_dir")
    workspace = workspace.resolve(strict=True)
    expected = (workspace / "storage" / provider).resolve()
    path = Path(config["storage_dir"])
    target = (workspace / path).resolve() if not path.is_absolute() else path.resolve()
    if target != expected or workspace not in target.parents or target.parent != workspace / "storage":
        raise ValueError("Effective provider storage_dir must be workspace/storage/provider")
    # Extra provider paths must not retain a second memory store between stages.
    for key, value in config.items():
        if not isinstance(value, str) or not (key.endswith("_path") or key.endswith("_dir")):
            continue
        candidate = Path(value)
        candidate = (workspace / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if candidate != target and target not in candidate.parents:
            raise ValueError(f"Provider configuration path escapes its independent storage: {key}")
    before = []
    if target.exists():
        before = [{"path": item.relative_to(target).as_posix(), "sha256": digest(item.read_bytes())}
                  for item in sorted(target.rglob("*")) if item.is_file()]
        # target has just been resolved and checked against the explicit workspace.
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return {"storage_dir": str(target), "previous_file_count": len(before),
            "previous_storage_sha256": digest(before), "reset": True}


def collect_completed_development(evolution_root: Path, *, before_round=None) -> list[dict]:
    """Freeze prior discovery and each round winner's newly encountered task.

    Losing candidates and repeated finalist tasks never supply warmup records.
    """
    records = []
    seen = set()
    for summary_path in sorted(Path(evolution_root).glob("round_*/round_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if before_round is not None and summary["round"] >= before_round:
            continue
        root = summary_path.parent
        batch = set(summary["batch_tasks"])
        sources = [(root / "base_logs", batch)]
        new = set(summary["finalist_tasks"]) - batch
        sources.append((root / "eval_finalists" / summary["winner"], new))
        for directory, indices in sources:
            for path in sorted(directory.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                index = data.get("global_task_index", data.get("item_index"))
                if index in indices and index not in seen:
                    seen.add(index)
                    records.append({"source": str(path.resolve()), "source_sha256": digest(path.read_bytes()),
                                    "global_task_index": index, "data": data})
    return sorted(records, key=lambda item: item["global_task_index"])


def _task_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append({"source": str(path.resolve()), "source_sha256": digest(path.read_bytes()),
                        "global_task_index": data["global_task_index"], "data": data})
    return records


class NativeTaskRunner:
    """The native AutoEvolver callback, with matched pre-stage memory history."""

    def __init__(self, run_dir, workspace, protocol, model_config, *, runner=None, native=None):
        self.run_dir = Path(run_dir)
        self.workspace = Path(workspace)
        self.protocol = protocol
        self.model_config = model_config
        self.runner = runner or importlib.import_module("run_flash_searcher_webwalkerqa")
        self.native = native or importlib.import_module("FlashOAgents")
        self.evolution_root = self.run_dir / "evolution"
        self.locks = {}
        self.locks_lock = threading.Lock()
        self.heldout_warmup = None
        self.stage_events = []

    def _lock(self, provider):
        with self.locks_lock:
            return self.locks.setdefault(provider, threading.Lock())

    def warmup_for(self, output_dir):
        output_dir = Path(output_dir)
        if output_dir == self.run_dir / "terminal" / "task_logs":
            if self.heldout_warmup is None:
                raise ValueError("Terminal warmup must be fixed before opening heldout data")
            return self.heldout_warmup, "heldout"
        round_root = next((item for item in [output_dir, *output_dir.parents]
                           if re.fullmatch(r"round_\d+", item.name)), None)
        if round_root is None:
            raise ValueError("Callback output must identify its native round")
        round_number = int(round_root.name.split("_")[1])
        history = collect_completed_development(self.evolution_root, before_round=round_number)
        if output_dir.name == "base_logs":
            return history, "discovery"
        history.extend(_task_records(round_root / "base_logs"))
        # Each comparison receives exactly the same fixed list, including order.
        return history, "finalists" if output_dir.parent.name == "eval_finalists" else "batch_comparison"

    def _rebuild_memory(self, provider, warmup):
        from EvolveLab.config import get_memory_config
        from EvolveLab.memory_types import MemoryType, TrajectoryData
        reset = reset_storage(self.workspace, provider, get_memory_config(MemoryType(provider)))
        events = []
        for record in warmup:
            data = record["data"]
            if data.get("status") != "success" or data.get("judgement") not in ("correct", "incorrect"):
                raise RuntimeError("Only completed development records may warm up memory")
            # Loading afresh for each warmup task prevents hidden short-term
            # task state from carrying between trajectories.
            with audit_scope(role="warmup_ingestion", warmup_task=record["global_task_index"]):
                model = self.native.OpenAIServerModel(**self.model_config)
                instance = self.runner.load_memory_provider(provider, model)
                success, message = instance.take_in_memory(TrajectoryData(
                    query=data["question"], trajectory=data.get("agent_messages", []),
                    result=data.get("agent_result"), metadata={
                        "status": "success", "is_correct": data["judgement"] == "correct",
                        "item_index": record["global_task_index"], "domain": data.get("domain", ""),
                        "full_query": data.get("enhanced_question", data["question"]),
                        "warmup": True,
                    }))
                if not success:
                    raise RuntimeError("Native warmup ingestion returned false")
                events.append({key: value for key, value in record.items() if key != "data"})
        return {**reset, "records": events, "warmup_sha256": digest(events),
                "native_cold_start": "provider_defined_may_differ_between_architectures"}

    def __call__(self, dataset_name, provider_name, task_indices, dataset_file, output_dir):
        output_dir = Path(output_dir).resolve()
        if dataset_name != "webwalkerqa":
            raise ValueError("This locked experiment only supports WebWalkerQA")
        with self._lock(provider_name):
            warmup, stage = self.warmup_for(output_dir)
            # Heldout content is read only on this final callback, after the
            # winning provider and warmup manifest have been committed.
            rows = [json.loads(line) for line in Path(dataset_file).read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(index < 0 or index >= len(rows) for index in task_indices):
                raise ValueError("Native task index outside locked dataset")
            artifact = _provider_artifact(self.run_dir, self.workspace, provider_name)
            stage_id = digest(str(output_dir))[:16]
            event = {"output_dir": str(output_dir), "stage": stage, "provider": provider_name,
                     "task_indices": list(task_indices), **artifact, "status": "started"}
            path = self.run_dir / "evidence" / "stages" / f"{stage_id}.json"
            with audit_scope(provider=provider_name, stage=stage, stage_id=stage_id,
                             code_sha256=artifact["code_sha256"],
                             orientation_items_sha256=artifact["orientation_items_sha256"]):
                try:
                    event["warmup"] = self._rebuild_memory(provider_name, warmup)
                    write_json(path, event)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Callback retries must not accidentally aggregate stale logs.
                    for stale in output_dir.glob("*.json"):
                        if not stale.stem.isdigit():
                            raise ValueError("Unexpected file in native task-log directory")
                        stale.unlink()
                    correct = 0
                    for index in task_indices:
                        snapshot = load(self.run_dir / "lineage", provider_name)
                        with audit_scope(role="actor", task_index=index, task_id=rows[index].get("task_id")):
                            result = self.runner.process_item(
                                rows[index], self.model_config, summary_interval=None, prompts_type="default",
                                max_steps=self.protocol.get("max_steps", 6), memory_type_str=provider_name,
                                item_index=index, enable_memory_evolution=stage != "heldout",
                                judge_model=self.model_config["model_id"],
                            )
                        assert_unchanged(snapshot)
                        if not isinstance(result, dict) or result.get("error") or result.get("status") != "success":
                            raise RuntimeError("Native task failed; it is not an incorrect-answer observation")
                        if result.get("judgement") not in ("correct", "incorrect"):
                            raise RuntimeError("Native judge failed; it is not an incorrect-answer observation")
                        actual_artifact = _provider_artifact(self.run_dir, self.workspace, provider_name)
                        if actual_artifact != artifact:
                            raise RuntimeError("Provider implementation or philosophy changed within a stage")
                        result.update(global_task_index=index, task_id=rows[index].get("task_id"),
                                      runtime_artifact=artifact, memory_evolution_enabled=stage != "heldout")
                        write_json(output_dir / f"{index + 1}.json", result)
                        correct += result["judgement"] == "correct"
                    event.update(status="success", task_count=len(task_indices), correct_count=correct)
                    self.stage_events.append(event)
                    return output_dir
                except Exception as exc:
                    event.update(status="error", error_type=type(exc).__name__)
                    raise
                finally:
                    write_json(path, event)


def refresh_registry(workspace, runner):
    """Refresh mutable native registrations once, before comparison threads start."""
    workspace = Path(workspace).resolve()
    for name in list(sys.modules):
        if name == "EvolveLab" or name.startswith("EvolveLab."):
            del sys.modules[name]
    importlib.invalidate_caches()
    types = importlib.import_module("EvolveLab.memory_types")
    config = importlib.import_module("EvolveLab.config")
    if workspace not in Path(types.__file__).resolve().parents:
        raise RuntimeError("Native registry imported from another workspace")
    for name in ("MemoryType", "TrajectoryData", "PROVIDER_MAPPING"):
        setattr(runner, name, getattr(types, name))
    runner.get_memory_config = config.get_memory_config


def _scoped_method(original, role, provider_getter=None):
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        provider = provider_getter(self, args, kwargs) if provider_getter else _CONTEXT.get().get("provider")
        with audit_scope(role=role, **({"provider": provider} if provider else {})):
            return original(self, *args, **kwargs)
    return wrapped


def install_audit(stack: ExitStack, sink: AuditSink, native, runner):
    """Patch actual SDK construction and role boundaries, preserving native logic."""
    import openai
    from MemEvolve.core import memory_evolver
    from MemEvolve.phases import phase_analyzer, phase_generator, phase_validator
    from MemEvolve.utils import trajectory_tools
    from . import runtime

    original_openai = openai.OpenAI

    def factory(*args, **kwargs):
        kwargs.update(timeout=sink.timeout, max_retries=0)
        return AuditedClient(original_openai(*args, **kwargs), sink)

    stack.enter_context(patch.object(openai, "OpenAI", factory))
    if getattr(openai, "Client", None) is original_openai:
        stack.enter_context(patch.object(openai, "Client", factory))
    # Imported-by-name bindings created before patching must also use the proxy.
    for module in (memory_evolver, phase_generator, trajectory_tools):
        stack.enter_context(patch.object(module, "OpenAI", factory))
    stack.enter_context(patch.object(phase_analyzer, "AnalysisAgent", partial(
        phase_analyzer.AnalysisAgent, summary_model_id=sink.model)))

    def provider(self, args, kwargs):
        return getattr(self, "default_provider", _CONTEXT.get().get("provider"))

    for cls, method, role, getter in (
        (phase_analyzer.PhaseAnalyzer, "__init__", "analysis", lambda self, args, kw: kw.get("default_provider", args[2] if len(args) > 2 else "agent_kb")),
        (phase_analyzer.PhaseAnalyzer, "run_analysis", "analysis", provider),
        (phase_generator.PhaseGenerator, "run_generation", "generation", provider),
        (trajectory_tools.TrajectoryViewerTool, "_generate_trajectory_summary", "trajectory_summary", None),
        (runtime.OrientedProvider, "provide_memory", "memory_retrieval", lambda self, args, kw: self._snapshot.provider_id),
        (runtime.OrientedProvider, "take_in_memory", "memory_ingestion", lambda self, args, kw: self._snapshot.provider_id),
    ):
        original = getattr(cls, method)
        # Warmup ingestion keeps its distinct role for cost attribution.
        if cls is runtime.OrientedProvider and method == "take_in_memory":
            def ingestion(self, *args, _original=original, **kwargs):
                role = "warmup_ingestion" if _CONTEXT.get().get("role") == "warmup_ingestion" else "memory_ingestion"
                with audit_scope(role=role, provider=self._snapshot.provider_id):
                    return _original(self, *args, **kwargs)
            stack.enter_context(patch.object(cls, method, ingestion))
        else:
            stack.enter_context(patch.object(cls, method, _scoped_method(original, role, getter)))

    original_judge = runner.judge_webwalkerqa_answer
    def judge(*args, **kwargs):
        with audit_scope(role="judge"):
            result = original_judge(*args, **kwargs)
        if result.get("judgement") not in ("correct", "incorrect"):
            raise RuntimeError("Native judge did not return a usable judgement")
        return result
    stack.enter_context(patch.object(runner, "judge_webwalkerqa_answer", judge))

    original_validator_init = phase_validator.PhaseValidator.__init__
    signature = inspect.signature(original_validator_init)
    def validator_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.arguments.update(enable_auto_fix=False, max_fix_attempts=1)
        return original_validator_init(*bound.args, **bound.kwargs)
    stack.enter_context(patch.object(phase_validator.PhaseValidator, "__init__", validator_init))
    original_validation = phase_validator.PhaseValidator.run_validation
    def validation(self, created_systems):
        provider_id = created_systems[0] if len(created_systems) == 1 else _CONTEXT.get().get("provider")
        with audit_scope(role="validator", provider=provider_id, stage="validation"):
            return original_validation(self, created_systems)
    stack.enter_context(patch.object(phase_validator.PhaseValidator, "run_validation", validation))


def _round_changes(run_dir, workspace):
    results = []
    for path in sorted((run_dir / "evolution").glob("round_*/round_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        parent = _provider_artifact(run_dir, workspace, summary["best_provider_before"])
        winner = _provider_artifact(run_dir, workspace, summary["winner"])
        candidates = [_provider_artifact(run_dir, workspace, provider)
                      for provider in summary["evolution"].get("validated_systems", [])]
        results.append({"round": summary["round"], "parent": parent, "winner": winner,
                        "winner_code_changed": parent["code_sha256"] != winner["code_sha256"],
                        "winner_orientation_items_changed": parent["orientation_items_sha256"] != winner["orientation_items_sha256"],
                        "candidates": [{**candidate,
                            "code_changed_vs_parent": candidate["code_sha256"] != parent["code_sha256"],
                            "orientation_items_changed_vs_parent": candidate["orientation_items_sha256"] != parent["orientation_items_sha256"],
                        } for candidate in candidates]})
    return results


def _loaded_evidence(run_dir, sink):
    """Use the real provider hook, rather than an imported source file, as proof."""
    records = {}
    for path in sorted((run_dir / "evidence" / "runtime").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            provider = event.get("details", {}).get("provider")
            if not provider or not event.get("disk_matches_loaded_code"):
                continue
            record = records.setdefault(provider, {"provider": provider, "code_sha256": event["loaded_code_sha256"],
                                                  "operations": {}, "orientation_sent_to_actor": False})
            operation = event["operation"]
            record["operations"][operation] = record["operations"].get(operation, 0) + 1
    calls = [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()] if sink.path.exists() else []
    for call in calls:
        if call.get("role") == "actor" and call.get("status") == "success" and call.get("orientation_present_in_request"):
            provider = call.get("provider")
            if provider in records:
                records[provider]["orientation_sent_to_actor"] = True
    return list(records.values())


def run_native_job(run_dir, job, protocol, dev_path, heldout_path, *, api_env=None) -> dict:
    """Execute one independent native job; credentials never enter its evidence."""
    from .cli import environment, PROVIDER
    from .live_smoke import activate_workspace

    run_dir = Path(run_dir).resolve(strict=True)
    dev_path = Path(dev_path).resolve(strict=True)
    # Do not resolve(strict=True), stat, hash, or open heldout contents yet.
    heldout_path = Path(heldout_path).absolute()
    if run_dir in heldout_path.parents:
        raise ValueError("Heldout must remain outside the job directory")
    values = dict(environment() if api_env is None else api_env)
    if not values.get("OPENAI_API_KEY") or not values.get("DEFAULT_MODEL"):
        raise ValueError("DEFAULT_MODEL and OPENAI_API_KEY are required")
    values["PHILO_SMOKE_THINKING"] = protocol.get("thinking", "disabled")
    values["PYTHONDONTWRITEBYTECODE"] = "1"
    values["DEFAULT_JUDGE_MODEL"] = values["DEFAULT_MODEL"]
    values["ANALYSIS_MODEL"] = values["GENERATION_MODEL"] = values["DEFAULT_MODEL"]
    os.environ.update({key: str(value) for key, value in values.items() if value is not None})
    sys.dont_write_bytecode = True
    workspace = activate_workspace(run_dir)
    os.environ["PHILO_EVIDENCE_DIR"] = str(run_dir / "evidence" / "runtime")

    import FlashOAgents as native
    import run_flash_searcher_webwalkerqa as runner
    from MemEvolve.core import auto_evolver

    # Policies are installed only in the fresh isolated workspace process.
    from .ablation_policy import install_native_policy
    policy = install_native_policy(run_dir, job)
    sink = AuditSink(run_dir / "evidence" / "api_calls.jsonl", run_dir / "lineage",
                     model=values["DEFAULT_MODEL"], thinking=protocol.get("thinking", "disabled"),
                     timeout=protocol.get("api_timeout", 120), max_tokens=protocol.get("max_tokens", 8192),
                     generation_tokens=protocol.get("generation_max_tokens", 16000))
    model_config = {"model_id": values["DEFAULT_MODEL"], "api_base": values.get("OPENAI_API_BASE"),
                    "api_key": values["OPENAI_API_KEY"], "temperature": 0, "max_tokens": sink.max_tokens,
                    "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"}}
    callback = NativeTaskRunner(run_dir, workspace, protocol, model_config, runner=runner, native=native)
    status_path = run_dir / "runtime_summary.json"
    with ExitStack() as stack:
        install_audit(stack, sink, native, runner)
        # Native run() otherwise reseeds all repetitions to its module constant.
        stack.enter_context(patch.object(auto_evolver, "EVOLVE_RANDOM_SEED", int(job.get("replicate", job.get("rep", 0))) + 20261029))
        original_parallel = auto_evolver.AutoEvolver._evaluate_providers_parallel
        def parallel(self, *args, **kwargs):
            refresh_registry(workspace, runner)
            return original_parallel(self, *args, **kwargs)
        stack.enter_context(patch.object(auto_evolver.AutoEvolver, "_evaluate_providers_parallel", parallel))
        original_evolve = auto_evolver.AutoEvolver._run_memory_evolver
        def evolving(self, round_dir, checkpoint=None, current_provider=None):
            with audit_scope(provider=current_provider or self.state.get("best_provider", PROVIDER), stage="outer_evolution"):
                return original_evolve(self, round_dir, checkpoint, current_provider)
        stack.enter_context(patch.object(auto_evolver.AutoEvolver, "_run_memory_evolver", evolving))

        rounds = protocol.get("rounds", 2)
        auto = auto_evolver.AutoEvolver(
            analysis_model_id=values["DEFAULT_MODEL"], gen_model_id=values["DEFAULT_MODEL"],
            work_root=str(run_dir / "evolution"), dataset_name="webwalkerqa", run_provider=callback,
            default_provider=PROVIDER, num_systems=protocol.get("candidates", protocol.get("num_systems", 1)),
            creativity_index=protocol.get("creativity_index", 0.2),
            task_batch_x=protocol.get("batch_size", protocol.get("task_batch_x", 1)),
            top_t=protocol.get("top_t", 2), extra_sample_y=protocol.get("extra_sample_y", 1),
            datasets_config={"webwalkerqa": str(dev_path)}, max_workers=2,
            clear_storage_per_round=False,
        )
        try:
            refresh_registry(workspace, runner)
            remaining = rounds - auto.state.get("round", 0)
            if remaining > 0:
                auto.run(num_rounds=remaining)
            if auto.state.get("round") != rounds:
                raise RuntimeError("Native evolution did not complete the locked number of rounds")
            refresh_registry(workspace, runner)
            winner = auto.state["best_provider"]
            warmup = collect_completed_development(run_dir / "evolution")
            if len(warmup) != protocol.get("development_task_count", rounds * 2):
                raise RuntimeError("Terminal warmup does not cover the complete locked development stream")
            warmup_manifest = [{key: value for key, value in item.items() if key != "data"} for item in warmup]
            terminal = run_dir / "terminal"
            write_json(terminal / "winner_before_heldout.json", _provider_artifact(run_dir, workspace, winner))
            write_json(terminal / "warmup_manifest.json", {"records": warmup_manifest,
                        "sha256": digest(warmup_manifest), "heldout_ingestion": False})
            callback.heldout_warmup = warmup
            # A distinct evaluator thread owns access to heldout contents. No
            # heldout record is supplied to native generation or analysis.
            def evaluate_heldout():
                rows = [json.loads(line) for line in heldout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(rows) != protocol.get("heldout_task_count", 4):
                    raise ValueError("Heldout task count differs from locked protocol")
                return callback("webwalkerqa", winner, list(range(len(rows))), heldout_path, terminal / "task_logs")
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="heldout-evaluator") as executor:
                logs = executor.submit(evaluate_heldout).result()
            from MemEvolve.utils.trajectory_tools import TrajectoryFeedbackAggregator
            terminal_score = TrajectoryFeedbackAggregator(str(logs)).aggregate()
            usage = sink.summary()
            active_errors = sum(event["status"] != "success" for event in sink.events)
            usage["active_attempt_failed_request_count"] = active_errors
            usage["historical_failed_request_count"] = usage["failed_request_count"] - active_errors
            changes = _round_changes(run_dir, workspace)
            candidates = [candidate for entry in changes for candidate in entry["candidates"]]
            loaded_evidence = _loaded_evidence(run_dir, sink)
            loaded_by_provider = {entry["provider"]: entry for entry in loaded_evidence}
            for candidate in candidates:
                observed = loaded_by_provider.get(candidate["provider"], {})
                candidate["actual_loaded"] = (observed.get("code_sha256") == candidate["code_sha256"]
                                              and bool(observed.get("operations", {}).get("provide_memory")))
                candidate["orientation_sent_to_actor"] = observed.get("orientation_sent_to_actor", False)
            code_frozen_ok = bool(job.get("code_evolve")) or all(not candidate["code_changed_vs_parent"] for candidate in candidates)
            phi_frozen_ok = bool(job.get("orientation_evolve")) or all(not candidate["orientation_items_changed_vs_parent"] for candidate in candidates)
            heldout_records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(logs).glob("*.json"))]
            summary = {"status": "complete", "job": dict(job), "protocol": dict(protocol),
                       "winner": winner, "winner_provider": winner, "terminal": terminal_score,
                       "rounds_completed": auto.state["round"], "heldout_count": len(heldout_records),
                       "heldout_accuracy": terminal_score["summary"]["accuracy"],
                       "heldout": terminal_score["summary"], "rounds": changes, "api_usage": usage,
                       "candidate_code_modified": any(candidate["code_changed_vs_parent"] for candidate in candidates),
                       "candidate_orientation_modified": any(candidate["orientation_items_changed_vs_parent"] for candidate in candidates),
                       "winner_code_modified": any(entry["winner_code_changed"] for entry in changes),
                       "winner_orientation_modified": any(entry["winner_orientation_items_changed"] for entry in changes),
                       "actual_loaded": bool(loaded_by_provider.get(winner, {}).get("operations", {}).get("provide_memory")),
                       "loaded_provider_evidence": loaded_evidence,
                       "factor_checks": {"frozen_code_preserved": code_frozen_ok,
                                         "frozen_orientation_preserved": phi_frozen_ok},
                       "history_paths": [str(run_dir / "evolution" / "auto_history.json"),
                                         *[str(path) for path in sorted((run_dir / "evolution").glob("round_*/round_summary.json"))]],
                       "ordinary_memory_learning": True, "heldout_ingestion": False,
                       "comparison_memory_policy": "reset_then_identical_development_trajectory_warmup",
                       "internal_cold_start": "native_provider_defined_may_differ_between_architectures"}
            if not code_frozen_ok or not phi_frozen_ok:
                raise RuntimeError("A frozen experimental factor changed in a candidate")
            if active_errors:
                raise RuntimeError("At least one API request failed; incomplete observations cannot be reported as a complete job")
            write_json(status_path, summary)
            return summary
        except Exception as exc:
            summary = {"status": "failed", "job": dict(job), "error_type": type(exc).__name__,
                       "completed_native_rounds": auto.state.get("round", 0), "api_usage": sink.summary()}
            write_json(status_path, summary)
            raise
        finally:
            if hasattr(policy, "close"):
                policy.close()
            elif callable(policy):
                policy()
