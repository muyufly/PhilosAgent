"""Fresh-process native evolution worker for the v0.0.6 HSS experiment.

AutoEvolver owns the schedule, candidates, checkpoints and selection. This
adapter supplies open-writing tasks and their continuous fitness, and freezes
the trained memory separately for each terminal evaluation task.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from functools import partial, wraps
import json
import math
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from .ablation_runtime import (
    _loaded_evidence, _provider_artifact, _round_changes, audit_scope,
    digest, refresh_registry, write_json,
)
from .hss_evaluation import native_quality_summary


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


class ContinuousQualityAggregator:
    """Drop-in aggregation at the native comparison boundary, never score > 0."""

    def __init__(self, logs_dir):
        self.logs_dir = Path(logs_dir)

    def aggregate(self):
        value = json.loads((self.logs_dir / "_quality" / "summary.json").read_text(encoding="utf-8"))
        summary = value.get("summary", {})
        fitness = summary.get("accuracy")
        if (value.get("success") is not True
                or summary.get("fitness_semantics") != "mean_continuous_quality_divided_by_100"
                or isinstance(fitness, bool) or not isinstance(fitness, (int, float))
                or not math.isfinite(fitness) or not 0 <= fitness <= 1):
            raise ValueError("Native comparison requires a complete continuous quality batch")
        return value


def _tree_manifest(path):
    path = Path(path)
    return [{"path": item.relative_to(path).as_posix(), "sha256": digest(item.read_bytes())}
            for item in sorted(path.rglob("*")) if item.is_file()] if path.exists() else []


def _remap_storage_config(value, workspace, snapshot_root):
    """Preserve provider settings while rebasing every declared storage path."""
    if isinstance(value, dict):
        return {key: _remap_storage_config(item, workspace, snapshot_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_storage_config(item, workspace, snapshot_root) for item in value]
    if not isinstance(value, str):
        return value
    candidate = Path(value.replace("\\", "/"))
    if not candidate.is_absolute() and (not candidate.parts or candidate.parts[0] != "storage"):
        return value
    resolved = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
    storage = (workspace / "storage").resolve()
    if resolved.is_relative_to(storage):
        return str(snapshot_root / "storage" / resolved.relative_to(storage))
    return value


@contextmanager
def _terminal_storage(workspace, frozen_storage, snapshot_root, runner):
    """One task gets its own copy; retrieval-time writes cannot train the next."""
    workspace, frozen_storage, snapshot_root = map(Path, (workspace, frozen_storage, snapshot_root))
    if snapshot_root.exists():
        raise FileExistsError("Terminal task snapshot already exists; do not silently replay it")
    snapshot_root.mkdir(parents=True)
    shutil.copytree(frozen_storage, snapshot_root / "storage")
    original_config = runner.get_memory_config
    previous = Path.cwd()

    def get_config(memory_type):
        return _remap_storage_config(deepcopy(original_config(memory_type)), workspace, snapshot_root)

    try:
        os.chdir(snapshot_root)
        with patch.object(runner, "get_memory_config", get_config):
            yield
    finally:
        os.chdir(previous)


def _install_native_adapters(stack, workspace, runner, protocol):
    """Restore normal native capacity and pass quality evidence to analysis."""
    from MemEvolve.core import auto_evolver
    from MemEvolve.core.memory_evolver import MemoryEvolver
    from MemEvolve.phases import phase_analyzer, phase_generator, phase_validator
    from MemEvolve.validators.swe_agent_validator import SWEAgentValidator
    from MemEvolve.utils.trajectory_tools import TrajectoryViewerTool
    from .runtime import OrientedProvider
    from .hss_tool_calls import install_tool_decoding
    from .hss_creation import guard_creation
    from .hss_comparison import guard_comparison
    from .hss_memory_format import guard_memory_response
    import FlashOAgents as native

    install_tool_decoding(stack, native)
    stack.enter_context(patch.object(OrientedProvider, "provide_memory",
                                    guard_memory_response(OrientedProvider.provide_memory)))
    stack.enter_context(patch.object(MemoryEvolver, "create", guard_creation(MemoryEvolver.create)))

    stack.enter_context(patch.object(auto_evolver, "TrajectoryFeedbackAggregator", ContinuousQualityAggregator))
    original_parallel = auto_evolver.AutoEvolver._evaluate_providers_parallel

    @wraps(original_parallel)
    def parallel(self, *args, **kwargs):
        refresh_registry(workspace, runner)
        result = original_parallel(self, *args, **kwargs)
        # Native sets success=True after aggregation; reinstate the adapter's
        # explicit completeness result and retain its continuous metric.
        for record in result.values():
            if record.get("success"):
                quality = ContinuousQualityAggregator(record["logs_dir"]).aggregate()
                record.update(quality)
        return result

    stack.enter_context(patch.object(auto_evolver.AutoEvolver, "_evaluate_providers_parallel", guard_comparison(parallel)))
    original_init = phase_analyzer.PhaseAnalyzer.__init__

    @wraps(original_init)
    def analyzer_init(self, *args, **kwargs):
        with audit_scope(role="analysis"):
            original_init(self, *args, **kwargs)
        self.base_model.kwargs.pop("max_tokens", None)
        self.base_model.kwargs["max_completion_tokens"] = int(protocol.get("analysis_max_output_tokens", 8192))

    stack.enter_context(patch.object(phase_analyzer.PhaseAnalyzer, "__init__", analyzer_init))
    original_analysis = phase_analyzer.PhaseAnalyzer.run_analysis

    @wraps(original_analysis)
    def analysis(self, *args, **kwargs):
        kwargs["max_steps"] = int(protocol.get("analysis_max_steps", 20))
        with audit_scope(role="analysis", provider=self.default_provider):
            return original_analysis(self, *args, **kwargs)

    stack.enter_context(patch.object(phase_analyzer.PhaseAnalyzer, "run_analysis", analysis))
    # Native's default trajectory summarizer otherwise selects an unrelated
    # default model instead of the configured analysis model.
    stack.enter_context(patch.object(phase_analyzer, "AnalysisAgent", partial(
        phase_analyzer.AnalysisAgent,
        summary_model_id=os.environ.get("ANALYSIS_MODEL", os.environ.get("DEFAULT_MODEL")),
    )))
    original_stats = phase_analyzer.PhaseAnalyzer._collect_task_stats

    @wraps(original_stats)
    def collect_stats(self, task_logs_dir):
        stats = original_stats(self, task_logs_dir)
        quality = []
        for path in sorted(Path(task_logs_dir).glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record.get("grade"), dict):
                quality.append({"task_id": record.get("task_id", path.stem), "grade": record["grade"]})
        stats["continuous_quality"] = quality
        return stats

    stack.enter_context(patch.object(phase_analyzer.PhaseAnalyzer, "_collect_task_stats", collect_stats))
    original_prompt = phase_analyzer.PhaseAnalyzer._build_analysis_prompt

    @wraps(original_prompt)
    def analysis_prompt(self, stats, *args, **kwargs):
        text = original_prompt(self, stats, *args, **kwargs)
        return text + ("\n\nThese are open writing tasks. Native CORRECT/WRONG flags only reflect "
                       "the 60/100 compatibility threshold, not the optimization target. "
                       "Improve actual continuous work quality using the dimension scores and reasons:\n"
                       + json.dumps(stats.get("continuous_quality", []), ensure_ascii=False))

    stack.enter_context(patch.object(phase_analyzer.PhaseAnalyzer, "_build_analysis_prompt", analysis_prompt))
    validator_init = phase_validator.PhaseValidator.__init__

    @wraps(validator_init)
    def validation_init(self, *args, **kwargs):
        validator_init(self, *args, **kwargs)
        # The old strict-factor adapter forcibly disabled native repairs when
        # lineage is active; v0.0.6 restores their ordinary behavior.
        self.enable_auto_fix = bool(protocol.get("enable_auto_fix", protocol.get("validator_auto_fix", True)))
        self.max_fix_attempts = int(protocol.get("max_fix_attempts", protocol.get("validator_max_fix_attempts", 3)))

    stack.enter_context(patch.object(phase_validator.PhaseValidator, "__init__", validation_init))
    # The global SDK audit observes every client; native role scopes distinguish
    # the ordinary analysis/generation/validation/repair work in that same ledger.
    for cls, method, role in (
        (phase_generator.PhaseGenerator, "run_generation", "generation"),
        (phase_validator.PhaseValidator, "run_validation", "validator"),
        (SWEAgentValidator, "fix_memory_system", "repair"),
        (TrajectoryViewerTool, "_generate_trajectory_summary", "trajectory_summary"),
        (OrientedProvider, "provide_memory", "memory_retrieval"),
        (OrientedProvider, "take_in_memory", "memory_ingestion"),
    ):
        original = getattr(cls, method)

        def scoped(self, *args, _original=original, _role=role, **kwargs):
            provider = getattr(self, "default_provider", None)
            snapshot = getattr(self, "_snapshot", None)
            if snapshot is not None:
                provider = snapshot.provider_id
            if _role in ("validator", "repair") and args:
                candidate = args[0]
                provider = candidate[0] if isinstance(candidate, list) and len(candidate) == 1 else (
                    candidate if isinstance(candidate, str) else provider)
            with audit_scope(role=_role, round_dir=str(getattr(self, "work_dir", "")),
                             **({"provider": provider} if provider else {})):
                return _original(self, *args, **kwargs)

        stack.enter_context(patch.object(cls, method, scoped))
    original_evolve = auto_evolver.AutoEvolver._run_memory_evolver

    @wraps(original_evolve)
    def evolving(self, round_dir, checkpoint=None, current_provider=None):
        with audit_scope(round_dir=str(round_dir), round=Path(round_dir).name,
                         provider=current_provider or self.state["best_provider"]):
            return original_evolve(self, round_dir, checkpoint, current_provider)

    stack.enter_context(patch.object(auto_evolver.AutoEvolver, "_run_memory_evolver", evolving))


def _terminal_evaluation(run_dir, workspace, winner, heldout_path, protocol, model_config, runner, native, sink):
    from .hss_tasks_runtime import NativeHSSRunner

    terminal = run_dir / "terminal"
    terminal.mkdir(parents=True, exist_ok=True)
    terminal_summary = terminal / "summary.json"
    if terminal_summary.exists():
        return json.loads(terminal_summary.read_text(encoding="utf-8"))
    rows = _read_rows(heldout_path)
    if len(rows) != 4:
        raise ValueError("Formal terminal evaluation requires four held-out tasks")
    storage = workspace / "storage"
    storage.mkdir(exist_ok=True)
    before = _tree_manifest(storage)
    frozen_storage = terminal / "frozen_training_storage"
    if frozen_storage.exists():
        if _tree_manifest(frozen_storage) != before:
            raise ValueError("Terminal training snapshot differs from current trained memory")
    else:
        shutil.copytree(storage, frozen_storage)
    write_json(terminal / "winner_before_heldout.json", _provider_artifact(run_dir, workspace, winner))
    write_json(terminal / "storage_freeze.json", {"manifest": before, "sha256": digest(before),
               "heldout_ingestion": False, "independent_snapshot_per_task": True})
    callback = NativeHSSRunner(run_dir, workspace, protocol, model_config, runner=runner, native=native,
                               usage_reader=lambda: sink.events, heldout=True)
    records = []
    for index in range(len(rows)):
        output = terminal / "task_logs" / str(index + 1)
        existing_record = output / f"{index + 1}.json"
        existing_summary = output / "_quality" / "summary.json"
        if existing_record.exists() and existing_summary.exists():
            ContinuousQualityAggregator(output).aggregate()
        else:
            with _terminal_storage(workspace, frozen_storage,
                                   terminal / "evaluation_snapshots" / str(index + 1), runner):
                callback("writingbench", winner, [index], heldout_path, output)
        records.append(json.loads(existing_record.read_text(encoding="utf-8")))
        if _tree_manifest(storage) != before:
            raise RuntimeError("Held-out evaluation changed trained memory outside its independent snapshot")
    summary = native_quality_summary(records)
    summary.update(provider=winner, heldout_task_count=len(records), heldout_ingestion=False,
                   independent_snapshot_per_task=True, training_storage_preserved=True,
                   training_storage_sha256=digest(before))
    write_json(terminal_summary, summary)
    return summary


def run_job(run_dir, job, protocol, budget_root, *, controller_policy=None):
    """Run exactly one authorized lineage in a fresh process.

    The manager injects credential configuration through the child environment.
    This worker never reads .env and never prints or serializes secret values.
    """
    from .cli import PROVIDER
    from .live_smoke import activate_workspace
    from .ablation_policy import install_native_policy
    from .hss_budget import install_hss_audit, summarize_job_usage
    from .hss_tasks_runtime import NativeHSSRunner

    run_dir = Path(run_dir).resolve(strict=True)
    protocol, job = deepcopy(protocol), deepcopy(job)
    learning_path = Path(job.get("learning_path", run_dir / "learning.jsonl")).resolve(strict=True)
    heldout_path = Path(job["heldout_path"]).absolute()
    if run_dir == heldout_path or run_dir in heldout_path.parents:
        raise ValueError("Held-out tasks must remain outside the training run")
    if len(_read_rows(learning_path)) != 8:
        raise ValueError("Two native rounds require exactly eight learning tasks")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("DEFAULT_MODEL"):
        raise ValueError("The worker requires configured model credentials in its environment")
    if job.get("code_evolve") is not True:
        raise ValueError("All v0.0.6 groups use normal code evolution")
    # Optional local encoders must use existing caches. Set these before any
    # candidate import, including when the parent shell explicitly used "0".
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("ANALYSIS_MODEL", os.environ["DEFAULT_MODEL"])
    os.environ.setdefault("GENERATION_MODEL", os.environ["DEFAULT_MODEL"])
    os.environ.setdefault("DEFAULT_JUDGE_MODEL", os.environ["DEFAULT_MODEL"])
    sys.dont_write_bytecode = True
    workspace = activate_workspace(run_dir)
    os.environ["PHILO_EVIDENCE_DIR"] = str(run_dir / "evidence" / "runtime")
    import FlashOAgents as native
    import run_flash_searcher_webwalkerqa as runner
    from MemEvolve.core import auto_evolver

    model_config = {"model_id": os.environ["DEFAULT_MODEL"], "api_key": os.environ["OPENAI_API_KEY"],
                    "api_base": os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"),
                    "temperature": protocol.get("actor_temperature", 0),
                    "max_tokens": int(protocol.get("actor_max_output_tokens", 8192)),
                    "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"}}
    status_path = run_dir / "runtime_summary.json"
    with ExitStack() as stack, audit_scope(job_id=job["job_id"]):
        stack.callback(install_native_policy(run_dir, job, compact_provider=False,
                                             generated_code_policy="native_local"))
        # Transport recovery changes the adapter, not the identity of a paid task.
        if set(controller_policy or {}) - {"connection_recovery"}:
            raise ValueError("Controller recovery cannot override task/model protocol fields")
        audit_protocol = {**protocol, **deepcopy(controller_policy or {})}
        sink = install_hss_audit(stack, run_dir, workspace, budget_root, audit_protocol)
        _install_native_adapters(stack, workspace, runner, protocol)
        stack.enter_context(patch.object(auto_evolver, "EVOLVE_RANDOM_SEED", int(job.get("order_seed", 20261029))))
        callback = NativeHSSRunner(run_dir, workspace, protocol, model_config, runner=runner, native=native,
                                   usage_reader=lambda: sink.events)
        auto = auto_evolver.AutoEvolver(
            analysis_model_id=os.environ["ANALYSIS_MODEL"], gen_model_id=os.environ["GENERATION_MODEL"],
            work_root=str(run_dir / "evolution"), dataset_name="writingbench", run_provider=callback,
            default_provider=PROVIDER, num_systems=2, creativity_index=protocol.get("creativity_index", .5),
            task_batch_x=2, top_t=2, extra_sample_y=1, datasets_config={"writingbench": str(learning_path)},
            max_workers=1, clear_storage_per_round=False,
        )
        try:
            refresh_registry(workspace, runner)
            remaining = 2 - auto.state.get("round", 0)
            if remaining < 0:
                raise ValueError("Native checkpoint exceeds the two authorized rounds")
            if remaining:
                with audit_scope(stage="native_evolution", provider=auto.state["best_provider"]):
                    auto.run(num_rounds=remaining)
            if auto.state.get("round") != 2 or auto.state.get("dataset_cursor") != 8:
                raise RuntimeError("Native evolution did not complete the full eight-task stream")
            refresh_registry(workspace, runner)
            winner = auto.state["best_provider"]
            terminal = _terminal_evaluation(run_dir, workspace, winner, heldout_path,
                                           protocol, model_config, runner, native, sink)
            result = {"status": "complete", "job": job, "protocol": protocol,
                      "winner": winner, "winner_provider": winner, "rounds_completed": 2,
                      "dataset_cursor": 8, "heldout_count": 4, "terminal": terminal,
                      "heldout_quality": terminal["summary"]["quality_score"],
                      "api_usage": sink.summary(), "rounds": _round_changes(run_dir, workspace),
                      "loaded_provider_evidence": _loaded_evidence(run_dir, sink),
                      "memory_learning": "native_persistent_no_resets_or_warmup_reconstruction"}
            result["api_usage_cumulative"] = summarize_job_usage(budget_root, job["job_id"])
            write_json(status_path, result)
            return result
        except Exception as exc:
            write_json(status_path, {"status": "failed", "job": job, "error_type": type(exc).__name__,
                       "rounds_completed": auto.state.get("round", 0),
                       "dataset_cursor": auto.state.get("dataset_cursor", 0), "api_usage": sink.summary(),
                       "api_usage_cumulative": summarize_job_usage(budget_root, job["job_id"]),
                       "native_checkpoint_preserved": True})
            raise
