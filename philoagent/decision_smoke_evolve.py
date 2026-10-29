"""One native Phi revision with P frozen; no evaluation or generated-code execution.

The caller activates the isolated workspace; API settings arrive as values.
Only the two completed training trajectories and their resulting memory are used.
This module itself never loads an environment file or selects another model.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import difflib
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .ablation_policy import install_native_policy, _parent_state, _without_routing
from .cli import PROVIDER
from .live_smoke import validate_analysis, write_json
from .orientation import assert_unchanged, load
from .reflection_view import (TRAIN_IDS, ReflectionClient, ReflectionCoverage, analysis_prompt,
                              build_view, compact, generation_prompt, replace_phi)
from .smoke_budget import Budget, wrap_native_model


def _sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _files(root, pattern="**/*"):
    return {path.relative_to(root).as_posix(): _sha(path.read_bytes())
            for path in sorted(root.glob(pattern)) if path.is_file()}


def evolve(run_dir, values, budget_root, *, output_name="evolve", limits=None):
    """Return the created alias and evidence; the caller must evaluate it afresh.

    Inputs: an already activated run, in-memory API settings, and the shared
    serial Budget root. Default limits preserve 100 requests / 300,000 tokens;
    an explicit limits mapping must match the already authorized shared ledger.
    Training logs: run/decision_smoke/train/task_logs/{train_01,train_02}.json.
    Outputs: run/decision_smoke/<output_name>; never launches a probe or a new process.
    Failures preserve completed artifacts and raise a sanitized RuntimeError.
    """
    run_dir = Path(run_dir).resolve(strict=True)
    workspace = (run_dir / "workspace").resolve(strict=True)
    training = run_dir / "decision_smoke/train/task_logs"
    if not isinstance(output_name, str) or not output_name.isidentifier():
        raise ValueError("Evolution output_name must be a simple identifier")
    output = run_dir / "decision_smoke" / output_name
    if Path.cwd().resolve() != workspace:
        raise ValueError("Activate this run's workspace before evolve")
    lineage = run_dir / "lineage"
    if Path(os.environ.get("PHILO_LINEAGE_ROOT", "")).resolve() != lineage.resolve():
        raise ValueError("The active lineage does not match this run")
    if output.exists():
        raise ValueError("Evolution output already exists; do not retry an unknown result")
    logs = sorted(training.glob("*.json"))
    if {path.name for path in logs} != {"train_01.json", "train_02.json"}:
        raise ValueError("Native evolution requires exactly the two training logs")
    model_id = values.get("DEFAULT_MODEL")
    api_base = values.get("OPENAI_API_BASE") or values.get("OPENAI_BASE_URL")
    if not model_id or not values.get("OPENAI_API_KEY"):
        raise ValueError("DEFAULT_MODEL and OPENAI_API_KEY are required")
    if os.environ.get("PHILO_SMOKE_THINKING", "disabled") != "disabled":
        raise ValueError("The decision smoke requires disabled thinking")

    # Native imports are deliberately deferred until workspace activation.
    import openai
    from base_agent import BaseAgent
    from MemEvolve.phases import phase_analyzer
    from MemEvolve.phases.phase_generator import PhaseGenerator
    from MemEvolve.phases.memory_creator import MemorySystemCreator
    from MemEvolve.utils import trajectory_tools

    if not Path(phase_analyzer.__file__).resolve().is_relative_to(workspace):
        raise ValueError("Native modules were loaded from another workspace")
    snapshot = load(lineage, PROVIDER)
    parent = _parent_state(workspace, PROVIDER)
    memory_root = workspace / "storage" / PROVIDER
    source_before = _files(workspace / "EvolveLab/providers", "*.py")
    memory_before = _files(memory_root) if memory_root.exists() else {}
    limits = {"max_calls": 100, "max_tokens": 300000} if limits is None else limits
    if set(limits) != {"max_calls", "max_tokens"}:
        raise ValueError("Explicit limits require max_calls and max_tokens")
    budget = Budget(budget_root, **limits)
    budget.bind_model(model_id)
    output.mkdir(parents=True)
    view = output / "reflection_view"
    projection = build_view(training, view, snapshot)
    projected_training = view / "task_logs"
    memory_files = [path.resolve() for path in sorted(memory_root.rglob("*"))
                    if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".txt", ".md"}]
    allowed_files = {path.resolve() for path in view.rglob("*.json")} | set(memory_files)
    write_json(output / "input_manifest.json", {
        "training_logs": _files(training, "*.json"),
        "ordinary_memory": memory_before,
        "parent_orientation_snapshot": snapshot.to_dict(),
        "parent_code_sha256": parent["source_sha256"],
        "parent_effective_config_sha256": _sha(_without_routing(parent["config"])),
        "analysis_max_steps": 4, "base_agent_max_attempts": 1,
        "analysis_max_output": 4096, "generation_max_output": 12000,
        "budget_before": budget.summary(), "formal_matrix": False,
        "reflection_projection": projection, "readable_files": [str(path) for path in sorted(allowed_files)],
        "reflection_mode": "phi_only_native_analysis_with_deterministic_evidence_projection",
    })

    def client(label, max_output):
        raw = openai.OpenAI(api_key=values["OPENAI_API_KEY"],
                            base_url=api_base,
                            timeout=90, max_retries=0)
        holder = SimpleNamespace(client=raw, model_id=model_id)
        return wrap_native_model(holder, budget, snapshot, label,
                                 max_output=max_output).client

    def summary_client(**_kwargs):
        # Constructor compatibility only. Deterministic summaries in the
        # projected logs make all model-generated summary calls unnecessary.
        def disabled_summary(**_request):
            raise RuntimeError("Generated trajectory summaries are disabled")
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=disabled_summary)))

    original_forward = BaseAgent.forward

    def single_attempt(self, *args, **kwargs):
        kwargs["max_retries"] = 1
        return original_forward(self, *args, **kwargs)

    original_view = trajectory_tools.MemoryDatabaseViewerTool.forward

    def training_memory_only(self, db_path, start_line, end_line):
        requested = Path(db_path).resolve()
        if requested not in allowed_files:
            return compact({"error": "Only the training projection, current Phi and listed training M files are readable"})
        if requested == (view / "current_phi.json").resolve():
            return compact({"phi_sha256": snapshot.sha256,
                            "reference": "The complete unchanged current_phi is already in the task prompt."})
        return compact(replace_phi(json.loads(original_view(self, str(requested), start_line, end_line)), snapshot))

    original_trajectories = trajectory_tools.TrajectoryViewerTool.forward
    original_steps = trajectory_tools.StepViewerTool.forward
    coverage = ReflectionCoverage()
    write_json(output / "analysis_coverage.json", coverage.report())

    def training_trajectories_only(self, task_ids):
        if not isinstance(task_ids, list) or any(task not in TRAIN_IDS for task in task_ids):
            return compact({"error": "Only the two training IDs are readable"})
        result = json.loads(original_trajectories(self, task_ids))
        coverage.trajectories(task_ids, result)
        write_json(output / "analysis_coverage.json", coverage.report())
        return compact(result)

    def training_steps_only(self, task_id, step_indices):
        if task_id not in TRAIN_IDS:
            return compact({"error": "Only the two training IDs are readable"})
        result = json.loads(original_steps(self, task_id, step_indices))
        coverage.specific_steps(task_id, step_indices, result)
        write_json(output / "analysis_coverage.json", coverage.report())
        return compact(result)

    stream = io.StringIO()
    stage = "analysis"
    try:
        with redirect_stdout(stream), ExitStack() as stack:
            api_environment = {"OPENAI_API_KEY": values["OPENAI_API_KEY"]}
            if api_base:
                api_environment.update(OPENAI_API_BASE=api_base, OPENAI_BASE_URL=api_base)
            stack.enter_context(patch.dict(os.environ, api_environment))
            restore = install_native_policy(run_dir, {
                "job_id": "minimal_decision_phi_smoke",
                "code_evolve": False, "orientation_evolve": True,
            })
            stack.callback(restore)
            stack.enter_context(patch.object(BaseAgent, "forward", single_attempt))
            stack.enter_context(patch.object(phase_analyzer, "AnalysisAgent", partial(
                phase_analyzer.AnalysisAgent, summary_model_id=model_id)))
            stack.enter_context(patch.object(trajectory_tools, "OpenAI", summary_client))
            stack.enter_context(patch.object(trajectory_tools.MemoryDatabaseViewerTool,
                                             "forward", training_memory_only))
            stack.enter_context(patch.object(trajectory_tools.TrajectoryViewerTool,
                                             "forward", training_trajectories_only))
            stack.enter_context(patch.object(trajectory_tools.StepViewerTool,
                                             "forward", training_steps_only))
            analyzer = phase_analyzer.PhaseAnalyzer(model_id, str(projected_training), PROVIDER)
            wrap_native_model(analyzer.base_model, budget, snapshot,
                              "decision-evolve-analysis", max_output=4096)
            analyzer.base_model.client = ReflectionClient(analyzer.base_model.client)
            analyzer._orientation_snapshot = snapshot
            bounded_analysis = analysis_prompt(snapshot, parent, view, memory_files)
            (output / "analysis_prompt.txt").write_text(bounded_analysis, encoding="utf-8")
            analyzer._build_analysis_prompt = lambda *args, **kwargs: bounded_analysis
            analysis = analyzer.run_analysis(max_steps=4)
            write_json(output / "analysis.json", analysis)
            validate_analysis(analysis)
            coverage.require_complete()
            if budget.state.get("blocked_reason"):
                raise RuntimeError("Budget blocked during native analysis")

            stage = "generation"
            generator = PhaseGenerator(client("decision-evolve-generation", 12000),
                                       model_id, output, PROVIDER)
            generator._orientation_snapshot = snapshot
            generator._get_memory_types_reference = lambda: "P is frozen; no implementation reference is required."

            def bounded_generation(text, _types, _creativity):
                prompt = generation_prompt(text, snapshot, generator._extract_existing_systems())
                (output / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
                return prompt

            generator._build_generation_prompt = bounded_generation
            generation = generator.run_generation(analysis, creativity_index=0.2)
            write_json(output / "generation.json", generation)
            if not generation.get("success") or budget.state.get("blocked_reason"):
                raise RuntimeError("Native generation did not complete successfully")

            stage = "creation"
            created = MemorySystemCreator.create_memory_system(generation["config"], str(workspace))
            write_json(output / "created.json", created)
            if not created.get("success"):
                raise RuntimeError("Frozen provider alias creation failed")

        stage = "evidence"
        candidate_id = created["memory_type"]
        candidate = load(lineage, candidate_id)
        child = _parent_state(workspace, candidate_id)
        assert_unchanged(snapshot)
        assert_unchanged(candidate)
        source_after = _files(workspace / "EvolveLab/providers", "*.py")
        memory_after = _files(memory_root) if memory_root.exists() else {}
        fixed = (parent["source_sha256"] == child["source_sha256"]
                 and parent["class_name"] == child["class_name"]
                 and parent["module_name"] == child["module_name"]
                 and _without_routing(parent["config"]) == _without_routing(child["config"])
                 and source_before == source_after)
        evidence = {
            "parent_provider": PROVIDER, "candidate_provider": candidate_id,
            "parent_code_sha256": parent["source_sha256"],
            "candidate_code_sha256": child["source_sha256"],
            "parent_module": parent["module_name"], "candidate_module": child["module_name"],
            "parent_class": parent["class_name"], "candidate_class": child["class_name"],
            "parent_effective_config_sha256": _sha(_without_routing(parent["config"])),
            "candidate_effective_config_sha256": _sha(_without_routing(child["config"])),
            "provider_sources_before": source_before, "provider_sources_after": source_after,
            "policy_fixed": fixed, "ordinary_memory_unchanged": memory_before == memory_after,
            "generated_code_executed": False, "alias_loaded_in_this_phase": False,
        }
        write_json(output / "policy_fixed_evidence.json", evidence)
        old_text = json.dumps(list(snapshot.items), ensure_ascii=False, indent=2).splitlines(keepends=True)
        new_text = json.dumps(list(candidate.items), ensure_ascii=False, indent=2).splitlines(keepends=True)
        (output / "phi.diff").write_text("".join(difflib.unified_diff(
            old_text, new_text, fromfile="parent_phi", tofile="candidate_phi")), encoding="utf-8")
        write_json(output / "phi_diff.json", {
            "changed": snapshot.items != candidate.items,
            "parent": snapshot.to_dict(), "candidate": candidate.to_dict(),
        })
        if not fixed or memory_before != memory_after:
            raise RuntimeError("Evolution changed frozen P or ordinary training memory")
        result = {
            "success": True, "kind": "minimal_decision_smoke_native_phi_revision",
            "formal_matrix": False, "parent_provider": PROVIDER,
            "candidate_provider": candidate_id, "policy_fixed": fixed,
            "orientation_changed": snapshot.items != candidate.items,
            "generated_code_executed": False, "probe_evaluation_started": False,
            "candidate_requires_fresh_process": True,
            "training_logs_after": _files(training, "*.json"),
            "reflection_projection": projection,
            "analysis_coverage": coverage.report(),
            "budget": budget.summary(), "output_dir": str(output),
        }
        write_json(output / "summary.json", result)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {
            "success": False, "stage": stage, "error_type": type(exc).__name__,
            "budget": budget.summary(), "generated_code_executed": False,
            "probe_evaluation_started": False,
        })
        raise RuntimeError("Native decision evolution failed in " + stage + "; inspect saved artifacts") from None
    finally:
        (output / "native_stdout.txt").write_text(stream.getvalue(), encoding="utf-8")
