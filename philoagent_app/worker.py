"""One-operation native bridge, always launched in a fresh Python process.

The product facade owns locking, workspace creation and evaluation snapshots.
This module deliberately does not call the fixed-size experimental controller.
Credentials are inherited from the parent environment and never serialized.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _positive_env(name, default):
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError("Product runtime limits must be positive integers")
    return value


def _protocol():
    """Normal native capacity, with a separate durable product budget."""
    model = os.environ.get("DEFAULT_MODEL")
    if not model or not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Model configuration is required in the worker environment")
    return {
        "model": model,
        "judge_model": os.environ.get("DEFAULT_JUDGE_MODEL") or model,
        "actor_max_steps": 32,
        "actor_max_output_tokens": 8192,
        "judge_max_output_tokens": 8192,
        "analysis_max_steps": 20,
        "analysis_max_output_tokens": 8192,
        "generation_max_output_tokens": 60000,
        "validation_max_output_tokens": 32768,
        "request_timeout_seconds": _positive_env("PHILOAGENT_APP_REQUEST_TIMEOUT", 300),
        "max_calls": _positive_env("PHILOAGENT_APP_MAX_CALLS", 60000),
        "max_tokens": _positive_env("PHILOAGENT_APP_MAX_TOKENS", 350000000),
        "max_inflight_requests": 1,
        "thinking": "provider-default",
    }


def _validate_request(root, request_path):
    request = _read_json(request_path)
    operation = request.get("operation")
    identifier = request.get("operation_id")
    if operation not in {"run", "evaluate_task", "evolve"}:
        raise ValueError("Unknown product worker operation")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
        raise ValueError("Invalid operation identifier")
    operation_dir = root / "operations" / identifier
    if request_path != operation_dir / "request.json":
        raise ValueError("Worker request must belong to this agent operation")
    payload = request.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Worker request requires a payload")
    # Validate all tasks before importing native modules or admitting paid work.
    from philoagent.hss_tasks_runtime import task_prompt

    if operation == "evolve":
        tasks, rounds = payload.get("tasks"), payload.get("rounds", 1)
        if type(rounds) is not int or rounds < 1:
            raise ValueError("Evolution rounds must be positive integers")
        if not isinstance(tasks, list) or len(tasks) < 4 * rounds:
            raise ValueError("Native evolution needs at least four development tasks per round")
    else:
        tasks = [payload.get("task")]
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Every task must be an object")
        task_prompt(task)
        if operation != "evaluate_task" and task.get("split") in {"evaluation", "heldout", "holdout", "test"}:
            raise ValueError("Evaluation tasks cannot enter product learning")
    return request, operation_dir, tasks


def _failure_stage(result, operation_dir):
    record_path = operation_dir / "output" / "1.json"
    if record_path.is_file():
        record = _read_json(record_path)
        result["record"] = record
        status = record.get("status", "execution")
        return status.removesuffix("_failed")
    return "evolution" if result["operation"] == "evolve" else "configuration"


def _include_task_criteria(original):
    def task_prompt(task):
        prompt = original(task)
        if task.get("task_criteria"):
            prompt += "\n评价要求：\n" + "\n".join(task["task_criteria"])
        return prompt
    return task_prompt


def execute(workspace_root, request_path):
    """Hold the native writer lock even if the launching parent is killed."""
    from .storage import file_lock

    root = Path(workspace_root).resolve(strict=True)
    with file_lock(root / ".worker.lock"):
        return _execute(root, request_path)


def _execute(workspace_root, request_path):
    """Execute once and persist a sanitized result, including paid partial work.

    An existing start marker prevents a second dispatch after a killed worker.
    There is intentionally no automatic actor/judge/ingestion recovery here.
    """
    root = Path(workspace_root).resolve(strict=True)
    request_path = Path(request_path).resolve(strict=True)
    request, operation_dir, tasks = _validate_request(root, request_path)
    result_path = operation_dir / "result.json"
    if result_path.is_file():
        return _read_json(result_path)
    marker = operation_dir / "worker_started.json"
    result = {"status": "failed", "operation": request["operation"],
              "operation_id": request["operation_id"], "api_usage": {},
              "artifacts": {"result": str(result_path), "api_calls": str(root / "api_calls.jsonl")}}
    if marker.exists():
        result.update(error_type="InterruptedOperation", failure_stage="interrupted",
                      retry_required="Create a new operation after reviewing the preserved work and memory.")
        _failure_stage(result, operation_dir)
        _write_json(result_path, result)
        return result
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
    started = time.monotonic()
    budget_root = Path(os.environ.get("PHILOAGENT_APP_BUDGET_ROOT", str(root / "budget"))).resolve()
    try:
        protocol = _protocol()
        payload, operation = request["payload"], request["operation"]
        state = _read_json(root / "state.json")
        provider = payload.get("provider", state["provider"])
        if operation != "evaluate_task" and provider != state["provider"]:
            raise ValueError("Only evaluation may override the current provider")
        result["provider"] = provider
        # Set isolation before the first native import. In particular, this
        # disables native import-time dotenv loading and optional downloads.
        os.environ["PHILO_LINEAGE_ROOT"] = str(root / "lineage")
        os.environ["PHILO_EVIDENCE_DIR"] = str(root / "evidence" / "runtime")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("ANALYSIS_MODEL", os.environ["DEFAULT_MODEL"])
        os.environ.setdefault("GENERATION_MODEL", os.environ["DEFAULT_MODEL"])
        os.environ.setdefault("DEFAULT_JUDGE_MODEL", os.environ["DEFAULT_MODEL"])
        sys.dont_write_bytecode = True
        native_workspace = root / "workspace"
        os.environ["PYTHONPATH"] = os.pathsep.join((str(native_workspace),
            os.environ.get("PYTHONPATH") or str(Path(__file__).resolve().parents[1])))
        from philoagent.live_smoke import activate_workspace
        from philoagent.ablation_policy import install_native_policy
        from philoagent.ablation_runtime import audit_scope, refresh_registry
        from philoagent.hss_budget import install_hss_audit, summarize_job_usage
        from philoagent.hss_native import _install_native_adapters
        from philoagent.hss_tasks_runtime import NativeHSSRunner
        from philoagent import hss_tasks_runtime
        from philoagent.orientation import load

        activate_workspace(root)
        os.environ["PHILO_EVIDENCE_DIR"] = str(root / "evidence" / "runtime")
        import FlashOAgents as native
        import run_flash_searcher_webwalkerqa as runner
        from MemEvolve.core import auto_evolver

        load(root / "lineage", provider)
        model_config = {"model_id": protocol["model"], "api_key": os.environ["OPENAI_API_KEY"],
                        "api_base": os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"),
                        "temperature": 0, "max_tokens": protocol["actor_max_output_tokens"],
                        "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"}}
        input_path = operation_dir / "input.jsonl"
        input_path.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
        result["artifacts"]["input"] = str(input_path)
        job = {"job_id": request["operation_id"], "code_evolve": True, "orientation_evolve": True}
        with ExitStack() as stack, audit_scope(job_id=job["job_id"]):
            stack.callback(install_native_policy(root, job, compact_provider=False, generated_code_policy="native_local"))
            sink = install_hss_audit(stack, root, native_workspace, budget_root, protocol)
            _install_native_adapters(stack, native_workspace, runner, protocol)
            stack.enter_context(patch.object(hss_tasks_runtime, "task_prompt",
                _include_task_criteria(hss_tasks_runtime.task_prompt)))
            callback = NativeHSSRunner(root, native_workspace, protocol, model_config, runner=runner,
                native=native, usage_reader=lambda: sink.events, heldout=operation == "evaluate_task")
            refresh_registry(native_workspace, runner)
            if operation == "evolve":
                evolution = operation_dir / "evolution"
                if evolution.exists():
                    raise RuntimeError("An evolution checkpoint already exists; it must not be silently replayed")
                auto = auto_evolver.AutoEvolver(
                    analysis_model_id=os.environ["ANALYSIS_MODEL"], gen_model_id=os.environ["GENERATION_MODEL"],
                    work_root=str(evolution), dataset_name="hss", run_provider=callback,
                    default_provider=provider, num_systems=2, creativity_index=.5,
                    task_batch_x=2, top_t=2, extra_sample_y=1, datasets_config={"hss": str(input_path)},
                    max_workers=1, clear_storage_per_round=False)
                result["artifacts"]["evolution"] = str(evolution)
                # The native controller normally deletes completed round
                # checkpoints. Retain them for this explicit product operation.
                with patch.object(auto, "_delete_checkpoint", lambda number: None):
                    auto.run(num_rounds=payload.get("rounds", 1))
                rounds = payload.get("rounds", 1)
                if auto.state.get("round") != rounds or auto.state.get("dataset_cursor") != 4 * rounds:
                    raise RuntimeError("Native evolution did not finish the requested development rounds")
                refresh_registry(native_workspace, runner)
                winner = auto.state["best_provider"]
                load(root / "lineage", winner)
                state.update(provider=winner, last_evolution=request["operation_id"])
                _write_json(root / "state.json", state)
                result.update(provider=winner, winner=winner, winner_provider=winner,
                              rounds_completed=rounds, dataset_cursor=auto.state["dataset_cursor"],
                              native_checkpoint_preserved=True, history=auto.state.get("history", []))
            else:
                output = operation_dir / "output"
                callback("hss", provider, [0], input_path, output)
                result["record"] = _read_json(output / "1.json")
                result["artifacts"]["record"] = str(output / "1.json")
            result.update(status="completed", api_usage=summarize_job_usage(budget_root, job["job_id"]))
    except Exception as exc:
        # Raw exception text can contain credentials or provider URLs. Native
        # callback records already preserve the answer and the exact phase.
        result.update(status="failed", error_type=type(exc).__name__)
        result["failure_stage"] = _failure_stage(result, operation_dir)
        if (budget_root / "budget.json").is_file():
            try:
                from philoagent.hss_budget import summarize_job_usage
                result["api_usage"] = summarize_job_usage(budget_root, request["operation_id"])
            except Exception:
                result["usage_read_failed"] = True
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _write_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    # Native libraries write verbose prompts and caught exception strings to
    # stdout. Product output consists solely of its sanitized JSON artifacts.
    try:
        with open(os.devnull, "w", encoding="utf-8") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            result = execute(args.workspace, args.request)
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        # Invalid requests never reach execution or model calls.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "failure_stage": "validation"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
