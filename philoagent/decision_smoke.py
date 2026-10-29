"""Six-task, serial native-agent mechanism smoke; never the formal matrix.

Each phase runs in a fresh Python process. The same persistent budget guards all
model clients, including native memory extraction and trajectory summarization.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .cli import PROJECT, PROVIDER, environment, init_run, source_hash
from .decision_fixture import Episode, TRAIN_IDS, PROBE_IDS
from .live_smoke import activate_workspace, digest, write_json
from .orientation import load
from .smoke_budget import Budget, wrap_native_model


MAX_CALLS = 100
MAX_TOKENS = 300000
NEW_LIMITS = {"max_calls": 150, "max_tokens": 800000}
PHASES = ("train", "evolve", "probe_old_01", "probe_new_01", "probe_new_02", "probe_old_02")


def run_limits(manifest):
    return {"max_calls": manifest["max_calls"], "max_tokens": manifest["max_tokens_soft_stop"]}


def file_manifest(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def message_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for part in value.values():
            yield from message_strings(part)
    elif isinstance(value, list):
        for part in value:
            yield from message_strings(part)


def guidance_provenance(budget_root, first_attempt, last_attempt, snapshot, memory_origin):
    """Observational sidecar; never modifies native memory or actor prompts."""
    records = []
    for number in range(first_attempt, last_attempt + 1):
        path = Path(budget_root) / "attempts" / f"{number:04d}.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(message_strings(audit["request"]["messages"]))
        if "You are a memory guidance system." not in text:
            continue
        content = (audit.get("response") or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = {}
        records.append({"attempt": number, "audit_path": str(path),
                        "source_kind": "retrieval_synthesis", "memory_origin": memory_origin,
                        "selected_indices": parsed.get("selected_indices") if isinstance(parsed, dict) else None,
                        "phi_visible_to_synthesis": bool(snapshot.items) and all(item in text for item in snapshot.items),
                        "interpretation": "Synthesized advice from task and supplied memories; not an observed prior episode."})
    return records


def freeze_training_memory(run_dir, summary):
    """Complete a missing snapshot offline, without replaying paid training."""
    run_dir = Path(run_dir)
    root = run_dir / "decision_smoke"
    source = run_dir / "workspace/storage" / PROVIDER
    frozen = root / "frozen_memory"
    expected = summary["memory_after"]
    if file_manifest(source) != expected:
        raise ValueError("Completed training memory differs from its saved summary")
    existing = file_manifest(frozen)
    if any(name not in expected or expected[name] != value for name, value in existing.items()):
        raise ValueError("Partial frozen memory differs from completed training")
    manifest_path = root / "frozen_memory.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != {"files": expected, "sha256": digest(expected)} or existing != expected:
            raise ValueError("Frozen training snapshot was changed")
        return
    frozen.mkdir(parents=True, exist_ok=True)
    for name in expected.keys() - existing.keys():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid saved memory path")
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    if file_manifest(frozen) != expected:
        raise ValueError("Frozen memory copy could not be verified")
    write_json(manifest_path, {"files": expected, "sha256": digest(expected)})


def make_question(task_id):
    return ("这是本地模拟决策任务。请依据情境目标自主选择信息和行动，遵守任务约束。"
            "必须调用 act 才会执行行动；final_answer 仅报告结果，不能代替实际行动。"
            "工具依次执行并计时，注意剩余时间。完成或到期后用 final_answer 简述"
            "实际行动、依据和不确定性；若已有哲学立场影响了选择，可以指出其关联。\n"
            + json.dumps(Episode(task_id).public_task(), ensure_ascii=False))


def prepare(name, *, max_calls=150, max_tokens=800000):
    if any(type(value) is not int or value <= 0 for value in (max_calls, max_tokens)):
        raise ValueError("Budget limits must be positive integers")
    values = environment()
    if not values.get("OPENAI_API_KEY") or not values.get("DEFAULT_MODEL"):
        raise ValueError("Configure OPENAI_API_KEY and DEFAULT_MODEL before preparing")
    result = init_run(name, "dialectical", seed_file=PROJECT / "seeds/systems/dialectical-with-methods-v1.json")
    run = Path(result["run_dir"])
    output = run / "decision_smoke"
    manifest = {
        "kind": "minimal_decision_mechanism_smoke", "formal_matrix_authorized": False,
        "model": values["DEFAULT_MODEL"], "temperature": 0, "thinking": "disabled",
        "train_ids": list(TRAIN_IDS), "probe_ids": list(PROBE_IDS),
        "task_executions": 6, "native_revision_proposals": 1,
        "factor_policy": {"code_evolve": False, "orientation_evolve": True},
        "max_calls": max_calls, "max_tokens_soft_stop": max_tokens,
        "protocol_version": 2,
        "budget_authorization": "User authorized expanding the minimal validation budget on 2026-10-29.",
        "actor_max_steps": 6, "actor_max_output_tokens": 3072,
        "analysis_max_steps": 4, "analysis_max_output_tokens": 4096,
        "generation_max_output_tokens": 12000,
        "adapter_sha256": result["adapter_sha256"],
        "fixture_sha256": hashlib.sha256((PROJECT / "philoagent/decision_fixture.py").read_bytes()).hexdigest(),
        "seed_sha256": result["seed_source"]["sha256"],
        "probe_protocol": "Fresh process and identical post-train M for each old/new Phi replay; no probe ingestion.",
        "selection": "One proposal is a diagnostic candidate, not a selected winner. No formal selection loop.",
        "stopping": "Reuse completed phases without payment; preserve failures and require an explicit repair before retry. Never expand the formal matrix.",
    }
    write_json(output / "manifest.json", manifest)
    Budget(output / "budget", max_calls=max_calls, max_tokens=max_tokens)
    return {"run_dir": str(run), **manifest}


def clone_probe(run_dir, task_id, provider_id, label):
    run_dir = Path(run_dir).resolve()
    if task_id not in PROBE_IDS:
        raise ValueError("Only held-out probe IDs can be cloned")
    # Keep every copied path inside this run, including provider storage aliases.
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or not re.fullmatch(r"[A-Za-z0-9_]+", provider_id):
        raise ValueError("Unsafe probe label or provider identifier")
    output = run_dir / "decision_smoke"
    frozen = output / "frozen_memory"
    expected = json.loads((output / "frozen_memory.json").read_text(encoding="utf-8"))
    if file_manifest(frozen) != expected["files"]:
        raise ValueError("Frozen post-training memory changed")
    trial = output / "trials" / label
    if trial.exists():
        raise FileExistsError("Probe trial already exists; paid trials cannot be silently rerun")
    shutil.copytree(run_dir / "workspace", trial / "workspace",
                    ignore=shutil.ignore_patterns("storage", "__pycache__", "*.pyc", ".env", ".env.*"))
    shutil.copytree(run_dir / "lineage", trial / "lineage")
    shutil.copyfile(run_dir / "run.json", trial / "run.json")
    shutil.copytree(frozen, trial / "workspace" / "storage" / provider_id)
    write_json(trial / "probe.json", {"task_id": task_id, "provider_id": provider_id,
                                      "label": label, "initial_memory_files": expected["files"],
                                      "common_memory_sha256": expected["sha256"]})
    return trial


def run_episodes(run_dir, values, budget_root, task_ids, provider_id, output, *, ingest, limits=None):
    task_ids = tuple(task_ids)
    if not task_ids or any(task not in (*TRAIN_IDS, *PROBE_IDS) for task in task_ids):
        raise ValueError("Unknown or empty decision task set")
    if ingest and any(task not in TRAIN_IDS for task in task_ids):
        raise ValueError("Probe observations cannot enter training memory")
    import FlashOAgents as native
    from base_agent import BaseAgent
    from EvolveLab.memory_types import TrajectoryData
    from run_flash_searcher_webwalkerqa import load_memory_provider

    output = Path(output)
    if output.exists():
        raise FileExistsError("Episode output already exists; automatic paid reruns are disabled")
    output.mkdir(parents=True)
    snapshot = load(run_dir / "lineage", provider_id)
    budget = Budget(budget_root, **(limits or {"max_calls": MAX_CALLS, "max_tokens": MAX_TOKENS}))
    model = native.OpenAIServerModel(model_id=values["DEFAULT_MODEL"], api_base=values.get("OPENAI_API_BASE"),
                                     api_key=values["OPENAI_API_KEY"], temperature=0, max_tokens=3072)
    wrap_native_model(model, budget, snapshot, "actor-and-memory:" + output.parent.name + ":" + output.name, 3072)
    storage = run_dir / "workspace/storage" / provider_id
    memory_before = file_manifest(storage)
    provider = load_memory_provider(provider_id, model)
    results = []
    for task_index, task_id in enumerate(task_ids):
        episode = Episode(task_id)
        memory_origin = "native_cold_start_priors" if task_index == 0 and not memory_before else "persistent_memories_may_mix_priors_and_learned_episodes"
        memory_input = output / "memory_inputs" / task_id
        if storage.exists():
            shutil.copytree(storage, memory_input)

        class Inspect(native.Tool):
            name = "inspect"
            description = "Inspect one listed source in this local episode; consumes its stated time and cost."
            inputs = {"source_id": {"type": "string", "description": "Exact source ID in the task."}}
            output_type = "string"

            def forward(self, source_id: str) -> str:
                return json.dumps(episode.inspect(source_id), ensure_ascii=False)

        class Act(native.Tool):
            name = "act"
            description = "Execute one listed action in this local episode; consumes its stated time."
            inputs = {"option": {"type": "string", "description": "Exact option ID in the task."}}
            output_type = "string"

            def forward(self, option: str) -> str:
                return json.dumps(episode.act(option), ensure_ascii=False)

        class Wait(native.Tool):
            name = "wait"
            description = "Let one time unit pass in the local episode."
            inputs = {}
            output_type = "string"

            def forward(self) -> str:
                return json.dumps(episode.wait(), ensure_ascii=False)

        question = make_question(task_id)
        agent = native.ToolCallingAgent(tools=[Inspect(), Act(), Wait()], model=model, memory_provider=provider,
                                        max_steps=6, summary_interval=None, verbosity_level=0)
        model.reset_total_counts()
        before = budget.summary()
        answer = agent.run(question)
        messages = agent.write_memory_to_messages(include_system_prompt=False)
        evaluation = episode.summary()  # Only evaluated after the actor has finished.
        wrapper = BaseAgent(model)
        wrapper.agent_fn = agent
        record = {"kind": "synthetic_decision_episode", "status": "success", "question": question,
                  "agent_result": answer, "is_correct": evaluation["success"],
                  "judgement": "correct" if evaluation["success"] else "incorrect",
                  "evaluation": evaluation, "agent_messages": messages, **wrapper.capture_trajectory(),
                  "memory_input_files": file_manifest(memory_input), "memory_input_origin": memory_origin,
                  "guidance_provenance": guidance_provenance(budget_root, before["attempt_count"] + 1,
                                                              budget.summary()["attempt_count"], snapshot, memory_origin),
                  "memory_ingestion": None, "ingestion_status": "pending" if ingest else "disabled",
                  "metrics": model.get_total_counts(), "orientation_snapshot": snapshot.to_dict(),
                  "budget_before": before, "budget_after": budget.summary()}
        write_json(output / "task_logs" / f"{task_id}.json", record)
        # Persist the completed physical episode before a separate memory call.
        # Ingestion failure must not erase evidence or silently replay the actor.
        if ingest:
            try:
                ingestion = provider.take_in_memory(TrajectoryData(
                    query=question, trajectory=messages, result=answer,
                    metadata={"status": "success", "is_correct": evaluation["success"],
                              "full_query": question, "task_id": task_id, "synthetic_local_fixture": True}))
                record["memory_ingestion"] = list(ingestion)
                record["ingestion_status"] = "accepted" if ingestion[0] else "not_accepted"
                if evaluation["success"] and not ingestion[0]:
                    raise RuntimeError("Native ingestion rejected a successful training episode")
            except Exception as exc:
                record.update(ingestion_status="failed", ingestion_error_type=type(exc).__name__)
                raise
            finally:
                record.update(metrics=model.get_total_counts(), budget_after=budget.summary())
                write_json(output / "task_logs" / f"{task_id}.json", record)
        if budget.summary()["blocked_reason"]:
            raise RuntimeError("Shared budget blocked; inspect the persisted budget ledger")
        results.append(evaluation)
    summary = {"provider": provider_id, "tasks": results, "ingestion_enabled": ingest,
               "phi_items_sha256": digest(list(snapshot.items)),
               "memory_before": memory_before, "memory_after": file_manifest(storage),
               "budget": budget.summary()}
    write_json(output / "summary.json", summary)
    return summary


def execute(run_dir, phase):
    run_dir = Path(run_dir).resolve(strict=True)
    output = run_dir / "decision_smoke"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    limits = run_limits(manifest)
    expected_adapter = manifest["adapter_sha256"]
    amendment_path = output / "protocol_amendment.json"
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if (amendment["original_adapter_sha256"] != expected_adapter
                or amendment["scope"] != "resume_unfinished_minimal_only"
                or amendment["budget_limits"] != limits):
            raise ValueError("Invalid minimal recovery amendment")
        expected_adapter = amendment["adapter_sha256"]
    if manifest["formal_matrix_authorized"] or source_hash(PROJECT / "philoagent") != expected_adapter:
        raise ValueError("Minimal protocol or adapter changed since preparation")
    if phase not in PHASES:
        raise ValueError("Unknown minimal experiment phase")
    success_path = (output / phase / "summary.json" if phase in {"train", "evolve"}
                    else output / "trials" / phase / "result/summary.json")
    if success_path.exists():
        saved = json.loads(success_path.read_text(encoding="utf-8"))
        if phase == "train":
            freeze_training_memory(run_dir, saved)
        return {"reused_completed_phase": phase, "network_dispatched": False,
                "summary": saved}
    values = environment()
    if values.get("DEFAULT_MODEL") != manifest["model"]:
        raise ValueError("Configured model changed since protocol freeze")
    values.update(ANALYSIS_MODEL=values["DEFAULT_MODEL"], GENERATION_MODEL=values["DEFAULT_MODEL"],
                  DEFAULT_SUMMARY_MODEL=values["DEFAULT_MODEL"], PHILO_SMOKE_THINKING="disabled")
    os.environ.update(values)
    budget_root = output / "budget"
    native_log = output / f"{phase}.native.log"
    if native_log.exists():
        raise FileExistsError("This phase was already attempted; inspect its evidence before any new run")
    with native_log.open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
        if phase == "train":
            activate_workspace(run_dir)
            result = run_episodes(run_dir, values, budget_root, TRAIN_IDS, PROVIDER, output / "train", ingest=True, limits=limits)
            freeze_training_memory(run_dir, result)
        elif phase == "evolve":
            frozen = json.loads((output / "frozen_memory.json").read_text(encoding="utf-8"))
            if file_manifest(run_dir / "workspace/storage" / PROVIDER) != frozen["files"]:
                raise ValueError("Training memory changed before native revision")
            activate_workspace(run_dir)
            from .decision_smoke_evolve import evolve
            result = evolve(run_dir, values, budget_root, limits=limits)
        else:
            summary_path = output / ("evolve_resume/summary.json" if amendment_path.exists() else "evolve/summary.json")
            candidate = json.loads(summary_path.read_text(encoding="utf-8"))["candidate_provider"]
            # Four separate process invocations, with balanced condition order.
            schedule = {"probe_old_01": ("probe_01", PROVIDER), "probe_new_01": ("probe_01", candidate),
                        "probe_new_02": ("probe_02", candidate), "probe_old_02": ("probe_02", PROVIDER)}
            task_id, provider_id = schedule[phase]
            trial = clone_probe(run_dir, task_id, provider_id, phase)
            activate_workspace(trial)
            result = run_episodes(trial, values, budget_root, (task_id,), provider_id, trial / "result", ingest=False, limits=limits)
    return result


def finalize(run_dir):
    """Offline acceptance from actual saved state; never from model self-reports."""
    run_dir = Path(run_dir).resolve(strict=True)
    root = run_dir / "decision_smoke"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    manifest = read(root / "manifest.json")
    if manifest["formal_matrix_authorized"] or source_hash(PROJECT / "philoagent") != manifest["adapter_sha256"]:
        raise ValueError("Finalization requires the original minimal run source and scope")
    train = read(root / "train/summary.json")
    evolution = read(root / "evolve/summary.json")
    fixed = read(root / "evolve/policy_fixed_evidence.json")
    frozen = read(root / "frozen_memory.json")
    if not (evolution["success"] and fixed["policy_fixed"] and fixed["ordinary_memory_unchanged"]):
        raise ValueError("Native evolution did not preserve fixed P and training M")
    if file_manifest(root / "frozen_memory") != frozen["files"]:
        raise ValueError("Frozen memory was modified")
    if tuple(task["task_id"] for task in train["tasks"]) != TRAIN_IDS:
        raise ValueError("Training task coverage differs from protocol")
    old_phi = load(run_dir / "lineage", PROVIDER)
    new_phi = load(run_dir / "lineage", evolution["candidate_provider"])
    trials = {}
    for phase in PHASES[2:]:
        trial = root / "trials" / phase
        saved = read(trial / "result/summary.json")
        meta = read(trial / "probe.json")
        expected_id = "probe_" + phase[-2:]
        is_new = "_new_" in phase
        phi = new_phi if is_new else old_phi
        provider = evolution["candidate_provider"] if is_new else PROVIDER
        if (meta["task_id"] != expected_id or meta["provider_id"] != provider
                or saved["provider"] != provider or saved["ingestion_enabled"]
                or meta["initial_memory_files"] != frozen["files"]
                or saved["memory_before"] != frozen["files"]
                or saved["phi_items_sha256"] != digest(list(phi.items))
                or [x["task_id"] for x in saved["tasks"]] != [expected_id]):
            raise ValueError("Probe identity, Phi, memory isolation or ingestion invariant failed")
        record = read(trial / "result/task_logs" / f"{expected_id}.json")
        if record["memory_ingestion"] is not None or record["ingestion_status"] != "disabled":
            raise ValueError("Probe feedback was ingested")
        trials[phase] = saved["tasks"][0]
    budget = Budget(root / "budget", **run_limits(manifest))
    state = budget.state
    if state["blocked_reason"] or state["unknown_usage_attempts"] or any(a["status"] != "success" for a in state["attempts"]):
        raise ValueError("Budget contains failed, pending or unknown paid calls")
    roles = {}
    for attempt in state["attempts"]:
        role = roles.setdefault(attempt["label"], {"requests": 0, "total_tokens": 0})
        role["requests"] += 1
        role["total_tokens"] += attempt["usage"]["total_tokens"]
    api_events = [read(path) for path in sorted((root / "budget/attempts").glob("*.json"))]
    runtime_checks = {}
    for phase in PHASES[2:]:
        trial = root / "trials" / phase
        phi = new_phi if "_new_" in phase else old_phi
        provider = evolution["candidate_provider"] if "_new_" in phase else PROVIDER
        events = [json.loads(line) for path in (trial / "live_smoke/runtime_evidence").glob("calls-*.jsonl")
                  for line in path.read_text(encoding="utf-8").splitlines()]
        if (not events or not any(e.get("operation") == "loaded" for e in events)
                or any(e.get("loaded_code_sha256") != fixed["parent_code_sha256"]
                       or not e.get("disk_matches_loaded_code")
                       or e.get("details", {}).get("provider") != provider for e in events)
                or any(e.get("operation") == "take_in_memory" for e in events)):
            raise ValueError("Probe execution evidence failed fixed-P or no-ingestion checks")
        actual_requests = [a for a in api_events if a["label"] == "actor-and-memory:" + phase + ":result"]
        phi_visible = bool(phi.items) and any(
            all(item in "\n".join(message_strings(a["request"]["messages"])) for item in phi.items)
            for a in actual_requests)
        if not actual_requests or (phi.items and not phi_visible):
            raise ValueError("Probe Phi was not observed in actual model requests")
        runtime_checks[phase] = {"loaded_P_sha256": fixed["parent_code_sha256"], "provider": provider,
                                 "phi_in_actual_request": phi_visible, "no_ingestion_runtime_calls": True}
    comparisons = []
    for task_id in PROBE_IDS:
        suffix = task_id[-2:]
        old, new = trials["probe_old_" + suffix], trials["probe_new_" + suffix]
        actions = lambda task: [(a["tool"], a["argument"]) for a in task["audit"]]
        comparisons.append({"task_id": task_id, "old_actions": actions(old), "new_actions": actions(new),
                            "action_sequence_changed": actions(old) != actions(new),
                            "old_utility": old["utility"], "new_utility": new["utility"],
                            "old_violations": old["violations"], "new_violations": new["violations"]})
    if not all(t["done"] for t in [*train["tasks"], *trials.values()]):
        raise ValueError("An actor ended without reaching a terminal environment state")
    result = {"protocol_completed": True, "formal_matrix_started": False,
              "model": manifest["model"], "training": train["tasks"], "trials": trials,
              "orientation_changed": old_phi.items != new_phi.items,
              "parent_phi_sha256": digest(list(old_phi.items)), "candidate_phi_sha256": digest(list(new_phi.items)),
              "policy_fixed": True, "all_probes_started_from_identical_memory": True,
              "probe_ingestion_disabled": True, "comparisons": comparisons,
              "runtime_checks": runtime_checks,
              "all_episodes_reached_terminal_state": all(t["done"] for t in [*train["tasks"], *trials.values()]),
              "decision_success_count": sum(t["success"] for t in [*train["tasks"], *trials.values()]),
              "task_executions": 6, "budget": budget.summary(), "roles": roles,
              "candidate_provider": evolution["candidate_provider"], "run_dir": str(run_dir),
              "interpretation": "Mechanism acceptance only; one proposed version and two probe pairs do not establish efficacy or causality."}
    write_json(root / "final_status.json", result)
    return result


def run_all(run_dir):
    """Serial fresh processes; a complete phase is read back without paying again."""
    run_dir = Path(run_dir).resolve(strict=True)
    for phase in PHASES:
        completed = subprocess.run([sys.executable, "-m", "philoagent.decision_smoke", phase, str(run_dir)],
                                   cwd=PROJECT, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError("Minimal phase failed: " + phase + "; inspect its local evidence")
        result = json.loads(completed.stdout)
        print(json.dumps({"phase_completed": phase, "reused": bool(result.get("reused_completed_phase"))}), flush=True)
    return finalize(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "all", "finalize", *PHASES])
    parser.add_argument("run", help="New run name for prepare; absolute run directory otherwise")
    parser.add_argument("--max-calls", type=int, default=NEW_LIMITS["max_calls"])
    parser.add_argument("--max-tokens", type=int, default=NEW_LIMITS["max_tokens"])
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            result = prepare(args.run, max_calls=args.max_calls, max_tokens=args.max_tokens)
        elif args.phase == "all":
            result = run_all(args.run)
        elif args.phase == "finalize":
            result = finalize(args.run)
        else:
            result = execute(args.run, args.phase)
    except Exception as exc:
        # Exceptions from external libraries can embed transport details. Full
        # local phase evidence remains available; never echo credentials/URLs.
        print(json.dumps({"phase": args.phase, "success": False, "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
