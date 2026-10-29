"""Offline recovery of two paid candidates folded into one native enum mapping."""

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

from .ablation_runtime import digest
from .hss_experiment import now, process_owned, read_json, write_json
from .hss_generation_recovery import _matching_attempt


def _paths(root, job_id, round_name):
    experiment = Path(root).resolve(strict=True)
    manifest = read_json(experiment / "manifest.json")
    if not any(job["job_id"] == job_id for job in manifest["jobs"]):
        raise ValueError("Unknown experiment job")
    directory = (experiment / "jobs" / job_id).resolve(strict=True)
    round_dir = (directory / "evolution" / round_name).resolve(strict=True)
    if not round_dir.is_relative_to(directory / "evolution"):
        raise ValueError("Candidate round escaped the evolution directory")
    return experiment, directory, round_dir


def enum_collision_input_hashes(experiment_root, job_id, *, round_name="round_00"):
    """Return the exact current input pins needed for a reviewable repair."""
    _, directory, round_dir = _paths(experiment_root, job_id, round_name)
    files = list(round_dir.glob("generated_system_*.json"))
    files += [round_dir / name for name in ("state.json", "checkpoint.json", "created_system.json", "validated_systems.json")]
    files += [directory / "workspace/EvolveLab" / name for name in ("memory_types.py", "config.py")]
    files.extend((directory / "evidence/runtime").glob("calls-*.jsonl"))
    for tree in (round_dir / "eval_batch", round_dir / "eval_finalists"):
        files.extend(path for path in tree.rglob("*") if path.is_file())
    for config_path in round_dir.glob("generated_system_*.json"):
        config = read_json(config_path)
        files.append(directory / "workspace/EvolveLab/providers" / (config["provider_code"]["module_name"] + ".py"))
        alias = config["memory_type_info"]["enum_value"]
        for tree in (directory / "workspace/storage" / alias, directory / "lineage" / alias):
            files.extend(path for path in tree.rglob("*") if path.is_file())
    return {path.relative_to(directory).as_posix(): digest(path.read_bytes())
            for path in sorted(set(files)) if path.is_file()}


def _proof_never_loaded(directory, first_module, second_module):
    observations = []
    for path in sorted((directory / "evidence/runtime").glob("calls-*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record = json.loads(line)
            code_path = record.get("code_path", "").replace("\\", "/")
            if code_path.endswith("/" + second_module + ".py"):
                raise ValueError("The second candidate already ran; enum-only recovery is unsafe")
            if record.get("operation") == "loaded" and code_path.endswith("/" + first_module + ".py"):
                observations.append({"path": str(path), "line": line_number,
                                     "timestamp_utc": record.get("timestamp_utc"),
                                     "loaded_code_sha256": record.get("loaded_code_sha256")})
    for path in (directory / "evolution").glob("round_*/eval_*/*/*.json"):
        artifact = read_json(path).get("runtime_artifact", {})
        if str(artifact.get("source_path", "")).replace("\\", "/").endswith("/" + second_module + ".py"):
            raise ValueError("The second candidate has an independent task record; preserve its actual execution")
    if len(observations) < 2:
        raise ValueError("Insufficient native loading evidence to prove duplicate validation")
    return observations


def repair_enum_collision(experiment_root, job_id, *, expected_hashes,
                          round_name="round_00"):
    """Project only slot 2; rewind selection while preserving valid slot 1 work.

    Keep the prior common finalist task set and paid records. Recompute ranking
    from the complete candidate set, reusing unchanged completed task records.
    """
    experiment, directory, round_dir = _paths(experiment_root, job_id, round_name)
    status = read_json(directory / "job_status.json")
    budget = read_json(experiment / "budget/budget.json")
    if status.get("status") not in ("failed", "interrupted") or process_owned(status) or budget.get("pending"):
        raise ValueError("Enum collision repair requires a stopped failed job and a drained budget")
    receipt_path = round_dir / "enum_collision_recovery_2.json"
    target = round_dir / "generated_system_2.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt["input_hashes"] != expected_hashes or digest(target.read_bytes()) != receipt["repaired_candidate_sha256"]
                or {key: digest((directory / key).read_bytes()) for key in receipt["preserved_hashes"]} != receipt["preserved_hashes"]):
            raise ValueError("Recorded enum recovery artifacts changed")
        return receipt
    observed = enum_collision_input_hashes(experiment, job_id, round_name=round_name)
    if not isinstance(expected_hashes, dict) or observed != expected_hashes:
        raise ValueError("Enum collision inputs differ from the explicitly pinned artifacts")
    generated = sorted(round_dir.glob("generated_system_*.json"))
    if [path.name for path in generated] != ["generated_system_1.json", "generated_system_2.json"]:
        raise ValueError("Enum collision repair supports exactly two generated candidates")
    first, second = map(read_json, generated)
    a, b = first["provider_code"], second["provider_code"]
    alias = first["memory_type_info"]["enum_value"]
    enum = first["memory_type_info"]["enum_name"]
    if (second["memory_type_info"] != first["memory_type_info"] or a["module_name"] == b["module_name"]
            or a["code"] == b["code"] or not re.fullmatch(r"[A-Za-z_]\w*", enum)):
        raise ValueError("Target is not the specific same-enum, different-module collision")
    workspace = directory / "workspace"
    registry = (workspace / "EvolveLab/memory_types.py").read_text(encoding="utf-8")
    mapping = re.search(r"MemoryType\." + re.escape(enum) + r''':\s*\(["']([^"']+)["'],\s*["']([^"']+)["']\)''', registry)
    if not mapping or mapping.groups() != (a["class_name"], a["module_name"]):
        raise ValueError("Native mapping does not uniquely select the first candidate")
    for provider in (a, b):
        path = workspace / "EvolveLab/providers" / (provider["module_name"] + ".py")
        if path.read_text(encoding="utf-8") != provider["code"]:
            raise ValueError("Installed candidate differs from its paid generated source")
    state_path, checkpoint_path = round_dir / "state.json", round_dir / "checkpoint.json"
    created_path, validated_path = round_dir / "created_system.json", round_dir / "validated_systems.json"
    state, checkpoint, created, validated = map(read_json, (state_path, checkpoint_path, created_path, validated_path))
    if ((round_dir / "round_summary.json").exists() or any((directory / "evolution").glob("round_01/*"))
            or checkpoint.get("step_completed") not in (2, 3, 4)
            or checkpoint.get("validated_systems") != [alias, alias]
            or set(created.get("created", [])) != {alias} or created.get("failed")
            or set(validated.get("validated", [])) != {alias} or validated.get("failed")
            or validated.get("details", {}).get(alias, {}).get("verdict") != "passed"
            or state["phases"]["validate"].get("validated_systems") != [alias]):
        raise ValueError("Unrelated controller failures or advanced selection require a separate recovery")
    proof = _proof_never_loaded(directory, a["module_name"], b["module_name"])
    parent = checkpoint["parent_provider"]
    if (first.get("_orientation_parent") != second.get("_orientation_parent")
            or second.get("_orientation_parent", {}).get("snapshot", {}).get("sha256") != checkpoint.get("parent_orientation_sha256")):
        raise ValueError("Generated candidates have different parent philosophy snapshots")
    attempt_path, attempt = _matching_attempt(experiment, job_id, round_dir, second.get("_raw_llm_output"), parent)
    from .ablation_policy import project_candidate_namespace
    repaired, projection = project_candidate_namespace(second, workspace, work_dir=round_dir,
                                                        job_id=job_id, exclude_config=target)
    if not projection or repaired["memory_type_info"]["enum_value"] == alias:
        raise ValueError("Namespace projection did not separate the folded candidate")
    if any(repaired.get(key) != second.get(key) for key in ("orientation_items", "_orientation_parent", "_raw_llm_output")):
        raise ValueError("Namespace projection changed philosophy or paid raw evidence")
    new_created, new_validated = deepcopy(created), deepcopy(validated)
    new_created["created"], new_validated["validated"] = [alias], [alias]
    new_state = deepcopy(state)
    new_state["phases"]["create"].update(completed=False, created_systems=[alias], failed_systems=[])
    new_state["phases"]["validate"].update(completed=True, validated_systems=[alias], failed_systems=[])
    new_checkpoint = deepcopy(checkpoint)
    new_checkpoint.update(step_completed=2, validated_systems=[alias])
    for key in ("eval_results", "finalist_results", "selected_providers", "top_candidates"):
        new_checkpoint.pop(key, None)
    old_evolution = new_checkpoint.get("evolution_results", {})
    new_checkpoint["evolution_results"] = {"analysis": old_evolution["analysis"]} if "analysis" in old_evolution else {}
    mutable = {path.relative_to(directory).as_posix() for path in (target, state_path, checkpoint_path, created_path, validated_path)}
    preserved = {key: value for key, value in observed.items() if key not in mutable}
    backup = round_dir / "_enum_collision_recovery_2"
    if backup.exists() or observed != enum_collision_input_hashes(experiment, job_id, round_name=round_name):
        raise ValueError("Partial enum recovery or concurrently changed artifacts require review")
    backup.mkdir()
    for path in (target, state_path, checkpoint_path, created_path, validated_path):
        (backup / (path.name + ".before")).write_bytes(path.read_bytes())
    for name in ("eval_batch", "eval_finalists"):
        source, destination = round_dir / name, backup / (name + ".before")
        if source.exists():
            if not source.resolve().is_relative_to(round_dir) or not destination.resolve().is_relative_to(backup):
                raise ValueError("Paid task evidence escaped the checked recovery directory")
            shutil.copytree(source, destination)
    repaired["_enum_collision_recovery"] = {"original_attempt": attempt["attempt"], "original_sha256": observed[target.relative_to(directory).as_posix()],
                                           "namespace_projection": projection}
    for path, value in ((target, repaired), (state_path, new_state), (checkpoint_path, new_checkpoint),
                        (created_path, new_created), (validated_path, new_validated)):
        write_json(path, value)
    if {key: digest((directory / key).read_bytes()) for key in preserved} != preserved:
        raise RuntimeError("Protected first-candidate artifacts changed during enum recovery")
    receipt = {"kind": "offline_paid_enum_mapping_recovery", "job_id": job_id, "round": round_name,
               "recorded_utc": now(), "input_hashes": observed, "preserved_hashes": preserved,
               "repaired_candidate_sha256": digest(target.read_bytes()), "original_attempt": attempt["attempt"],
               "original_attempt_path": str(attempt_path), "original_attempt_usage": attempt["usage"],
               "new_provider": repaired["memory_type_info"]["enum_value"], "namespace_projection": projection,
               "old_second_never_loaded_evidence": proof, "paid_task_evidence_backup": str(backup),
               "finalist_tasks_preserved": new_checkpoint.get("finalist_tasks"),
               "first_candidate_batch_records_reused": True, "first_candidate_memory_preserved": True,
               "native_second_creation_and_validation_required": True, "selection_recomputed": True,
               "new_model_requests": 0, "generated_code_executed": False}
    write_json(receipt_path, receipt)
    return receipt
