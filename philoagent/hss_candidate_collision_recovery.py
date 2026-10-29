"""Repair one paid, uncreated candidate's duplicate native namespace offline."""

from copy import deepcopy
from pathlib import Path
import re

from .ablation_runtime import digest
from .hss_experiment import now, process_owned, read_json, write_json
from .hss_generation_recovery import _matching_attempt


def _hashes(paths):
    return {str(path): digest(path.read_bytes()) for path in paths if path.is_file()}


def _collision(error, alias, module):
    return (isinstance(error, dict) and error.get("system") == alias
            and "already exists" in error.get("error", "")
            and f"{module}.py" in error["error"])


def repair_candidate_collision(experiment_root, job_id, *, candidate_index=2,
                               round_name="round_00", expected_hashes):
    """Rename only a duplicate paid slot; native creation/validation still run.

    Earlier generated candidates and their installed code, storage, orientation
    and validation evidence are pinned. Other experiment jobs may keep running.
    """
    if type(candidate_index) is not int or candidate_index < 2:
        raise ValueError("Collision repair requires a later candidate slot")
    experiment = Path(experiment_root).resolve(strict=True)
    manifest = read_json(experiment / "manifest.json")
    job = next((item for item in manifest["jobs"] if item["job_id"] == job_id), None)
    if job is None:
        raise ValueError("Unknown experiment job")
    directory = (experiment / "jobs" / job_id).resolve(strict=True)
    round_dir = (directory / "evolution" / round_name).resolve(strict=True)
    if not round_dir.is_relative_to(directory / "evolution"):
        raise ValueError("Candidate round escaped its native evolution directory")
    status = read_json(directory / "job_status.json")
    if status.get("status") not in ("failed", "interrupted") or process_owned(status):
        raise ValueError("Candidate collision recovery requires a stopped failed job")
    budget = read_json(experiment / "budget/budget.json")
    for number, pending in budget.get("pending", {}).items():
        attempt = read_json(experiment / "budget" / "attempts" / f"{int(number):06d}.json")
        if attempt.get("job_id", pending.get("job_id")) == job_id:
            raise ValueError("Target job still has an unsettled model request")
    target = round_dir / f"generated_system_{candidate_index}.json"
    receipt_path = round_dir / f"candidate_collision_recovery_{candidate_index}.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt["input_candidate_hashes"] != expected_hashes
                or digest(target.read_bytes()) != receipt["repaired_candidate_sha256"]
                or _hashes(map(Path, receipt["preserved_artifact_hashes"])) != receipt["preserved_artifact_hashes"]):
            raise ValueError("Collision recovery artifacts changed after the recorded repair")
        return receipt
    generated = sorted(round_dir.glob("generated_system_*.json"))
    observed = {path.name: digest(path.read_bytes()) for path in generated}
    required = {f"generated_system_{index}.json" for index in range(1, candidate_index + 1)}
    if not isinstance(expected_hashes, dict) or set(observed) != required or observed != expected_hashes:
        raise ValueError("All generated candidates must match their explicitly pinned hashes")
    original = read_json(target)
    info, provider = original["memory_type_info"], original["provider_code"]
    alias, module = info["enum_value"], provider["module_name"]
    if not all(re.fullmatch(r"[A-Za-z_]\w*", value) for value in (alias, module)):
        raise ValueError("Invalid collision namespace")
    previous = [read_json(path) for path in generated if path != target]
    if not any(item["memory_type_info"]["enum_value"] == alias
               and item["provider_code"]["module_name"] == module for item in previous):
        raise ValueError("The paid target does not collide with an earlier generated candidate")
    state_path, created_path = round_dir / "state.json", round_dir / "created_system.json"
    checkpoint_path, validated_path = round_dir / "checkpoint.json", round_dir / "validated_systems.json"
    state, created = read_json(state_path), read_json(created_path)
    checkpoint, validated = read_json(checkpoint_path), read_json(validated_path)
    if alias not in created.get("created", []) or alias not in validated.get("validated", []):
        raise ValueError("Earlier collision sibling is not both created and validated")
    failures = [item for item in created.get("failed", []) if _collision(item, alias, module)]
    if not failures:
        raise ValueError("No saved provider-file collision failure matches this target")
    parent = checkpoint.get("parent_provider")
    snapshot = original.get("_orientation_parent", {}).get("snapshot")
    if (not snapshot or snapshot.get("provider_id") != parent
            or snapshot.get("sha256") != checkpoint.get("parent_orientation_sha256")):
        raise ValueError("Paid candidate differs from the checkpoint's parent orientation")
    attempt_path, attempt = _matching_attempt(experiment, job_id, round_dir,
                                               original.get("_raw_llm_output"), parent)
    workspace = directory / "workspace"
    protected = [round_dir / name for name in required if name != target.name]
    protected += [validated_path, checkpoint_path,
                  workspace / "EvolveLab/memory_types.py", workspace / "EvolveLab/config.py"]
    for item in previous:
        value = item["memory_type_info"]["enum_value"]
        protected.append(workspace / "EvolveLab/providers" / f"{item['provider_code']['module_name']}.py")
        for tree in (workspace / "storage" / value, directory / "lineage" / value):
            protected.extend(path for path in tree.rglob("*") if path.is_file())
    preserved = _hashes(protected)
    from .ablation_policy import project_candidate_namespace
    repaired, projection = project_candidate_namespace(original, workspace, work_dir=round_dir,
                                                        job_id=job_id, exclude_config=target)
    if not projection or repaired["memory_type_info"]["enum_value"] == alias:
        raise ValueError("Namespace projection did not resolve the recorded collision")
    if (repaired.get("orientation_items") != original.get("orientation_items")
            or repaired.get("_orientation_parent") != original.get("_orientation_parent")
            or repaired.get("_raw_llm_output") != original.get("_raw_llm_output")):
        raise ValueError("Collision projection changed paid orientation or raw evidence")
    new_created = deepcopy(created)
    new_created["failed"] = [item for item in created.get("failed", []) if not _collision(item, alias, module)]
    new_state = deepcopy(state)
    phase = new_state["phases"]["create"]
    phase.update(completed=False, created_systems=list(created["created"]),
                 failed_systems=[item for item in phase.get("failed_systems", []) if not _collision(item, alias, module)])
    backup = round_dir / f"_candidate_collision_recovery_{candidate_index}"
    if backup.exists():
        raise ValueError("Partial collision recovery exists; inspect its preserved artifacts")
    if (observed != {path.name: digest(path.read_bytes()) for path in generated}
            or _hashes(map(Path, preserved)) != preserved
            or read_json(state_path) != state or read_json(created_path) != created):
        raise ValueError("Stopped job artifacts changed while preparing collision recovery")
    backup.mkdir()
    for path in (target, state_path, created_path, checkpoint_path, validated_path):
        (backup / f"{path.stem}.before{path.suffix}").write_bytes(path.read_bytes())
    repaired["_collision_recovery"] = {"candidate_index": candidate_index, "original_candidate_sha256": observed[target.name],
                                        "original_attempt": attempt["attempt"], "namespace_projection": projection}
    write_json(target, repaired)
    write_json(created_path, new_created)
    write_json(state_path, new_state)
    if _hashes(map(Path, preserved)) != preserved:
        raise RuntimeError("Protected sibling artifacts changed during collision recovery")
    receipt = {"kind": "offline_paid_candidate_namespace_recovery", "job_id": job_id,
               "round": round_name, "candidate_index": candidate_index, "recorded_utc": now(),
               "input_candidate_hashes": observed, "repaired_candidate_sha256": digest(target.read_bytes()),
               "original_attempt": attempt["attempt"], "original_attempt_path": str(attempt_path),
               "original_attempt_usage": attempt["usage"], "original_provider": alias,
               "repaired_provider": repaired["memory_type_info"]["enum_value"],
               "namespace_projection": projection, "preserved_artifact_hashes": preserved,
               "backup_directory": str(backup), "new_model_requests": 0,
               "generated_code_executed": False, "native_creation_and_validation_required": True}
    write_json(receipt_path, receipt)
    return receipt
