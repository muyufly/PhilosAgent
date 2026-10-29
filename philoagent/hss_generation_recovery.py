"""Reuse a settled native candidate response after a controller-only rejection.

Run one job per fresh interpreter. Generated code is parsed and projected through
the native generator and current factor policy; it is never imported or executed
here. Native creation and validation remain part of the resumed experiment.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

from .ablation_runtime import digest
from .hss_experiment import now, process_owned, read_json, write_json


def _raw_response(path):
    raw = path.read_text(encoding="utf-8")
    match = re.fullmatch(
        r"Temperature: ([0-9.]+)\nModel: ([^\n]+)\nTimestamp: ([^\n]+)\n=+\n\n([\s\S]+)", raw)
    if not match:
        raise ValueError("Saved candidate lacks its native raw-response metadata")
    return raw, match.group(4), {"temperature": float(match.group(1)),
                               "model_id": match.group(2), "timestamp": match.group(3)}


def _matching_attempt(experiment, job_id, round_dir, response, parent):
    matches = []
    for path in sorted((experiment / "budget/attempts").glob("*.json")):
        attempt = read_json(path)
        if (attempt.get("job_id") != job_id or attempt.get("role") != "generation"
                or attempt.get("provider") != parent
                or Path(attempt.get("round_dir", "")).resolve() != round_dir):
            continue
        choices = (attempt.get("response") or {}).get("choices", [])
        if (len(choices) != 1 or choices[0].get("finish_reason") != "stop"
                or choices[0].get("message", {}).get("content") != response):
            continue
        usage = attempt.get("usage")
        if (attempt.get("status") != "success" or not isinstance(usage, dict)
                or any(type(usage.get(key)) is not int or usage[key] < 0 for key in
                       ("prompt_tokens", "completion_tokens", "total_tokens"))):
            continue
        matches.append((path, attempt))
    if len(matches) != 1:
        raise ValueError("Saved candidate must match exactly one complete settled generation response")
    return matches[0]


class _SavedResponseClient:
    """One-shot transport fixture; it has no SDK or network access."""

    def __init__(self, response):
        self.response, self.calls = response, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        if self.calls:
            raise RuntimeError("Saved candidate transport may be replayed only once")
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.response))])


def recover_saved_candidate(experiment_root, job_id, *, round_name="round_00", raw_path=None,
                            candidate_index=1, recovery_scope="experiment",
                            expected_existing_hashes=None):
    """Hydrate one native candidate from an already paid response.

    Default recovery requires drained maintenance. Explicit ``stopped_job``
    recovery can repair an exited job while other jobs run, with exact hashes
    pinning every earlier candidate. No model/provider code is executed here.
    """
    if type(candidate_index) is not int or candidate_index < 1:
        raise ValueError("Candidate index must be a positive integer")
    if recovery_scope not in ("experiment", "stopped_job"):
        raise ValueError("Unknown candidate recovery scope")
    experiment = Path(experiment_root).resolve(strict=True)
    manifest = read_json(experiment / "manifest.json")
    job = next((item for item in manifest["jobs"] if item["job_id"] == job_id), None)
    if job is None:
        raise ValueError("Unknown experiment job")
    directory = (experiment / "jobs" / job_id).resolve(strict=True)
    round_dir = (directory / "evolution" / round_name).resolve(strict=True)
    if not round_dir.is_relative_to(directory / "evolution"):
        raise ValueError("Candidate round escaped the native evolution directory")
    budget = read_json(experiment / "budget/budget.json")
    reserved = budget.get("reserved_unknown_attempts", [])
    if budget.get("unknown_usage_attempts", 0) != len(reserved):
        raise ValueError("Candidate recovery requires accounted historical usage")
    if recovery_scope == "experiment":
        if (not str(budget.get("blocked_reason", "")).startswith("maintenance_") or budget.get("pending")):
            raise ValueError("Candidate recovery requires drained maintenance with accounted usage")
        process_paths = (experiment / "supervisor.json", directory / "job_status.json")
    else:
        status = read_json(directory / "job_status.json")
        if status.get("status") not in ("failed", "interrupted"):
            raise ValueError("Isolated candidate recovery requires an exited failed or interrupted job")
        for number, pending in budget.get("pending", {}).items():
            attempt = read_json(experiment / "budget" / "attempts" / f"{int(number):06d}.json")
            if attempt.get("job_id", pending.get("job_id")) == job_id:
                raise ValueError("Isolated candidate recovery has a pending target-job request")
        process_paths = (directory / "job_status.json",)
    for path in process_paths:
        if path.exists() and process_owned(read_json(path)):
            raise ValueError("Candidate recovery requires stopped native workers")
    target = round_dir / f"generated_system_{candidate_index}.json"
    suffix = "" if candidate_index == 1 else f"_{candidate_index}"
    receipt_path = round_dir / f"saved_generation_recovery{suffix}.json"
    preserved = {path.name: digest(path.read_bytes()) for path in round_dir.glob("generated_system_*.json")
                 if path != target}
    required_names = {f"generated_system_{index}.json" for index in range(1, candidate_index)}
    if candidate_index > 1 and (raw_path is None or not isinstance(expected_existing_hashes, dict)):
        raise ValueError("Later candidate recovery requires an explicit raw response and earlier candidate hashes")
    if candidate_index > 1 and (set(preserved) != required_names or preserved != expected_existing_hashes):
        raise ValueError("Earlier generated candidates differ from the explicitly preserved hashes")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if digest(target.read_bytes()) != receipt["generated_file_sha256"]:
            raise ValueError("Recovered native candidate changed after offline recovery")
        if receipt.get("preserved_existing_candidates", {}) != preserved:
            raise ValueError("Earlier native candidate changed after offline recovery")
        if raw_path is not None and (Path(raw_path).resolve(strict=True) != Path(receipt["original_raw_path"])
                                    or digest(Path(raw_path).read_bytes()) != receipt["original_raw_sha256"]):
            raise ValueError("Recovery receipt refers to a different saved raw response")
        return receipt
    if target.exists() or (candidate_index == 1 and preserved):
        raise ValueError("Native generated candidates already exist; do not overwrite them")
    files = [Path(raw_path).resolve(strict=True)] if raw_path else sorted(round_dir.glob("llm_raw_output_*.txt"))
    if len(files) != 1 or files[0].parent.resolve() != round_dir:
        raise ValueError("Select exactly one saved raw candidate in the requested native round")
    raw_path = files[0]
    raw, response, metadata = _raw_response(raw_path)
    state_path, checkpoint_path = round_dir / "state.json", round_dir / "checkpoint.json"
    before_state, before_checkpoint = read_json(state_path), read_json(checkpoint_path)
    analysis_state = before_state["phases"]["analyze"]
    parent = analysis_state.get("default_provider")
    if not analysis_state.get("completed") or before_checkpoint.get("parent_provider") != parent:
        raise ValueError("Saved generation has no completed analysis for the current native parent")
    analysis_path = Path(analysis_state["output"]).resolve(strict=True)
    if analysis_path != round_dir / "analysis_report.json":
        raise ValueError("Saved analysis is outside the candidate's native round")
    analysis = read_json(analysis_path)
    from .orientation import load
    snapshot = load(directory / "lineage", parent)
    if (analysis.get("orientation_snapshot") != snapshot.to_dict()
            or before_checkpoint.get("parent_orientation_sha256") != snapshot.sha256):
        raise ValueError("Saved analysis differs from the current parent orientation")
    for name in preserved:
        prior = read_json(round_dir / name)
        if (prior.get("_generation_metadata", {}).get("parent_provider") != parent
                or prior.get("_orientation_parent", {}).get("snapshot") != snapshot.to_dict()):
            raise ValueError("Earlier generated candidate differs from the current native parent")
    attempt_path, attempt = _matching_attempt(experiment, job_id, round_dir, response, parent)
    if metadata["model_id"] != manifest["protocol"]["generation_model"]:
        raise ValueError("Saved generation model differs from the experiment model")

    # Import only native controller code after selecting this job's workspace.
    # The supplied one-shot client is the sole transport the generator can use.
    from .live_smoke import activate_workspace
    from .ablation_policy import install_native_policy
    activate_workspace(directory)
    from MemEvolve.phases.phase_generator import PhaseGenerator
    client = _SavedResponseClient(response)
    restore = install_native_policy(directory, job, compact_provider=False,
                                    generated_code_policy="native_local")
    try:
        with tempfile.TemporaryDirectory(prefix="hss-saved-generation-") as scratch, redirect_stdout(io.StringIO()):
            generator = PhaseGenerator(client, metadata["model_id"], Path(scratch), default_provider=parent)
            result = generator.run_generation(analysis, creativity_index=manifest["protocol"].get("creativity_index", .5))
    finally:
        restore()
    if not result.get("success") or not result.get("config") or client.calls != 1:
        raise ValueError("Saved native candidate failed parsing or current factor projection")
    config = result["config"]
    config["_generation_metadata"] = {**metadata, "parent_provider": parent}
    config["_generation_recovery"] = {
        "kind": "offline_saved_response_reuse", "original_attempt": attempt["attempt"],
        "original_raw_path": str(raw_path), "original_raw_sha256": digest(raw_path.read_bytes()),
        "original_response_sha256": digest(response.encode("utf-8")),
        "new_model_requests": 0, "native_creation_and_validation_required": True,
    }
    state = deepcopy(before_state)
    state["phases"]["generate"] = {**state["phases"].get("generate", {}), "completed": True, "output": str(target),
        "creativity_index": manifest["protocol"].get("creativity_index", .5), "timestamp": now(),
        "saved_response_recovery": config["_generation_recovery"]}
    checkpoint = deepcopy(before_checkpoint)
    checkpoint.setdefault("evolution_completed", {})["analysis"] = True
    checkpoint.setdefault("evolution_results", {}).setdefault("analysis", {
        "success": True, "report_path": str(analysis_path), "stats": analysis.get("stats", {})})
    backups = round_dir / f"_saved_generation_recovery{suffix}"
    if backups.exists():
        raise ValueError("Partial saved-candidate recovery exists; inspect it before writing")
    if target.exists() or preserved != {path.name: digest(path.read_bytes()) for path in round_dir.glob("generated_system_*.json")}:
        raise ValueError("Native candidate files changed during offline recovery")
    if read_json(state_path) != before_state or read_json(checkpoint_path) != before_checkpoint:
        raise ValueError("Native controller state changed during offline recovery")
    backups.mkdir()
    write_json(backups / "state.before.json", before_state)
    write_json(backups / "checkpoint.before.json", before_checkpoint)
    write_json(target, config)
    write_json(state_path, state)
    write_json(checkpoint_path, checkpoint)
    receipt = {**config["_generation_recovery"], "job_id": job_id, "round": round_name,
        "candidate_index": candidate_index, "recovery_scope": recovery_scope,
        "preserved_existing_candidates": preserved,
        "recorded_utc": now(), "original_attempt_path": str(attempt_path),
        "original_attempt_usage": attempt["usage"], "generated_config": str(target),
        "generated_file_sha256": digest(target.read_bytes()), "provider": config["memory_type_info"]["enum_value"],
        "state_before": str(backups / "state.before.json"),
        "checkpoint_before": str(backups / "checkpoint.before.json"),
        "analysis_reused": True, "generated_code_executed": False}
    write_json(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("job_id")
    parser.add_argument("--round", default="round_00")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--scope", choices=("experiment", "stopped_job"), default="experiment")
    parser.add_argument("--existing-hashes", type=Path,
                        help="JSON mapping of earlier generated candidate filenames to SHA-256 hashes")
    args = parser.parse_args()
    result = recover_saved_candidate(args.experiment_root, args.job_id, round_name=args.round, raw_path=args.raw,
                                    candidate_index=args.candidate_index, recovery_scope=args.scope,
                                    expected_existing_hashes=read_json(args.existing_hashes) if args.existing_hashes else None)
    print(json.dumps(result, ensure_ascii=False))
