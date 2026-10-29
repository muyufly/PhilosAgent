"""Reopen a native finalist comparison that advanced after an interruption.

The rollback preserves all paid task records, generated providers, orientation,
memory and accounting. Separate task-recovery authorization owns failed phases.
"""

from copy import deepcopy
from pathlib import Path
import shutil

from .ablation_runtime import digest
from .hss_budget import _file_lock
from .hss_evaluation import native_quality_summary
from .hss_experiment import now, process_owned, read_json, write_json
from .orientation import load


def _job(root, job_id):
    root = Path(root).resolve(strict=True)
    directory = (root / "jobs" / job_id).resolve(strict=True)
    if not directory.is_relative_to(root / "jobs") or directory.parent != root / "jobs":
        raise ValueError("Finalist recovery target escaped the experiment jobs directory")
    return root, directory


def _mutable_files(directory):
    files = [directory / "evolution/evolve_state.json", directory / "evolution/round_00/round_summary.json"]
    files.extend(path for path in (
        directory / "evolution/round_00/checkpoint.json", directory / "runtime_summary.json",
        directory / "result.json", directory / "evolution/auto_history.json") if path.exists())
    next_round = directory / "evolution/round_01"
    if next_round.exists():
        files.extend(path for path in next_round.rglob("*") if path.is_file())
    return sorted(files)


def finalist_recovery_hashes(root, job_id):
    """Return the exact file hashes a caller approves before the rollback."""
    _, directory = _job(root, job_id)
    return {path.relative_to(directory).as_posix(): digest(path.read_bytes())
            for path in _mutable_files(directory)}


def _protected_files(directory):
    paths = []
    for relative in ("lineage", "workspace/storage", "workspace/EvolveLab/providers", "evolution/round_00"):
        folder = directory / relative
        if folder.exists():
            paths.extend(path for path in folder.rglob("*") if path.is_file()
                         and path != directory / "evolution/round_00/checkpoint.json"
                         and path != directory / "evolution/round_00/round_summary.json")
    return {path.relative_to(directory).as_posix(): digest(path.read_bytes()) for path in sorted(paths)}


def restore_incomplete_finalists(root, job_id, expected_hashes, *, revision="repair-r7"):
    """Archive a false round completion and resume its original step-4 tasks.

    Failed-provider results must be absent from the active checkpoint because
    native resume treats every cached provider key as completed, even failures.
    Paid task records inside those provider folders remain available for reuse.
    """
    root, directory = _job(root, job_id)
    archive = (root / "revisions" / revision / "finalist_recovery" / job_id).resolve()
    if not archive.is_relative_to(root / "revisions"):
        raise ValueError("Finalist archive escaped the experiment revisions directory")
    receipt_path = archive / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for relative, expected in receipt["restored_hashes"].items():
            if digest((directory / relative).read_bytes()) != expected:
                raise ValueError("Recovered finalist state changed after rollback")
        return receipt
    if archive.exists():
        raise ValueError("Partial finalist rollback archive exists; inspect before writing")
    with _file_lock(root / "budget/ledger.lock", 30):
        ledger_path = root / "budget/budget.json"
        ledger_before = ledger_path.read_bytes()
        ledger = read_json(ledger_path)
        if ledger.get("pending"):
            raise ValueError("Finalist recovery requires zero pending requests")
        status_path = directory / "job_status.json"
        status = read_json(status_path)
        if process_owned(status.get("worker_source") or status):
            raise ValueError("Finalist recovery target worker is still active")
        hashes = finalist_recovery_hashes(root, job_id)
        if hashes != expected_hashes:
            raise ValueError("Finalist recovery inputs differ from the approved hashes")
        state_path = directory / "evolution/evolve_state.json"
        summary_path = directory / "evolution/round_00/round_summary.json"
        state, summary = read_json(state_path), read_json(summary_path)
        if (state.get("round") != 1 or state.get("dataset_cursor") != 4 or summary.get("round") != 0
                or summary.get("dataset_cursor_after") != 4
                or state.get("best_provider") != summary.get("winner")
                or summary.get("best_provider_after") != summary.get("winner")
                or any(item.get("round", -1) > 0 for item in state.get("history", []))):
            raise ValueError("Expected exactly one falsely advanced native round")
        parent = summary["best_provider_before"]
        parent_snapshot = load(directory / "lineage", parent)
        finalists = summary.get("finalist_results", {})
        successful = {key: deepcopy(value) for key, value in finalists.items() if value.get("success") is True}
        failed = {key: value for key, value in finalists.items() if value.get("success") is not True}
        if not successful or not failed:
            raise ValueError("Round does not contain a mixture of complete and interrupted finalists")
        round_dir = directory / "evolution/round_00"
        for provider, result in successful.items():
            records = [read_json(round_dir / "eval_finalists" / provider / f"{index + 1}.json")
                       for index in summary["finalist_tasks"]]
            actual = native_quality_summary(records)
            if (not actual["success"] or abs(actual["summary"]["quality_score"]
                    - result.get("summary", {}).get("quality_score", -1)) > 1e-6):
                raise ValueError("Cached successful finalist differs from its paid task grades")
        for provider in failed:
            for path in (round_dir / "eval_finalists" / provider).glob("[0-9]*.json"):
                record = read_json(path)
                if ((record.get("memory_ingestion_started") or record.get("status") == "ingestion_failed")
                        and not record.get("memory_ingestion_complete")):
                    raise ValueError("Interrupted finalist ingestion requires separate memory reconciliation")
        next_round = directory / "evolution/round_01"
        if next_round.exists():
            for path in next_round.rglob("[0-9]*.json"):
                record = read_json(path)
                if (record.get("total_model_tokens", 0) or record.get("api_usage", {}).get("request_count", 0)
                        or record.get("answer") or record.get("grade") or record.get("memory_ingestion_started")
                        or record.get("memory_ingestion_complete") or record.get("status") == "completed"):
                    raise ValueError("Next-round paid work requires a broader recovery plan")
        result_path = directory / "result.json"
        if result_path.exists() and read_json(result_path).get("status") == "complete":
            raise ValueError("Completed terminal result cannot be rolled back by this helper")
        checkpoint = {"parent_provider": parent, "parent_orientation_sha256": parent_snapshot.sha256,
            "step_completed": 4, "batch_indices": summary["batch_tasks"],
            "finalist_tasks": summary["finalist_tasks"], "eval_results": deepcopy(summary["eval_results"]),
            "finalist_results": successful,
            "validated_systems": summary["evolution"]["validated_systems"],
            "evolution_results": {key: deepcopy(summary["evolution"].get(key))
                                  for key in ("analysis", "generation", "creation", "validation")},
            "evolution_completed": {key: True for key in ("analysis", "generation", "creation", "validation")}}
        restored = deepcopy(state)
        restored.update(round=0, dataset_cursor=0, best_provider=parent,
                        best_orientation_sha256=parent_snapshot.sha256,
                        history=[item for item in state.get("history", []) if item.get("round", 0) < 0])
        protected = _protected_files(directory)
        # All semantic checks precede the first archive or state mutation.
        for path in _mutable_files(directory):
            destination = archive / path.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        shutil.copy2(status_path, archive / "job_status.before.json")
        if next_round.exists():
            destination = archive / "next_round_original"
            if not next_round.resolve().is_relative_to(directory / "evolution") or not destination.is_relative_to(archive):
                raise ValueError("Next-round archive escaped the verified native directories")
            shutil.move(str(next_round), str(destination))
        for path in (summary_path, directory / "runtime_summary.json", result_path,
                     directory / "evolution/auto_history.json"):
            if path.exists():
                path.unlink()
        write_json(state_path, restored)
        checkpoint_path = round_dir / "checkpoint.json"
        write_json(checkpoint_path, checkpoint)
        if _protected_files(directory) != protected or ledger_path.read_bytes() != ledger_before:
            raise RuntimeError("Finalist rollback unexpectedly changed protected paid artifacts")
        receipt = {"revision": revision, "job_id": job_id, "recorded_utc": now(), "archive": str(archive),
            "approved_hashes": hashes, "invalidated_winner": summary["winner"],
            "retained_finalists": list(successful), "reopened_finalists": list(failed),
            "reused_original_finalist_tasks": summary["finalist_tasks"],
            "restored_hashes": {path.relative_to(directory).as_posix(): digest(path.read_bytes())
                                for path in (state_path, checkpoint_path)},
            "preserved_artifact_hashes": protected, "same_budget_ledger": str(ledger_path),
            "task_phase_authorization": "separate audited recovery; task records unchanged",
            "native_memory_preserved": True, "paid_model_requests": 0}
        write_json(receipt_path, receipt)
        return receipt
