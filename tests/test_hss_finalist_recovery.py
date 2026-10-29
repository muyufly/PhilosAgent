"""Offline rollback of a winner chosen from an interrupted comparison."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from philoagent.hss_evaluation import RUBRICS, native_quality_summary, validate_grade
from philoagent.hss_experiment import read_json, write_json
from philoagent.orientation import bootstrap, load


def task_record(task_id, score=3):
    grade = validate_grade({"dimensions": {name: {"score": score, "reason": "Recorded evidence."}
        for name in RUBRICS["humanities"]}, "overall_comment": "A complete report."}, "humanities")
    return {"status": "completed", "task_id": task_id, "domain": "humanities", "grade": grade,
            "answer": "A paid completed report.", "total_model_tokens": 20,
            "memory_ingestion_complete": True}


@pytest.fixture
def interrupted(tmp_path):
    root = tmp_path / "experiment"
    job = root / "jobs/humanities-native-r1"
    evolution, round_dir = job / "evolution", job / "evolution/round_00"
    round_dir.mkdir(parents=True)
    bootstrap(job / "lineage", "parent", ["Parent orientation."])
    bootstrap(job / "lineage", "candidate", ["Candidate orientation."])
    memory = job / "workspace/storage/candidate/memory.json"
    write_json(memory, {"completed_tasks": ["task-1"], "preserve": True})
    provider = job / "workspace/EvolveLab/providers/candidate.py"
    provider.parent.mkdir(parents=True)
    provider.write_text("class Candidate: pass\n", encoding="utf-8")
    write_json(root / "budget/budget.json", {"pending": {}, "blocked_reason": "maintenance_test"})
    write_json(job / "job_status.json", {"status": "failed", "pid": -1})
    write_json(job / "runtime_summary.json", {"status": "failed", "rounds_completed": 1})
    write_json(evolution / "evolve_state.json", {"round": 1, "dataset_name": "writingbench",
        "dataset_cursor": 4, "best_provider": "parent", "best_orientation_sha256": load(job / "lineage", "parent").sha256,
        "history": [{"round": 0, "winner": "parent", "cursor": 4}]})
    results = {}
    for name, score in (("parent", 3), ("candidate", 4)):
        folder = round_dir / "eval_batch" / name
        results[name] = {**native_quality_summary([task_record("task-0", score), task_record("task-1", score)]),
                         "logs_dir": str(folder)}
    finalists = round_dir / "eval_finalists"
    parent_records = [task_record(f"task-{i}") for i in (1, 2, 3)]
    for index, record in zip((1, 2, 3), parent_records):
        write_json(finalists / "parent" / f"{index + 1}.json", record)
    write_json(finalists / "candidate/2.json", task_record("task-1", 4.7))
    write_json(finalists / "candidate/3.json", {"status": "actor_failed", "task_id": "task-2",
        "task_execution_id": "paid-before-transport-error", "answer": "", "partial_agent_messages": ["retained"],
        "memory_ingestion_complete": False, "total_model_tokens": 10})
    summary = {"round": 0, "timestamp": "2026-10-29T21:14:18.851466", "dataset": "writingbench",
        "batch_tasks": [0, 1], "finalist_tasks": [1, 2, 3],
        "evolution": {"analysis": {"success": True}, "generation": [], "creation": [], "validation": [],
                      "validated_systems": ["candidate"]},
        "eval_results": results, "finalist_results": {
            "parent": {**native_quality_summary(parent_records), "logs_dir": str(finalists / "parent")},
            "candidate": {"success": False, "error": "Shared budget blocked: unknown_usage"}},
        "winner": "parent", "best_provider_before": "parent", "best_provider_after": "parent",
        "dataset_cursor_after": 4}
    write_json(round_dir / "round_summary.json", summary)
    write_json(evolution / "round_01/checkpoint.json", {"parent_provider": "parent"})
    write_json(evolution / "round_01/base_logs/5.json", {"status": "actor_failed", "answer": "",
        "total_model_tokens": 0, "memory_ingestion_complete": False})
    return root, job, summary


def test_restore_keeps_paid_finalist_artifacts_and_reopens_only_the_failed_provider(interrupted):
    from philoagent.hss_finalist_recovery import finalist_recovery_hashes, restore_incomplete_finalists
    root, job, summary = interrupted
    memory = (job / "workspace/storage/candidate/memory.json").read_bytes()
    failed = job / "evolution/round_00/eval_finalists/candidate/3.json"
    paid = failed.read_bytes()
    ledger = (root / "budget/budget.json").read_bytes()
    evidence = restore_incomplete_finalists(root, job.name, finalist_recovery_hashes(root, job.name))
    state = read_json(job / "evolution/evolve_state.json")
    assert state["round"] == 0 and state["dataset_cursor"] == 0 and state["best_provider"] == "parent"
    checkpoint = read_json(job / "evolution/round_00/checkpoint.json")
    assert checkpoint["step_completed"] == 4
    assert checkpoint["finalist_tasks"] == summary["finalist_tasks"]
    assert checkpoint["eval_results"] == summary["eval_results"]
    assert set(checkpoint["finalist_results"]) == {"parent"}
    assert checkpoint["validated_systems"] == ["candidate"]
    assert failed.read_bytes() == paid
    assert (job / "workspace/storage/candidate/memory.json").read_bytes() == memory
    assert (root / "budget/budget.json").read_bytes() == ledger
    assert not (job / "evolution/round_01").exists()
    assert not (job / "evolution/round_00/round_summary.json").exists()
    assert read_json(Path(evidence["archive"]) / "evolution/round_00/round_summary.json") == summary


@pytest.mark.parametrize("phase", ["actor", "judge"])
def test_actual_native_resume_finishes_missing_comparison_before_selecting_new_winner(interrupted, monkeypatch, phase):
    from philoagent.hss_finalist_recovery import finalist_recovery_hashes, restore_incomplete_finalists
    from test_evolution_integration import AUTO
    root, job, summary = interrupted
    failure = job / "evolution/round_00/eval_finalists/candidate/3.json"
    if phase == "judge":
        record = read_json(failure)
        record.update(status="judge_failed", answer="The complete paid report already exists.",
                      agent_messages=[{"role": "assistant", "content": "The complete paid report already exists."}])
        write_json(failure, record)
    restore_incomplete_finalists(root, job.name, finalist_recovery_hashes(root, job.name))
    dataset = job / "learning.jsonl"
    dataset.write_text("".join(json.dumps({"task_id": f"task-{i}", "question": f"Question {i}"}) + "\n"
                               for i in range(8)), encoding="utf-8")
    calls, phases = [], []
    paid_first = (job / "evolution/round_00/eval_finalists/candidate/2.json").read_bytes()

    def callback(dataset_name, provider, indices, dataset_file, output_dir):
        assert provider in {"candidate", "parent"}
        assert "eval_finalists" in output_dir.parts
        assert not (job / "evolution/round_00/round_summary.json").exists()
        calls.append((provider, list(indices)))
        records = []
        for index in indices:
            path = output_dir / f"{index + 1}.json"
            previous = read_json(path) if path.exists() else {}
            if previous.get("status") == "completed":
                records.append(previous)
                continue
            phases.append("judge" if previous.get("answer") else "actor")
            record = task_record(f"task-{index}", 4.5)
            if previous.get("answer"):
                record["answer"] = previous["answer"]
            write_json(path, record)
            records.append(record)
        write_json(output_dir / "_quality/summary.json", native_quality_summary(records))
        return output_dir

    class Aggregator:
        def __init__(self, output):
            self.output = Path(output)

        def aggregate(self):
            return read_json(self.output / "_quality/summary.json")

    monkeypatch.setenv("PHILO_LINEAGE_ROOT", str(job / "lineage"))
    monkeypatch.setattr(AUTO, "TrajectoryFeedbackAggregator", Aggregator)
    native = AUTO.AutoEvolver(work_root=str(job / "evolution"), dataset_name="writingbench",
        run_provider=callback, default_provider="parent", num_systems=1, task_batch_x=2,
        top_t=2, extra_sample_y=1, datasets_config={"writingbench": str(dataset)},
        clear_storage_per_round=False, max_workers=1)
    monkeypatch.setattr(native, "_run_memory_evolver", lambda *args, **kwargs: deepcopy(summary["evolution"]))
    native.run(num_rounds=1)
    # This upstream revision resets step_completed to 3 after reloading the
    # batch results. The complete provider's callback is invoked again, but
    # every one of its paid task records is reused without actor/judge work.
    assert calls == [("candidate", [1, 2, 3]), ("parent", [1, 2, 3])]
    assert phases == [phase, "actor"]
    assert (job / "evolution/round_00/eval_finalists/candidate/2.json").read_bytes() == paid_first
    completed = read_json(job / "evolution/round_00/round_summary.json")
    assert all(result["success"] for result in completed["finalist_results"].values())
    assert completed["winner"] == "candidate"
    assert native.state["round"] == 1 and native.state["dataset_cursor"] == 4
    assert native.state["best_provider"] == "candidate"


def test_actual_native_resume_guard_keeps_round_open_when_finalist_fails_again(interrupted, monkeypatch):
    from types import MethodType
    from philoagent.hss_budget import BudgetRefused
    from philoagent.hss_comparison import HSSComparisonIncomplete, guard_comparison
    from philoagent.hss_finalist_recovery import finalist_recovery_hashes, restore_incomplete_finalists
    from test_evolution_integration import AUTO
    root, job, summary = interrupted
    restore_incomplete_finalists(root, job.name, finalist_recovery_hashes(root, job.name))
    dataset = job / "learning.jsonl"
    dataset.write_text("".join(json.dumps({"task_id": f"task-{i}", "question": f"Question {i}"}) + "\n"
                               for i in range(8)), encoding="utf-8")
    records_dir = job / "evolution/round_00/eval_finalists"
    completed_before = {p: p.read_bytes() for p in records_dir.glob("*/*.json")
                        if read_json(p).get("status") == "completed"}
    reused, attempted = [], []

    def callback(dataset_name, provider, indices, dataset_file, output_dir):
        for index in indices:
            path = output_dir / f"{index + 1}.json"
            if path.exists() and read_json(path).get("status") == "completed":
                reused.append((provider, index))
                continue
            attempted.append((provider, index))
            raise BudgetRefused("Shared budget blocked: offline second interruption")
        return output_dir

    class Aggregator:
        def __init__(self, output):
            self.output = Path(output)

        def aggregate(self):
            return native_quality_summary([read_json(path) for path in sorted(self.output.glob("[0-9]*.json"))])

    monkeypatch.setenv("PHILO_LINEAGE_ROOT", str(job / "lineage"))
    monkeypatch.setattr(AUTO, "TrajectoryFeedbackAggregator", Aggregator)
    native = AUTO.AutoEvolver(work_root=str(job / "evolution"), dataset_name="writingbench",
        run_provider=callback, default_provider="parent", num_systems=1, task_batch_x=2,
        top_t=2, extra_sample_y=1, datasets_config={"writingbench": str(dataset)},
        clear_storage_per_round=False, max_workers=1)
    monkeypatch.setattr(native, "_run_memory_evolver", lambda *args, **kwargs: deepcopy(summary["evolution"]))
    monkeypatch.setattr(native, "_evaluate_providers_parallel", MethodType(
        guard_comparison(AUTO.AutoEvolver._evaluate_providers_parallel), native))
    original_select, selections = native._select_top, []

    def select(results, k=1):
        selections.append(k)
        return original_select(results, k=k)

    monkeypatch.setattr(native, "_select_top", select)
    with pytest.raises(HSSComparisonIncomplete, match="candidate"):
        native.run(num_rounds=1)
    assert selections == [2]  # Batch shortlist only; no winner selection.
    assert attempted == [("candidate", 2)]
    assert set(reused) == {("candidate", 1), ("parent", 1), ("parent", 2), ("parent", 3)}
    assert all(path.read_bytes() == before for path, before in completed_before.items())
    state = read_json(job / "evolution/evolve_state.json")
    assert state["round"] == 0 and state["dataset_cursor"] == 0 and state["best_provider"] == "parent"
    assert state["history"] == []
    assert not (job / "evolution/round_00/round_summary.json").exists()
    evidence = read_json(records_dir / "_comparison_incomplete.json")
    assert evidence["winner_selection_allowed"] is False
    assert evidence["incomplete"] == ["candidate"]
    assert evidence["results"]["parent"]["success"] is True


@pytest.mark.parametrize("damage", ["stale_hash", "pending", "live_worker", "paid_next_round", "ambiguous_ingestion"])
def test_rollback_refuses_unsafe_inputs_before_archiving(interrupted, monkeypatch, damage):
    from philoagent.hss_finalist_recovery import finalist_recovery_hashes, restore_incomplete_finalists
    root, job, _ = interrupted
    hashes = finalist_recovery_hashes(root, job.name)
    if damage == "stale_hash":
        state = read_json(job / "evolution/evolve_state.json")
        state["extra"] = "changed after approval"
        write_json(job / "evolution/evolve_state.json", state)
    elif damage == "pending":
        write_json(root / "budget/budget.json", {"pending": {"99": {"pid": -1}}})
    elif damage == "live_worker":
        monkeypatch.setattr("philoagent.hss_finalist_recovery.process_owned", lambda record: True)
    elif damage == "paid_next_round":
        record = read_json(job / "evolution/round_01/base_logs/5.json")
        record["total_model_tokens"] = 100
        write_json(job / "evolution/round_01/base_logs/5.json", record)
        hashes = finalist_recovery_hashes(root, job.name)
    else:
        record = read_json(job / "evolution/round_00/eval_finalists/candidate/3.json")
        record["memory_ingestion_started"] = True
        write_json(job / "evolution/round_00/eval_finalists/candidate/3.json", record)
    before = {p.relative_to(job): p.read_bytes() for p in job.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        restore_incomplete_finalists(root, job.name, hashes)
    assert not (root / "revisions").exists()
    assert before == {p.relative_to(job): p.read_bytes() for p in job.rglob("*") if p.is_file()}
