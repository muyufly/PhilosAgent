"""A parser fix must reuse the paid answer and judge, then ingest exactly once."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from philoagent.ablation_runtime import digest, items_digest
from philoagent.hss_experiment import read_json, write_json
from philoagent.hss_judge_recovery import recover_saved_judge, saved_judge_recovery_hashes
from philoagent.orientation import bootstrap, load
from test_hss_tasks_runtime import harness, grade_output, run, record


@pytest.fixture
def saved_judge(harness, tmp_path, monkeypatch):
    h = harness
    root = tmp_path / "experiment"
    job_id = "offline-judge-r1"
    directory = root / "jobs" / job_id
    h.callback.run_dir = directory
    h.callback.workspace = directory / "workspace"
    h.callback.protocol = {"model": "offline-model", "resume_incomplete_tasks": True}
    h.dataset = directory / "learning.jsonl"
    h.dataset.parent.mkdir(parents=True)
    h.dataset.write_text("\n".join(json.dumps(task) for task in h.tasks), encoding="utf-8")
    h.output = directory / "evolution/round_00/base_logs"
    source = directory / "workspace/EvolveLab/providers/parent_provider.py"
    source.parent.mkdir(parents=True)
    source.write_text("class ParentProvider: pass\n", encoding="utf-8")
    registry = directory / "workspace/EvolveLab/memory_types.py"
    registry.write_text('MemoryType.PARENT: ("ParentProvider", "parent_provider"),\n', encoding="utf-8")
    bootstrap(directory / "lineage", "parent", ["A revisable criterion."])
    snapshot = load(directory / "lineage", "parent")
    artifact = {"provider": "parent", "module": "parent_provider", "class_name": "ParentProvider",
                "source_path": str(source), "code_sha256": digest(source.read_bytes()),
                "orientation_snapshot": snapshot.to_dict(), "orientation_items_sha256": items_digest(snapshot.items)}
    h.callback._artifact = lambda provider: deepcopy(artifact)
    paid_grade = json.loads(grade_output())
    paid_grade["dimensions"]["argument_placeholder"] = 0
    h.state.judge_output = json.dumps(paid_grade)
    original_parse = h.callback._grade_output

    def legacy_parser_failure(*args, **kwargs):
        raise ValueError("The previous parser rejected this paid output")

    h.callback._grade_output = legacy_parser_failure
    with pytest.raises(ValueError):
        run(h)
    h.callback._grade_output = original_parse
    assert record(h)["status"] == "judge_failed"
    write_json(root / "manifest.json", {"jobs": [{"job_id": job_id}], "protocol": h.callback.protocol})
    write_json(directory / "job_status.json", {"status": "failed", "pid": -1, "error_type": "ValueError"})
    write_json(directory / "runtime_summary.json", {"status": "failed", "error_type": "ValueError"})
    write_json(directory / "evolution/round_00/checkpoint.json", {"parent_provider": "parent", "step_completed": 0})
    write_json(directory / "workspace/storage/parent/memory.json", {"previous_learning": "preserve"})
    attempts = []
    for number, event in enumerate(h.events, 1):
        attempt = {**event, "attempt": number, "job_id": job_id}
        if event["role"] == "judge":
            attempt["response"] = {"choices": [{"finish_reason": "stop", "message": {"content": h.state.judge_output}}]}
        write_json(root / f"budget/attempts/{number:06d}.json", attempt)
        attempts.append(attempt)
    # Six other requests may remain in flight while this one dead job is fixed.
    pending = {}
    for number in range(100, 106):
        pending[str(number)] = {"job_id": "another-job", "pid": -1}
        write_json(root / f"budget/attempts/{number:06d}.json", {"attempt": number, "job_id": "another-job", "status": "pending"})
    write_json(root / "budget/budget.json", {"pending": pending})
    return h, root, job_id, directory, Path("evolution/round_00/base_logs/1.json")


def test_native_runner_resume_reuses_saved_actor_and_judge_and_only_ingests(saved_judge):
    h, root, job_id, directory, relative = saved_judge
    before = record(h)
    pins = saved_judge_recovery_hashes(root, job_id, relative)
    receipt = recover_saved_judge(root, job_id, relative, expected_hashes=pins)
    assert recover_saved_judge(root, job_id, relative, expected_hashes=pins) == receipt
    authorized = record(h)
    for key in ("answer", "agent_result", "judge_output", "agent_messages", "task_identity", "task_execution_id", "status"):
        assert authorized[key] == before[key]
    assert "grade" not in authorized and not authorized["memory_ingestion_complete"]
    assert authorized["api_usage"]["total_tokens"] == 10
    assert receipt["new_model_requests"] == 0
    assert receipt["validated_quality_score"] == 75
    assert receipt["normalization"]["excluded_fields"] == {"dimensions.argument_placeholder": 0}
    assert (Path(receipt["archive"]) / "job_status.json.before").exists()
    h.events.clear()  # The replacement worker has an empty local audit sink.
    h.state.actor_failure = True
    h.state.judge_output = "No new judge request may be made"
    run(h)
    completed = record(h)
    assert completed["status"] == "completed" and completed["quality_score"] == 75
    assert completed["answer"] == before["answer"] and completed["judge_output"] == before["judge_output"]
    assert [event["role"] for event in h.events] == ["memory_ingestion"]
    assert len(h.agents) == 1 and len(h.ingestions) == 1
    assert completed["total_model_tokens"] == 12
    before_events = list(h.events)
    run(h)
    assert h.events == before_events and len(h.ingestions) == 1


@pytest.mark.parametrize("damage", ["stale_hash", "target_pending", "live_job", "ingestion_started",
                                  "response_mismatch", "source_changed", "wrong_identity", "unparseable"])
def test_unsafe_saved_judge_recovery_preserves_all_target_artifacts(saved_judge, damage, monkeypatch):
    h, root, job_id, directory, relative = saved_judge
    pins = saved_judge_recovery_hashes(root, job_id, relative)
    path = directory / relative
    value = read_json(path)
    if damage == "stale_hash":
        pins[str(relative).replace("\\", "/")] = "wrong"
    elif damage == "target_pending":
        budget = read_json(root / "budget/budget.json")
        budget["pending"]["900"] = {"job_id": job_id}
        write_json(root / "budget/budget.json", budget)
        write_json(root / "budget/attempts/000900.json", {"job_id": job_id})
    elif damage == "live_job":
        monkeypatch.setattr("philoagent.hss_judge_recovery.process_owned", lambda status: True)
    elif damage == "ingestion_started":
        value["memory_ingestion_started"] = True
        write_json(path, value)
        pins = saved_judge_recovery_hashes(root, job_id, relative)
    elif damage == "response_mismatch":
        attempt = read_json(root / "budget/attempts/000002.json")
        attempt["response"]["choices"][0]["message"]["content"] += " changed"
        write_json(root / "budget/attempts/000002.json", attempt)
    elif damage == "source_changed":
        source = directory / "workspace/EvolveLab/providers/parent_provider.py"
        source.write_text("class ParentProvider: changed = True\n", encoding="utf-8")
        pins = saved_judge_recovery_hashes(root, job_id, relative)
    elif damage == "wrong_identity":
        value["task_identity"] = "another task"
        write_json(path, value)
        pins = saved_judge_recovery_hashes(root, job_id, relative)
    else:
        value["judge_output"] = "invalid paid output"
        write_json(path, value)
        pins = saved_judge_recovery_hashes(root, job_id, relative)
    before = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    events = list(h.events)
    with pytest.raises(ValueError):
        recover_saved_judge(root, job_id, relative, expected_hashes=pins)
    assert before == {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert h.events == events
    assert not (root / "revisions").exists()


@pytest.mark.parametrize("relative", ["../elsewhere.json", "evolution/round_00/base_logs/_recovery/1.json"])
def test_judge_recovery_scope_refuses_paths_outside_active_task_records(saved_judge, relative):
    _, root, job_id, _, _ = saved_judge
    with pytest.raises((ValueError, FileNotFoundError)):
        saved_judge_recovery_hashes(root, job_id, relative)
