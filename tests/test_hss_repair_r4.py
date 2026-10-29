"""Correct the delivered text before judging and restore consistent memory."""

from pathlib import Path
import pytest

from philoagent.hss_experiment import write_json
from philoagent.hss_repair import _records
from philoagent.hss_repair_r4 import corrected_record


def test_full_paid_answer_requires_new_grade_and_fresh_ingestion_without_actor_replay(tmp_path):
    old = {"task_execution_id": "paid", "status": "completed", "answer": "Report start",
           "agent_result": "Report start", "agent_messages": [{"role": "assistant", "content": "Full original trajectory"}],
           "grade": {"quality_score": 10.8}, "quality_score": 10.8,
           "judge_output": "Old grade of prefix", "memory_ingestion_started": True,
           "memory_ingestion_complete": True, "judge_recovery_attempts": 1}
    answer = "Report start\nFull argument and complete appendix."
    payload = {"tools": [{"arguments": {"answer": answer}}]}
    usage = {"total_tokens": 25000, "unrecovered_failed_attempt_ids": [],
             "reserved_unknown_attempt_ids": []}
    fixed = corrected_record(old, payload, {"source_attempt": 705}, usage, tmp_path, tmp_path / "archive")
    assert fixed["answer"] == fixed["agent_result"] == answer
    assert fixed["agent_messages"] == old["agent_messages"]
    assert fixed["status"] == "answer_saved"
    assert fixed["memory_ingestion_complete"] is False
    assert "memory_ingestion_started" not in fixed
    assert "grade" not in fixed and "judge_output" not in fixed
    assert fixed["recovery_authorized"]["allow_actor_retry"] is False
    assert fixed["answer_extraction_recovery"]["old_grade"] == old["grade"]
    assert old["quality_score"] == 10.8 and old["memory_ingestion_complete"] is True


def test_unrelated_changed_answer_is_not_a_prefix_loss_repair(tmp_path):
    with pytest.raises(ValueError, match="prefix loss"):
        corrected_record({"answer": "Original"}, {"tools": [{"arguments": {"answer": "Different"}}]},
                         {}, {}, tmp_path, tmp_path / "archive")


def test_migration_discovers_nested_native_comparison_tasks_but_not_recovery_archives(tmp_path):
    paths = [tmp_path / "evolution/round_00/base_logs/1.json",
             tmp_path / "evolution/round_00/eval_batch/parent/1.json",
             tmp_path / "evolution/round_00/eval_final/candidate/3.json"]
    for path in paths:
        write_json(path, {"task_execution_id": "paid"})
    write_json(tmp_path / "evolution/round_00/base_logs/_recovery/1/01-paid.json", {"task_execution_id": "old"})
    assert _records(tmp_path) == sorted(paths)
