"""Offline temp-only checks of the single paid grade incident repair."""
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paid_grade_encoding", PROJECT / "research/repair_humanities_full_r2_paid_grade.py")
repair_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair_module)


@pytest.fixture
def incident(tmp_path):
    original_root = PROJECT / "runs/hss-business-formal-20261029-r01"
    paid = json.loads((original_root / "budget/attempts/010232.json").read_text(encoding="utf-8"))
    raw = paid["response"]["choices"][0]["message"]["content"]
    record = {"task_id": "WB0804", "domain": "humanities", "status": "judge_failed",
              "judge_output": raw, "judge_recovery_attempts": 1, "answer": "Preserved paid answer",
              "memory_ingestion_complete": False, "task_execution_id": "fixture"}
    path = tmp_path / repair_module.RELATIVE_RECORD
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    receipt = tmp_path / "budget/attempts/010232.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(paid), encoding="utf-8")
    return tmp_path, path


def test_dry_run_preserves_every_file_and_validates_original_score(incident):
    root, path = incident
    before = path.read_bytes()
    result = repair_module.repair(root, expected_record_sha256=repair_module.sha(before))
    assert result["quality_score_validated"] == 81.8 and not result["applied"]
    assert path.read_bytes() == before
    assert not (root / "native_recovery_20261030").exists()


def test_apply_only_changes_encoding_and_audit_then_native_parser_can_reuse_it(incident):
    root, path = incident
    before = path.read_bytes()
    old = json.loads(before)
    result = repair_module.repair(root, expected_record_sha256=repair_module.sha(before), apply=True)
    new = json.loads(path.read_bytes())
    parsed = json.loads(new["judge_output"])
    assert parsed["dimensions"]["communication"]["reason"].count('"救救孩子"') == 1
    assert new["judge_output_original"] == old["judge_output"]
    assert new["status"] == "judge_failed" and not new["memory_ingestion_complete"]
    assert "grade" not in new
    assert {k: v for k, v in new.items() if k not in {"judge_output", "judge_output_original", "judge_encoding_repair"}} == {k: v for k, v in old.items() if k != "judge_output"}
    assert (Path(result["backup_dir"]) / "record.before.json").read_bytes() == before
    from philoagent.hss_tasks_runtime import NativeHSSRunner
    callback = object.__new__(NativeHSSRunner)
    callback.protocol = {"resume_incomplete_tasks": True, "judge_recovery_max_attempts": 1}
    callback.data_first = False
    # Add fixture-only identity/accounting/recovery metadata to reach the native reuse seam.
    new.update(task_identity="identity", api_usage={"usage_complete": True},
               recovery_authorized={"revision": "repair-r13"}, agent_messages=[])
    path.write_text(json.dumps(new), encoding="utf-8")
    assert callback._saved_task(path, "identity")["judge_output"] == new["judge_output"]


def test_changed_record_sha_is_rejected_before_writing(incident):
    root, path = incident
    before = path.read_bytes()
    with pytest.raises(ValueError, match="SHA256"):
        repair_module.repair(root, expected_record_sha256="0" * 64, apply=True)
    assert path.read_bytes() == before
