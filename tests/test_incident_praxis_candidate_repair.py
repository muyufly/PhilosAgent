"""The paid PraxisLoom recovery runs against temporary copies only."""
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("praxis_candidate_repair", PROJECT / "research/repair_business_fixed_r3_praxis.py")
repair_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair_module)


@pytest.fixture
def incident(tmp_path):
    original = PROJECT / "runs/hss-business-formal-20261029-r01"
    job = Path("jobs") / repair_module.JOB
    paths = [repair_module.PROVIDER, repair_module.GENERATED, repair_module.TEMP_PROVIDER,
             repair_module.ROUND / "validated_systems.json", repair_module.ROUND / "state.json",
             repair_module.ROUND / "generated_system_2.json", repair_module.ROUND / "created_system.json",
             repair_module.ROUND / "checkpoint.json", Path("workspace/EvolveLab/memory_types.py"),
             Path("workspace/EvolveLab/providers/verity_forge_memory_provider.py")]
    for relative in paths:
        target = tmp_path / job / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / job / relative).read_bytes())
    paid = Path("budget/attempts/012991.json")
    (tmp_path / paid).parent.mkdir(parents=True)
    (tmp_path / paid).write_bytes((original / paid).read_bytes())
    # Ensure retry clears only this candidate even when a sibling is validated.
    path = tmp_path / job / repair_module.ROUND / "validated_systems.json"
    validation = json.loads(path.read_text(encoding="utf-8"))
    validation["validated"] = [repair_module.SIBLING]
    validation["details"][repair_module.SIBLING] = {"verdict": "passed", "retained": [1, 2, 3]}
    path.write_text(json.dumps(validation), encoding="utf-8")
    return tmp_path, tmp_path / job


def all_files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_dry_run_changes_no_file_and_validates_real_coldstart(incident):
    root, job = incident
    before = all_files(root)
    result = repair_module.repair(root, expected_provider_sha256=repair_module.sha((job / repair_module.PROVIDER).read_bytes()))
    assert result["coldstart_check"] == {"seed_count": 6, "seed_content_preserved": True,
                                         "next_id": 7, "filesystem_or_model_calls": 0}
    assert not result["applied"] and all_files(root) == before


def test_temp_apply_preserves_paid_raw_sibling_namespace_and_original_backups(incident):
    root, job = incident
    before = all_files(root)
    original_config = json.loads((job / repair_module.GENERATED).read_text(encoding="utf-8"))
    result = repair_module.repair(root, expected_provider_sha256=repair_module.sha((job / repair_module.PROVIDER).read_bytes()), apply=True)
    updated_config = json.loads((job / repair_module.GENERATED).read_text(encoding="utf-8"))
    old_code = original_config["provider_code"].pop("code")
    new_code = updated_config["provider_code"].pop("code")
    assert updated_config == original_config and old_code != new_code
    assert new_code == repair_module.patch_provider(old_code)
    for relative, expected in result["preserved_hashes"].items():
        assert repair_module.sha((job / relative).read_bytes()) == expected
    validation = json.loads((job / repair_module.ROUND / "validated_systems.json").read_text(encoding="utf-8"))
    assert validation["validated"] == [repair_module.SIBLING]
    assert validation["failed"] == []
    assert validation["details"] == {repair_module.SIBLING: {"verdict": "passed", "retained": [1, 2, 3]}}
    for relative in result["input_hashes"]:
        assert (Path(result["backup_dir"]) / "before" / relative).read_bytes() == before[(Path("jobs") / repair_module.JOB / relative).as_posix()]
    assert (Path(result["backup_dir"]) / "receipt.json").is_file()


def test_existing_candidate_learning_record_refuses_before_writing(incident):
    root, job = incident
    path = job / repair_module.ROUND / "eval_batch/praxis_loom_memory/1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"provider": repair_module.CANDIDATE, "status": "completed"}), encoding="utf-8")
    before = all_files(root)
    with pytest.raises(ValueError, match="learning record"):
        repair_module.repair(root, expected_provider_sha256=repair_module.sha((job / repair_module.PROVIDER).read_bytes()), apply=True)
    assert all_files(root) == before
