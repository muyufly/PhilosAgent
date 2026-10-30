"""Pinned instance compatibility and meaningful-state recovery for VerityLoom."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import typing

import pytest

from philoagent.hss_tasks_runtime import HSSResumeRequired, NativeHSSRunner
from test_social_fixed_r2_ingestion_plan import incident

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "runs/hss-business-formal-20261029-r01/native_recovery_20261030/resume_prepare.py"
spec = importlib.util.spec_from_file_location("verity_r19_helper", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def prepared_verity(incident, monkeypatch):
    root, original, events = incident
    source = root / "jobs/social-fixed-r2/workspace/EvolveLab/providers/verity_loom_memory_provider.py"
    record_path = root / "jobs/social-fixed-r2/evolution/round_00/eval_batch/verity_loom_memory/1.json"
    original["runtime_artifact"]["source_path"] = str(source.resolve())
    record_path.write_text(json.dumps(original), encoding="utf8")
    (root / "manifest.json").write_text(json.dumps({"jobs": [{"job_id": "social-fixed-r2"}]}), encoding="utf8")
    monkeypatch.setattr(helper, "process_owned", lambda _: False)
    result = helper.prepare(root)
    assert result["resume_allowed"], result.get("blockers")
    return root, helper.read_json(record_path), source


def isolated_provider(source):
    tree = ast.parse(source.read_text(encoding="utf8"))
    provider = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VerityLoomMemoryProvider")
    methods = [n for n in provider.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_make_item", "_signature", "_apply_outcome")]
    subset = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Isolated", bases=[], keywords=[],
                                                                     body=methods, decorator_list=[])], type_ignores=[]))
    scope = {**vars(typing), "time": time, "hashlib": hashlib, "re": re,
             "VALID_KINDS": {"CONSTRAINT", "PROCEDURE"}, "VALID_TIERS": {"instance", "procedure"},
             "VALID_PROVENANCE": {"task-source", "observed", "unknown"}}
    exec(compile(subset, "isolated-native-verity-methods", "exec"), scope)
    memory = scope["Isolated"]()
    memory._id_counter = 0
    memory.utility_prior, memory.utility_alpha, memory.success_ratio = .5, .32, .7
    return memory


def test_signed_verity_restore_credits_nine_original_ids_and_keeps_disk_untouched(prepared_verity):
    root, record, source = prepared_verity
    assert NativeHSSRunner._verified_ingestion_rollback(record, record["recovery_authorized"])
    memory = isolated_provider(source)
    NativeHSSRunner._apply_provider_compatibility(memory, record, "ingestion")
    record.setdefault("recovery_history", []).append({"prior_task_execution_id": record["task_execution_id"]})
    record["task_execution_id"] = "new-real-ingestion-only"
    NativeHSSRunner._restore_ingestion_context(memory, record)
    assert memory._id_counter == 11 and memory._task_counter == 2 and memory._task["step"] == 7
    assert record["volatile_memory_restoration"]["bitwise_original_state_claimed"] is False
    assert len(record["actor_memory_degradation"]["in_attempts"]) == 5
    assert memory._apply_outcome(memory._task["surfaced_ids"], {"overall_norm": None, "success": True}) == 9
    assert all(item["appearances"] == item["successes"] == 1 for item in memory._store["items"])
    durable = helper.read_json(root / "jobs/social-fixed-r2/workspace/storage/verity_loom_memory/loom_store.json")
    assert len(durable["items"]) == 3 and all(item["appearances"] == 0 for item in durable["items"])


def test_pinned_make_item_compatibility_only_drops_applies_now(prepared_verity):
    _, record, source = prepared_verity
    memory = isolated_provider(source)
    arguments = dict(content="An original candidate", kind="PROCEDURE", polarity="do", binding=False,
                     trigger="when relevant", scope="global", domain="writing", tier="procedure", provenance="observed")
    with pytest.raises(TypeError, match="applies_now"):
        memory._make_item(applies_now=True, **arguments)
    NativeHSSRunner._apply_provider_compatibility(memory, record, "actor")
    item = memory._make_item(applies_now=True, **arguments)
    assert item["content"] == arguments["content"] and "applies_now" not in item
    with pytest.raises(TypeError, match="unrelated_typo"):
        memory._make_item(unrelated_typo=True, **arguments)
    wrong = isolated_provider(source)
    wrong_record = deepcopy(record)
    wrong_record["runtime_artifact"]["code_sha256"] = "unreviewed-source"
    NativeHSSRunner._apply_provider_compatibility(wrong, wrong_record, "actor")
    with pytest.raises(TypeError, match="applies_now"):
        wrong._make_item(applies_now=True, **arguments)


@pytest.mark.parametrize("damage", ["context", "paid_receipt", "grade"])
def test_verity_restoration_rejects_changed_paid_or_meaningful_evidence(prepared_verity, damage):
    root, record, _ = prepared_verity
    if damage == "context":
        record["native_ingestion_recovery"]["runtime_context"]["store"]["items"][0]["appearances"] = 5
    elif damage == "paid_receipt":
        paid = root / "budget/attempts/015359.json"
        paid.write_text(paid.read_text(encoding="utf8") + " ", encoding="utf8")
    else:
        record["grade"]["quality_score"] = 99.9
    with pytest.raises(HSSResumeRequired, match="verified completed storage rollback"):
        NativeHSSRunner._verified_ingestion_rollback(record, record["recovery_authorized"])
