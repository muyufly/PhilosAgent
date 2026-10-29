"""No network: budget freezing, successful phase reuse and provenance evidence."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from philoagent import decision_smoke as smoke


def test_completed_phase_is_read_without_model_or_environment(tmp_path):
    root = tmp_path / "run/decision_smoke"
    root.mkdir(parents=True)
    manifest = {"max_calls": 150, "max_tokens_soft_stop": 800000,
                "adapter_sha256": "frozen", "formal_matrix_authorized": False}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "evolve").mkdir()
    (root / "evolve/summary.json").write_text('{"success": true}')
    with patch.object(smoke, "source_hash", return_value="frozen"), patch.object(
        smoke, "environment", side_effect=AssertionError("Completed phase must not load credentials")):
        result = smoke.execute(root.parent, "evolve")
    assert result["reused_completed_phase"] == "evolve"
    assert result["network_dispatched"] is False
    assert not (root / "evolve.native.log").exists()


def test_training_freeze_can_finish_partial_copy_without_replaying_actor(tmp_path):
    source = tmp_path / "workspace/storage/lightweight_memory"
    source.mkdir(parents=True)
    (source / "memory.json").write_text('{"observed":1}')
    (source / "index.json").write_text('[1]')
    frozen = tmp_path / "decision_smoke/frozen_memory"
    frozen.mkdir(parents=True)
    (frozen / "memory.json").write_bytes((source / "memory.json").read_bytes())
    summary = {"memory_after": smoke.file_manifest(source)}
    smoke.freeze_training_memory(tmp_path, summary)
    smoke.freeze_training_memory(tmp_path, summary)
    assert smoke.file_manifest(frozen) == summary["memory_after"]
    assert (tmp_path / "decision_smoke/frozen_memory.json").is_file()


def test_run_limits_are_saved_per_run_not_current_defaults():
    assert smoke.run_limits({"max_calls": 100, "max_tokens_soft_stop": 300000}) == {
        "max_calls": 100, "max_tokens": 300000}
    assert smoke.run_limits({"max_calls": 150, "max_tokens_soft_stop": 800000}) == smoke.NEW_LIMITS


def test_guidance_provenance_does_not_claim_synthesis_is_an_experience(tmp_path):
    directory = tmp_path / "attempts"
    directory.mkdir()
    data = {"request": {"messages": [{"role": "system", "content": "You are a memory guidance system."},
                                     {"role": "user", "content": "Task and seven default priors"}]},
            "response": {"choices": [{"message": {"content": '{"selected_indices":[1,2],"guidance":"Try an action"}'}}]}}
    (directory / "0001.json").write_text(json.dumps(data))
    result = smoke.guidance_provenance(tmp_path, 1, 1, SimpleNamespace(items=["A philosophical claim"]),
                                       "native_cold_start_priors")
    assert result[0]["selected_indices"] == [1, 2]
    assert result[0]["source_kind"] == "retrieval_synthesis"
    assert result[0]["phi_visible_to_synthesis"] is False
    assert result[0]["memory_origin"] == "native_cold_start_priors"
    assert json.loads((directory / "0001.json").read_text()) == data
