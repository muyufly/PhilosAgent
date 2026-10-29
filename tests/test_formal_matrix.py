"""Design-integrity tests. These never load credentials or contact a model."""

from copy import deepcopy
import json
import os
import socket
from unittest.mock import patch

import pytest

from philoagent.formal_matrix import (
    GROUPS, budget_estimate, build_bundle, build_schedule, canonical, main,
    protocol, schedule_counts, sha, verify_bundle, write_bundle,
)


SAMPLE_STREAMS = {
    f"flow_{index}": {"train_ids": [f"f{index}_learn_{i}" for i in range(4)],
                      "probe_ids": [f"f{index}_holdout_{i}" for i in range(2)]}
    for index in range(3)
}


def test_same_learning_task_order_and_no_probe_feedback_or_m0_learning():
    schedule = build_schedule(SAMPLE_STREAMS)
    for stream_id, stream in SAMPLE_STREAMS.items():
        rows = [row for row in schedule if row["stream"] == stream_id]
        for group in GROUPS:
            primary = [row for row in rows if row["group"] == group and row["phase"] == "primary_learning"]
            assert [row["task_id"] for row in primary] == stream["train_ids"]
            assert all(row["ingest"] == (group != "M0") for row in primary)
        for row in rows:
            if row["task_id"] in stream["probe_ids"]:
                assert row["ingest"] is False
                assert row["phase"] in {"primary_holdout", "same_phi_baseline", "paired_probe"}
            if row["phase"] in {"candidate_comparison", "finalist_comparison"}:
                assert row["task_id"] in stream["train_ids"]
                assert row["ingest"] is False
                assert row["memory"] == "fresh_final_primary_M_copy"
        last_learning = max(i for i, row in enumerate(rows) if row["phase"] == "primary_learning")
        first_reflection = min(i for i, row in enumerate(rows) if row["phase"] == "system_reflection")
        first_holdout = min(i for i, row in enumerate(rows) if row["phase"] == "primary_holdout")
        last_selection = max(i for i, row in enumerate(rows) if row["phase"] == "native_selection")
        assert last_learning < first_reflection < last_selection < first_holdout


def test_static_and_revisable_receive_identical_shadow_computation():
    rows = build_schedule(SAMPLE_STREAMS)
    for stream in SAMPLE_STREAMS:
        extras = []
        for group in ("S0", "S1"):
            selected = [row for row in rows if row["stream"] == stream and row["group"] == group
                        and row["phase"] in {"system_reflection", "native_interface_validation",
                                             "candidate_comparison", "finalist_comparison"}]
            extras.append([{key: value for key, value in row.items() if key not in {"event_id", "group"}}
                           for row in selected])
        assert extras[0] == extras[1]
    config = protocol()["system_reflection"]
    assert config["S0_actor_receives_shadow_candidate_or_analysis"] is False
    assert config["extra_experience_returned_to_primary_M"] is False


def test_conditional_probes_have_fresh_same_memory_and_balanced_order():
    rows = build_schedule(SAMPLE_STREAMS)
    for stream_id, stream in SAMPLE_STREAMS.items():
        for index, task in enumerate(stream["probe_ids"]):
            paired = [row for row in rows if row["stream"] == stream_id
                      and row["task_id"] == task and row["phase"] == "paired_probe"]
            assert [row["phi"] for row in paired] == (
                ["initial", "selected", "selected", "initial"] if index == 0
                else ["selected", "initial", "initial", "selected"])
            assert [row["repetition"] for row in paired] == [0, 0, 1, 1]
            assert all(row["memory"] == "fresh_final_primary_M_copy" and not row["ingest"] for row in paired)
            baseline = [row for row in rows if row["stream"] == stream_id
                        and row["task_id"] == task and row["phase"] == "same_phi_baseline"]
            assert len(baseline) == 2
            assert all(row["phi"] == "initial" for row in baseline)


def test_cost_includes_discovery_selection_baseline_and_interface_validation():
    full = build_schedule(SAMPLE_STREAMS)
    counts = schedule_counts(full)
    assert counts["actor_task_executions"] == 192
    assert counts["by_phase"]["primary_learning"] == 48
    assert counts["by_phase"]["primary_holdout"] == 24
    assert counts["by_phase"]["candidate_comparison"] == 48
    assert counts["by_phase"]["finalist_comparison"] == 36
    assert counts["by_phase"]["same_phi_baseline"] == 12
    assert counts["by_phase"]["paired_probe"] == 24
    assert counts["by_phase"]["native_interface_validation"] == 6
    assert counts["system_lineages"] == 6
    budget = budget_estimate(full)
    assert budget["planning_envelope"] == {
        "requests": 2376, "input_tokens": 9216000,
        "output_tokens": 1447728, "total_tokens": 10663728}
    assert budget["approved_limits"] is None
    pilot = build_schedule(dict(list(SAMPLE_STREAMS.items())[:1]))
    assert schedule_counts(pilot)["actor_task_executions"] == 64
    assert budget_estimate(pilot)["planning_envelope"]["total_tokens"] == 3554576


@pytest.mark.parametrize("change", [lambda s: s["flow_0"]["train_ids"].pop(),
                                   lambda s: s["flow_0"]["probe_ids"].append("extra"),
                                   lambda s: s["flow_0"]["probe_ids"].__setitem__(0, "f0_learn_0"),
                                   lambda s: s["flow_0"]["train_ids"].__setitem__(0, "f0_learn_1"),
                                   lambda s: s["flow_0"]["probe_ids"].__setitem__(0, "f0_holdout_1"),
                                   lambda s: s["flow_1"]["train_ids"].__setitem__(0, "f0_learn_0"),
                                   lambda s: s["flow_0"].__setitem__("train_ids", "abcd"),
                                   lambda s: s["flow_0"]["train_ids"].__setitem__(0, "../escape")])
def test_wrong_split_shape_is_rejected(change):
    streams = deepcopy(SAMPLE_STREAMS)
    change(streams)
    with pytest.raises(ValueError):
        build_schedule(streams)


def test_review_creation_and_verification_cannot_load_environment_or_network(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline review tried to load credentials or contact a service")

    with patch("philoagent.cli.environment", forbidden), patch.object(socket.socket, "connect", forbidden):
        manifest = write_bundle(tmp_path / "pilot")
        assert verify_bundle(tmp_path / "pilot")["verified"]
        assert manifest["formal_matrix_authorized"] is False
        assert manifest["live_executor_ready"] is False
        assert manifest["budget"]["approved_limits"] is None
        assert manifest["live_model_calls_performed"] is False


def test_learning_files_never_include_holdout_or_evaluator_spec(tmp_path):
    manifest, artifacts = build_bundle(scope="full")
    for stream_id, stream in manifest["streams"].items():
        raw = artifacts[f"learning/{stream_id}.json"].decode("utf-8")
        learning = json.loads(raw)
        assert [row["task_id"] for row in learning] == stream["train_ids"]
        assert all(task_id not in raw for task_id in stream["probe_ids"])
        assert "evaluator_only" not in raw
        assert "_hidden" not in raw
        forbidden_keys = {"evaluator_only", "initial_state", "actions", "requirements", "design_note",
                          "when", "updates", "reward", "utility", "violations"}

        def check_public(value):
            if isinstance(value, dict):
                assert not forbidden_keys.intersection(value)
                for child in value.values():
                    check_public(child)
            elif isinstance(value, list):
                for child in value:
                    check_public(child)

        check_public(learning)
        evaluated = json.loads(artifacts[f"evaluation_only/{stream_id}.json"])
        assert set(evaluated) == set(stream["train_ids"] + stream["probe_ids"])
    phi = json.loads(artifacts["seeds/phi0.json"])["items"]
    initial_m = json.loads(artifacts["seeds/initial_memory.json"])
    assert len(phi) == 6
    assert not any("METHOD-" in item for item in phi)
    memory_text = json.dumps(initial_m, ensure_ascii=False)
    assert all(f"METHOD-{i}" in memory_text for i in range(1, 5))
    assert not any(item in memory_text for item in phi)


def test_changed_artifact_and_self_signed_approval_cannot_pass(tmp_path):
    root = tmp_path / "review"
    write_bundle(root)
    path = root / "manifest.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["formal_matrix_authorized"] = True
    saved["proposal_sha256"] = sha({key: value for key, value in saved.items() if key != "proposal_sha256"})
    path.write_bytes(canonical(saved))
    with pytest.raises(ValueError, match="pending approval state"):
        verify_bundle(root)


def test_conflicting_review_refuses_partial_overwrite_and_extra_files(tmp_path):
    root = tmp_path / "review"
    write_bundle(root)
    protocol_path = root / "protocol.json"
    original = protocol_path.read_bytes()
    protocol_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Refusing changed review artifact"):
        write_bundle(root)
    assert protocol_path.read_bytes() == b"changed"
    protocol_path.write_bytes(original)
    (root / "untracked_result.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(root)


def test_identical_review_retry_is_byte_stable(tmp_path):
    root = tmp_path / "review"
    write_bundle(root)
    before = {path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in root.rglob("*") if path.is_file()}
    write_bundle(root)
    after = {path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
             for path in root.rglob("*") if path.is_file()}
    assert before == after


def test_linked_bundle_root_and_artifact_parent_are_rejected(tmp_path):
    def directory_link(link, target):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            import _winapi
            _winapi.CreateJunction(str(target), str(link))

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked"
    directory_link(linked_root, outside)
    with pytest.raises(ValueError, match="links"):
        write_bundle(linked_root)
    with pytest.raises(ValueError, match="links"):
        verify_bundle(linked_root)
    root = tmp_path / "review"
    root.mkdir()
    directory_link(root / "learning", outside)
    with pytest.raises(ValueError, match="links"):
        write_bundle(root)
    assert not list(outside.iterdir())


def test_cli_has_only_offline_prepare_and_verify(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["run", "anything"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
