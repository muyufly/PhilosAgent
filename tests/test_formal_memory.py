"""Real native JSON compatibility and episode isolation, without network calls."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from philoagent import formal_memory as memory


def test_initial_memory_is_deterministic_and_keeps_methods_out_of_phi(tmp_path):
    db = memory.build_initial_memory()
    assert db == memory.build_initial_memory()
    records = db["strategic"] + db["operational"]
    assert len(records) == 11
    assert sum(item["source_kind"] == "native_cold_start_prior" for item in records) == 7
    assert [item["source_id"] for item in records if item["source_kind"] == "shared_method_resource"] == [
        "METHOD-1", "METHOD-2", "METHOD-3", "METHOD-4"]
    assert all(item["usage_count"] == item["success_count"] == 0 for item in records)
    phi = memory.phi_seed()
    assert len(phi) == 6
    assert not any("METHOD-" in item for item in phi)
    assert not any(item in record["content"] for item in phi for record in records)
    first = memory.write_initial_memory(tmp_path / "first", boundary_root=tmp_path)
    second = memory.write_initial_memory(tmp_path / "second", boundary_root=tmp_path)
    assert first == second
    assert json.loads((tmp_path / "first/longterm_memory.json").read_text(encoding="utf-8")) == db


def test_real_native_provider_loads_without_model_then_retrieves_method_candidates(tmp_path):
    # Import the actual lightweight provider. No heavy framework/model is loaded.
    from EvolveLab.memory_types import MemoryRequest, MemoryStatus
    from EvolveLab.providers.lightweight_memory_provider import LightweightMemoryProvider

    root = tmp_path / "ordinary"
    frozen = memory.write_initial_memory(root, boundary_root=tmp_path)
    provider = LightweightMemoryProvider({"storage_dir": str(root), "enable_longterm_provision": True})
    assert provider.model is None
    assert provider.initialize() is True
    assert provider.longterm_db == memory.build_initial_memory()
    assert memory.snapshot_manifest(root) == frozen
    assert all(record["signature"] == provider._compute_signature(record["content"].lower())
               for record in provider.longterm_db["strategic"] + provider.longterm_db["operational"])

    requests = []
    methods = [record["content"] for record in provider.longterm_db["strategic"]
               if record["source_kind"] == "shared_method_resource"]

    def offline_model(messages):
        requests.append(messages)
        return SimpleNamespace(content=json.dumps({"selected_indices": [6, 7, 8, 9],
                                                   "guidance": "Consider the four available shared methods."}))

    provider.model = offline_model
    response = provider.provide_memory(MemoryRequest(query="A new local decision", context="",
                                                     status=MemoryStatus.BEGIN))
    assert len(requests) == 1
    prompt = requests[0][0]["content"][0]["text"]
    assert all(method in prompt for method in methods)
    assert response.total_count == 1
    assert "four available shared methods" in response.memories[0].content
    assert [item["usage_count"] for item in provider.longterm_db["strategic"][5:]] == [1, 1, 1, 1]


def test_every_episode_gets_full_pristine_directory_copy(tmp_path):
    source = tmp_path / "frozen"
    memory.write_initial_memory(source, boundary_root=tmp_path)
    (source / "private/cache").mkdir(parents=True)
    (source / "private/cache/index.json").write_text('{"native_cache":1}', encoding="utf-8")
    (source / "empty_native_directory").mkdir()
    expected = memory.snapshot_manifest(source)
    trial1, trial2 = tmp_path / "trials/one", tmp_path / "trials/two"
    assert memory.clone_memory(source, trial1, expected_manifest=expected, boundary_root=tmp_path) == expected
    (trial1 / "private/cache/index.json").write_text('{"changed_by_retrieval":1}', encoding="utf-8")
    (trial1 / "longterm_memory.json").write_text('{"changed_by_ingestion":1}', encoding="utf-8")
    assert memory.clone_memory(source, trial2, expected_manifest=expected, boundary_root=tmp_path) == expected
    assert memory.snapshot_manifest(source) == memory.snapshot_manifest(trial2) == expected
    assert (trial2 / "empty_native_directory").is_dir()


def test_clone_refuses_source_changes_existing_targets_and_empty_snapshots(tmp_path):
    source = tmp_path / "source"
    expected = memory.write_initial_memory(source, boundary_root=tmp_path)
    target = tmp_path / "trial"
    memory.clone_memory(source, target, expected_manifest=expected, boundary_root=tmp_path)
    with pytest.raises(FileExistsError):
        memory.clone_memory(source, target, expected_manifest=expected, boundary_root=tmp_path)
    with pytest.raises(FileExistsError):
        memory.write_initial_memory(source, boundary_root=tmp_path)
    (source / "private.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen manifest"):
        memory.clone_memory(source, tmp_path / "new", expected_manifest=expected, boundary_root=tmp_path)
    assert not (tmp_path / "new").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="Empty"):
        memory.snapshot_manifest(empty)


def test_paths_cannot_escape_boundary_or_overlap_source(tmp_path):
    boundary = tmp_path / "boundary"
    boundary.mkdir()
    source = boundary / "source"
    expected = memory.write_initial_memory(source, boundary_root=boundary)
    for target in (tmp_path / "outside", boundary / ".." / "escape", source / "nested"):
        with pytest.raises(ValueError):
            memory.clone_memory(source, target, expected_manifest=expected, boundary_root=boundary)
    with pytest.raises(ValueError):
        memory.write_initial_memory(boundary, boundary_root=boundary)
    with pytest.raises(ValueError):
        memory.clone_memory(source / ".." / "source", boundary / "trial",
                            expected_manifest=expected, boundary_root=boundary)


def test_memory_tree_links_and_linked_parents_are_rejected(tmp_path):
    source = tmp_path / "source"
    expected = memory.write_initial_memory(source, boundary_root=tmp_path)
    linked = tmp_path / "linked"
    def directory_link(link, target):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            # Junctions require no Windows developer-mode symlink permission.
            import _winapi
            _winapi.CreateJunction(str(target), str(link))

    directory_link(linked, source)
    with pytest.raises(ValueError, match="links"):
        memory.snapshot_manifest(linked)
    with pytest.raises(ValueError, match="links"):
        memory.clone_memory(source, linked / "trial", expected_manifest=expected, boundary_root=tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    directory_link(source / "escaped", elsewhere)
    with pytest.raises(ValueError, match="link"):
        memory.snapshot_manifest(source)


def test_combined_phi_and_invalid_resource_schema_are_rejected(tmp_path):
    combined = tmp_path / "combined.json"
    combined.write_text(json.dumps({"schema_version": 1, "items": [*memory.phi_seed(), "METHOD-1"]}),
                        encoding="utf-8")
    with pytest.raises(ValueError):
        memory.phi_seed(combined)
    methods = tmp_path / "methods.json"
    methods.write_text(json.dumps({"schema_version": 1, "items": ["wrong"] * 4}), encoding="utf-8")
    with pytest.raises(ValueError):
        memory.build_initial_memory(methods)
    methods.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        memory.build_initial_memory(methods)


def test_change_during_copy_is_detected_without_deleting_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source"
    expected = memory.write_initial_memory(source, boundary_root=tmp_path)
    original_copy = memory.shutil.copytree

    def changing_copy(src, dst, *args, **kwargs):
        result = original_copy(src, dst, *args, **kwargs)
        (source / "changed.json").write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(memory.shutil, "copytree", changing_copy)
    target = tmp_path / "trial"
    with pytest.raises(ValueError, match="changed while copying"):
        memory.clone_memory(source, target, expected_manifest=expected, boundary_root=tmp_path)
    assert target.is_dir()
    assert (source / "changed.json").exists()
