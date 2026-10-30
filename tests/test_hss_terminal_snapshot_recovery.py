import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from philoagent.hss_native import _terminal_storage, _tree_manifest


def test_terminal_retry_archives_old_retrieval_writes_and_restarts_frozen_memory(tmp_path):
    workspace = tmp_path / "workspace"
    frozen = tmp_path / "frozen"
    snapshot = tmp_path / "evaluation_snapshots" / "3"
    (workspace / "storage").mkdir(parents=True)
    frozen.mkdir()
    (frozen / "memory.json").write_text('{"usage": 0}', encoding="utf-8")
    (workspace / "storage/memory.json").write_text("training-preserved", encoding="utf-8")
    (snapshot / "storage").mkdir(parents=True)
    (snapshot / "storage/memory.json").write_text('{"usage": 7}', encoding="utf-8")
    before = _tree_manifest(workspace / "storage")
    original = Path.cwd()
    runner = SimpleNamespace(get_memory_config=lambda _: {"storage_path": "storage/memory.json"})

    with _terminal_storage(workspace, frozen, snapshot, runner, resume=True):
        assert Path.cwd() == snapshot
        assert json.loads((snapshot / "storage/memory.json").read_text())["usage"] == 0
        assert runner.get_memory_config("test")["storage_path"] == str(snapshot / "storage/memory.json")
    assert Path.cwd() == original
    assert _tree_manifest(workspace / "storage") == before
    archives = list((snapshot.parent / "_recovery").glob("3-*/storage/memory.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text())["usage"] == 7
    receipts = list((snapshot.parent / "_recovery").glob("3-*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["trained_storage_modified"] is False


def test_existing_snapshot_still_requires_explicit_recovery(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    snapshot = tmp_path / "task"
    snapshot.mkdir()
    runner = SimpleNamespace(get_memory_config=lambda _: {})
    with pytest.raises(FileExistsError):
        with _terminal_storage(tmp_path, frozen, snapshot, runner):
            pass
