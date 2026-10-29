"""Offline checks for held-out prompts and independent post-training replays."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from philoagent import decision_smoke
from philoagent.decision_fixture import Episode, PROBE_IDS, TRAIN_IDS, _OUTCOMES


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_run(root):
    run = Path(root) / "run"
    _write(run / "run.json", json.dumps({"provider": "old_provider"}))
    _write(run / "workspace/agent.py", "# post-evolution shared code\n")
    _write(run / "workspace/storage/old_provider/private.txt", "not frozen")
    _write(run / "workspace/storage/new_provider/private.txt", "candidate private state")
    _write(run / "workspace/.env", "do not copy")
    _write(run / "workspace/__pycache__/cache.pyc", "do not copy")
    _write(run / "lineage/old_provider.json", '{"items": ["old philosophy"]}')
    _write(run / "lineage/new_provider.json", '{"items": ["new philosophy"]}')
    frozen = run / "decision_smoke/frozen_memory"
    _write(frozen / "memory.json", '{"training": ["observed result"]}')
    _write(frozen / "nested/index.json", '{"index": 1}')
    files = decision_smoke.file_manifest(frozen)
    _write(run / "decision_smoke/frozen_memory.json",
           json.dumps({"files": files, "sha256": decision_smoke.digest(files)}))
    return run


class DecisionSmokeIsolationTests(unittest.TestCase):
    def test_question_contains_only_public_task_and_execution_instructions(self):
        for task_id in TRAIN_IDS + PROBE_IDS:
            with self.subTest(task_id=task_id):
                question = decision_smoke.make_question(task_id)
                public = json.loads(question[question.index("{"):])
                self.assertEqual(public, Episode(task_id).public_task())
                self.assertIn("act", question)
                self.assertIn("final_answer", question)
                self.assertNotIn("_OUTCOMES", question)
                self.assertNotIn("utility", question)
                hidden = _OUTCOMES[task_id]
                for observation in hidden["sources"].values():
                    self.assertNotIn(observation, question)
                for _, outcome in hidden["actions"].values():
                    self.assertNotIn(outcome, question)

    def test_clones_share_frozen_memory_but_have_independent_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(temporary)
            old = decision_smoke.clone_probe(run, PROBE_IDS[0], "old_provider", "old_01")
            new = decision_smoke.clone_probe(run, PROBE_IDS[0], "new_provider", "new_01")
            old_memory = old / "workspace/storage/old_provider"
            new_memory = new / "workspace/storage/new_provider"
            frozen = run / "decision_smoke/frozen_memory"
            original = decision_smoke.file_manifest(frozen)
            self.assertEqual(decision_smoke.file_manifest(old_memory), original)
            self.assertEqual(decision_smoke.file_manifest(new_memory), original)
            _write(old_memory / "memory.json", '{"modified": true}')
            self.assertEqual(decision_smoke.file_manifest(new_memory), original)
            self.assertEqual(decision_smoke.file_manifest(frozen), original)
            self.assertNotEqual(decision_smoke.file_manifest(old_memory), original)
            for trial, provider in ((old, "old_provider"), (new, "new_provider")):
                self.assertEqual({p.name for p in (trial / "workspace/storage").iterdir()}, {provider})
                metadata = json.loads((trial / "probe.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["initial_memory_files"], original)
                self.assertEqual(metadata["common_memory_sha256"], decision_smoke.digest(original))

    def test_clones_copy_same_code_and_lineage_without_transient_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(temporary)
            trial = decision_smoke.clone_probe(run, PROBE_IDS[1], "new_provider", "new_02")
            self.assertEqual((trial / "workspace/agent.py").read_bytes(),
                             (run / "workspace/agent.py").read_bytes())
            self.assertEqual((trial / "run.json").read_bytes(), (run / "run.json").read_bytes())
            self.assertEqual(decision_smoke.file_manifest(trial / "lineage"),
                             decision_smoke.file_manifest(run / "lineage"))
            self.assertFalse((trial / "workspace/.env").exists())
            self.assertFalse((trial / "workspace/__pycache__").exists())
            _write(trial / "workspace/agent.py", "# local mutation\n")
            self.assertNotEqual((trial / "workspace/agent.py").read_bytes(),
                                (run / "workspace/agent.py").read_bytes())

    def test_train_and_unknown_task_ids_cannot_be_cloned_as_probes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(temporary)
            for task_id in TRAIN_IDS + ("unknown",):
                with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                    decision_smoke.clone_probe(run, task_id, "old_provider", "invalid")
            self.assertFalse((run / "decision_smoke/trials").exists())

    def test_duplicate_labels_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(temporary)
            trial = decision_smoke.clone_probe(run, PROBE_IDS[0], "old_provider", "once")
            before = decision_smoke.file_manifest(trial)
            with self.assertRaises(FileExistsError):
                decision_smoke.clone_probe(run, PROBE_IDS[1], "new_provider", "once")
            self.assertEqual(decision_smoke.file_manifest(trial), before)

    def test_changed_frozen_memory_fails_before_creating_a_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(temporary)
            _write(run / "decision_smoke/frozen_memory/memory.json", "changed")
            with self.assertRaisesRegex(ValueError, "memory changed"):
                decision_smoke.clone_probe(run, PROBE_IDS[0], "old_provider", "changed")
            self.assertFalse((run / "decision_smoke/trials").exists())

    def test_storage_and_trial_paths_cannot_escape_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(temporary)
            for provider, label in (("../escape", "valid"), ("valid", "../escape"),
                                    ("", "valid"), ("valid", "")):
                with self.subTest(provider=provider, label=label), self.assertRaises(ValueError):
                    decision_smoke.clone_probe(run, PROBE_IDS[0], provider, label)
            self.assertFalse((run / "decision_smoke/trials").exists())

    def test_probe_ingestion_rejected_before_native_import_or_paid_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(sys.modules, {"FlashOAgents": None}), self.assertRaises(ValueError):
                decision_smoke.run_episodes(root, {}, root / "budget", PROBE_IDS,
                                           "old_provider", root / "result", ingest=True)
            self.assertFalse((root / "result").exists())


if __name__ == "__main__":
    unittest.main()
