"""Offline checks for lineage isolation, resume behavior and task immutability."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from philoagent.orientation import (
    OrientationConflictError,
    OrientationIntegrityError,
    OrientationValidationError,
    assert_unchanged,
    bootstrap,
    env_context,
    fork,
    fork_from_snapshot,
    load,
    prompt,
)


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run" / "lineage"

    def test_bootstrap_is_once_and_resume_cannot_replace_seed(self):
        path = bootstrap(self.root, "base", ["initial"])
        before = path.read_bytes()
        self.assertEqual(bootstrap(self.root, "base", ["initial"]), path)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(OrientationConflictError):
            bootstrap(self.root, "base", ["different"])
        self.assertEqual(load(self.root, "base").items, ("initial",))

    def test_fork_inherits_current_parent_not_original_seed(self):
        bootstrap(self.root, "base", ["initial"])
        fork(self.root, "base", "revised", ["new method"])
        fork(self.root, "revised", "descendant")
        parent = load(self.root, "revised")
        child = load(self.root, "descendant")
        self.assertEqual(child.items, ("new method",))
        self.assertEqual(child.parent_provider_id, parent.provider_id)
        self.assertEqual(child.parent_sha256, parent.sha256)
        self.assertEqual(load(self.root, "base").items, ("initial",))

    def test_empty_replacement_is_preserved_across_resume_and_fork(self):
        bootstrap(self.root, "base", ["initial"])
        fork(self.root, "base", "empty", [])
        fork(self.root, "base", "empty", [])
        fork(self.root, "empty", "still_empty")
        self.assertEqual(load(self.root, "still_empty").items, ())
        self.assertEqual(prompt(self.root, "empty"), "")
        with self.assertRaises(OrientationConflictError):
            fork(self.root, "base", "empty")

    def test_null_is_invalid_not_an_alias_for_inherit_or_delete(self):
        bootstrap(self.root, "base", ["initial"])
        with self.assertRaises(OrientationValidationError):
            fork(self.root, "base", "child", None)
        self.assertFalse((self.root / "child" / "orientation.json").exists())

    def test_siblings_and_runs_have_independent_namespaces(self):
        bootstrap(self.root, "base", ["initial"])
        fork(self.root, "base", "left", ["left method"])
        fork(self.root, "base", "right", ["right method"])
        second_run = Path(self.temporary.name) / "second" / "lineage"
        bootstrap(second_run, "base", ["another run"])
        self.assertEqual(load(self.root, "left").items, ("left method",))
        self.assertEqual(load(self.root, "right").items, ("right method",))
        self.assertEqual(load(self.root, "base").items, ("initial",))
        self.assertEqual(load(second_run, "base").items, ("another run",))

    def test_saved_parent_snapshot_must_match_active_lineage(self):
        bootstrap(self.root, "base", ["initial"])
        saved = load(self.root, "base").to_dict()
        fork_from_snapshot(self.root, saved, "child")
        saved["items"] = ["stale"]
        with self.assertRaises(OrientationConflictError):
            fork_from_snapshot(self.root, saved, "other_child")
        self.assertFalse((self.root / "other_child").exists())

    def test_existing_child_cannot_be_reparented(self):
        bootstrap(self.root, "a", ["same content"])
        bootstrap(self.root, "b", ["same content"])
        fork(self.root, "a", "child")
        with self.assertRaises(OrientationConflictError):
            fork(self.root, "b", "child")

    def test_integrity_detects_content_tampering_and_task_deletion(self):
        path = bootstrap(self.root, "base", ["initial"])
        snapshot = load(self.root, "base")
        assert_unchanged(snapshot)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["items"] = ["silently mutated"]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(OrientationIntegrityError):
            load(self.root, "base")
        with self.assertRaises(OrientationIntegrityError):
            assert_unchanged(snapshot)
        path.unlink()
        with self.assertRaises(OrientationIntegrityError):
            assert_unchanged(snapshot)

    def test_task_guard_detects_replacement_with_valid_new_snapshot(self):
        path = bootstrap(self.root, "base", ["initial"])
        snapshot = load(self.root, "base")
        path.unlink()
        bootstrap(self.root, "base", ["replacement with valid hash"])
        with self.assertRaises(OrientationIntegrityError):
            assert_unchanged(snapshot)

    def test_optional_environment_never_seeds_missing_or_empty_states(self):
        self.assertIsNone(env_context("base", environ={}))
        with self.assertRaises(FileNotFoundError):
            env_context("base", environ={"PHILO_LINEAGE_ROOT": str(self.root)})
        with self.assertRaises(OrientationValidationError):
            env_context("base", environ={"PHILO_LINEAGE_ROOT": ""})
        bootstrap(self.root, "base", [])
        self.assertEqual(
            env_context("base", environ={"PHILO_LINEAGE_ROOT": str(self.root)}).items, (),
        )

    def test_unsafe_provider_ids_and_same_namespace_are_rejected(self):
        for name in ("../escape", "a/b", "a\\b", "", ".", "NUL", "COM1", "bad name"):
            with self.subTest(name=name), self.assertRaises(OrientationValidationError):
                bootstrap(self.root, name, [])
        bootstrap(self.root, "base", [])
        for child in ("base", "BASE"):
            with self.assertRaises(OrientationValidationError):
                fork(self.root, "base", child)
        with self.assertRaises(OrientationValidationError):
            fork(self.root, None, "child")

    def test_invalid_items_do_not_create_state(self):
        for items in (None, "a string", [1], ["  "], [False]):
            with self.subTest(items=items), self.assertRaises(OrientationValidationError):
                bootstrap(self.root, "base", items)
        self.assertFalse(self.root.exists())

    def test_atomic_creation_handles_identical_and_conflicting_writers(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: bootstrap(self.root, "same", ["method"]), range(8)))
        self.assertEqual(len(set(paths)), 1)

        def create(value):
            try:
                bootstrap(self.root, "competing", [value])
                return value
            except OrientationConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(create, ("first", "second")))
        winner = [value for value in results if value is not None]
        self.assertEqual(len(winner), 1)
        self.assertEqual(load(self.root, "competing").items, tuple(winner))
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_seed_files_have_same_schema_and_four_nonempty_methods(self):
        seeds = Path(__file__).resolve().parents[1] / "seeds"
        lengths = []
        for name in ("engineering", "dialectical", "critical"):
            data = json.loads((seeds / f"{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"schema_version", "items"})
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(len(data["items"]), 4)
            self.assertTrue(all(isinstance(item, str) and item.strip() for item in data["items"]))
            lengths.append(sum(map(len, data["items"])))
        self.assertLess(max(lengths) / min(lengths), 1.25)


if __name__ == "__main__":
    unittest.main()
