"""Offline checks for a locked, paired factorial matrix and evaluator isolation."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

from philoagent.ablation_design import SEEDS, build_design, write_design
from philoagent.data import read_jsonl


class AblationDesignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        (self.project / "seeds").mkdir(parents=True)
        for seed in SEEDS[:-1]:
            (self.project / "seeds" / f"{seed}.json").write_text(
                json.dumps({"schema_version": 1, "items": [f"{seed} principle"]}), encoding="utf-8")
        self.prepared = self.project / "data" / "webwalkerqa" / "prepared"
        self.prepared.mkdir(parents=True)
        self.dev = self.rows("dev", 12)
        self.heldout = self.rows("heldout", 10)
        self.save("dev", self.dev)
        self.save("heldout", self.heldout)

    @staticmethod
    def rows(split, count):
        return [{"task_id": f"{split}-{index}", "question": f"{split} question {index}",
                 "answer": f"{split} private answer {index}",
                 "root_url": f"https://{split}{index}.example/"}
                for index in range(count)]

    def save(self, split, rows):
        (self.prepared / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_complete_matrix_and_paired_orders(self):
        design = build_design(self.project)
        self.assertEqual(design["job_count"], 48)
        actual = {(job["seed"], job["code_evolve"], job["orientation_evolve"], job["replicate"])
                  for job in design["jobs"]}
        expected = set(itertools.product(SEEDS, (False, True), (False, True), range(3)))
        self.assertEqual(actual, expected)
        self.assertEqual(len({job["job_id"] for job in design["jobs"]}), 48)
        for repetition in design["repetitions"]:
            jobs = [job for job in design["jobs"] if job["replicate"] == repetition["replicate"]]
            self.assertEqual(len(jobs), 16)
            for job in jobs:
                self.assertTrue(job["ordinary_memory_learning"])
                self.assertEqual(job["dev_file"], repetition["dev"]["file"])
                self.assertEqual(job["heldout_file"], repetition["heldout"]["file"])
        self.assertFalse(design["protocol"]["heldout_ingestion"])
        self.assertEqual(design["protocol"]["comparison_memory_policy"],
                         "each_comparison_reset_then_same_warmup")
        self.assertEqual(design["protocol"]["selection"], "native")

    def test_seed_pins_and_none_initial_state(self):
        design = build_design(self.project)
        self.assertEqual(design["seeds"]["none"]["items"], [])
        self.assertIsNone(design["seeds"]["none"]["file"])
        hashes = [entry["items_sha256"] for entry in design["seeds"].values()]
        self.assertEqual(len(set(hashes)), 4)
        for seed in SEEDS[:-1]:
            self.assertEqual(len(design["seeds"][seed]["file_sha256"]), 64)

    def test_selection_is_deterministic_excludes_prior_roots_and_ignores_answers(self):
        # Another dev row shares a previously observed root and must be excluded.
        self.dev[2]["root_url"] = self.dev[0]["root_url"]
        self.save("dev", self.dev)
        first = build_design(self.project)
        self.assertEqual(first, build_design(self.project))
        self.assertFalse({"dev-0", "dev-1", "dev-2"} & set(first["selected_tasks"]["dev"]["task_ids"]))
        for split in ("dev", "heldout"):
            selected = first["selected_tasks"][split]
            self.assertEqual(len(set(selected["root_urls"])), 4)
            for repetition in first["repetitions"]:
                self.assertEqual(set(repetition[split]["task_ids"]), set(selected["task_ids"]))
        for row in self.dev + self.heldout:
            row["answer"] = "Changed answer never used by selection"
            row["question"] = "Changed question never used by selection"
        self.save("dev", self.dev)
        self.save("heldout", self.heldout)
        changed = build_design(self.project)
        self.assertEqual(first["selected_tasks"], changed["selected_tasks"])
        self.assertNotEqual(first["design_sha256"], changed["design_sha256"])

    def test_overlapping_root_or_task_ids_rejected(self):
        self.heldout[0]["root_url"] = self.dev[0]["root_url"]
        self.save("heldout", self.heldout)
        with self.assertRaisesRegex(ValueError, "root_url groups overlap"):
            build_design(self.project)
        self.heldout[0]["root_url"] = "https://unique.example/"
        self.heldout[0]["task_id"] = self.dev[0]["task_id"]
        self.save("heldout", self.heldout)
        with self.assertRaisesRegex(ValueError, "task IDs overlap"):
            build_design(self.project)

    def test_insufficient_unused_roots_rejected(self):
        self.save("dev", self.dev[:5])
        with self.assertRaisesRegex(ValueError, "four unused distinct"):
            build_design(self.project)

    def test_write_has_evaluator_only_heldout_and_identical_resume(self):
        design = build_design(self.project)
        output = self.root / "experiment"
        self.assertEqual(write_design(output, design, project=self.project), design)
        before = {path.relative_to(output).as_posix(): path.read_bytes()
                  for path in output.rglob("*") if path.is_file()}
        self.assertEqual(len(before), 7)
        self.assertFalse((output / "jobs").exists())
        manifest = (output / "manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("private answer", manifest)
        self.assertNotIn(str(self.project), manifest)
        for repetition in design["repetitions"]:
            dev_rows = read_jsonl(output / repetition["dev"]["file"])
            heldout_rows = read_jsonl(output / repetition["heldout"]["file"])
            self.assertEqual([row["task_id"] for row in dev_rows], repetition["dev"]["task_ids"])
            self.assertTrue(all(row["task_id"].startswith("dev-") for row in dev_rows))
            self.assertTrue(all(row["task_id"].startswith("heldout-") for row in heldout_rows))
        write_design(output, design, project=self.project)
        after = {path.relative_to(output).as_posix(): path.read_bytes()
                 for path in output.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_conflicting_output_is_refused_before_other_artifacts_written(self):
        output = self.root / "experiment"
        conflict = output / "evaluation_only" / "rep-01" / "heldout.jsonl"
        conflict.parent.mkdir(parents=True)
        conflict.write_text("different experiment\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            write_design(output, project=self.project)
        self.assertEqual(list(output.rglob("*.jsonl")), [conflict])
        self.assertFalse((output / "manifest.json").exists())

    def test_source_or_seed_drift_invalidates_built_design(self):
        design = build_design(self.project)
        self.dev[3]["answer"] = "source drift"
        self.save("dev", self.dev)
        with self.assertRaisesRegex(ValueError, "pinned sources, seeds, or protocol"):
            write_design(self.root / "experiment", design, project=self.project)
        design = build_design(self.project)
        seed = self.project / "seeds" / "critical.json"
        seed.write_text(json.dumps({"items": ["changed principle"]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pinned sources, seeds, or protocol"):
            write_design(self.root / "experiment", design, project=self.project)


if __name__ == "__main__":
    unittest.main()
