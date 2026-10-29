import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from philoagent.data import normalize_root_url, prepare_dataset, read_jsonl, split_rows, verify_manifest


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [
            {"id": f"q{i}", "question": f"Question {i}", "answer": f"Answer {i}", "root_url": f"https://site{i // 2}.example/", "info": {"type": "test"}}
            for i in range(12)
        ]
        self.source = self.root / "source.jsonl"
        self.source.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n", encoding="utf-8")

    def test_deterministic_group_split_and_source_hash(self):
        first = prepare_dataset(self.source, self.root / "first")
        second = prepare_dataset(self.source, self.root / "second")
        self.assertEqual(first["manifest"], second["manifest"])
        self.assertEqual(Path(first["dev_path"]).read_bytes(), Path(second["dev_path"]).read_bytes())
        dev, heldout = read_jsonl(first["dev_path"]), read_jsonl(first["heldout_path"])
        self.assertTrue(dev and heldout)
        self.assertFalse({row["task_id"] for row in dev} & {row["task_id"] for row in heldout})
        self.assertFalse({normalize_root_url(row["root_url"]) for row in dev} & {normalize_root_url(row["root_url"]) for row in heldout})
        self.assertEqual(len(dev) + len(heldout), len(self.rows))
        self.assertEqual(first["manifest"]["source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(verify_manifest(first["manifest_path"]), first["manifest"])
        self.assertEqual(prepare_dataset(self.source, self.root / "first")["manifest"], first["manifest"])
        self.assertIn("info", dev[0])

    def test_input_order_does_not_change_membership(self):
        a = split_rows(self.rows)
        b = split_rows(list(reversed(self.rows)))
        self.assertEqual(a, b)

    def test_missing_invalid_duplicate_and_small_inputs_rejected(self):
        for field in ("question", "answer", "root_url"):
            bad = [dict(row) for row in self.rows]
            del bad[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                split_rows(bad)
        with self.assertRaisesRegex(ValueError, "Duplicate task ID"):
            split_rows(self.rows + [dict(self.rows[0])])
        duplicate = dict(self.rows[0], id="different-id")
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            split_rows(self.rows + [duplicate])
        with self.assertRaises(ValueError):
            split_rows(self.rows[:1])
        with self.assertRaisesRegex(ValueError, "distinct root_url"):
            split_rows(self.rows[:2])
        for fraction in (0, 1, -0.1, float("nan"), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_rows(self.rows, heldout_fraction=fraction)

    def test_generated_ids_and_root_normalization(self):
        rows = [{key: value for key, value in row.items() if key != "id"} for row in self.rows]
        dev, heldout = split_rows(rows)
        self.assertTrue(all(row["task_id"].startswith("wwqa-") for row in dev + heldout))
        self.assertEqual(normalize_root_url("HTTPS://EXAMPLE.com:443/path/#fragment"), "https://example.com/path")
        with self.assertRaises(ValueError):
            normalize_root_url("https://user:secret@example.com/")

    def test_tampering_and_overwrite_rejected(self):
        result = prepare_dataset(self.source, self.root / "out")
        Path(result["dev_path"]).write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_manifest(result["manifest_path"])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_dataset(self.source, self.root / "out")

    def test_source_tamper_rejected(self):
        result = prepare_dataset(self.source, self.root / "out")
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Source dataset"):
            verify_manifest(result["manifest_path"])

    def test_manifest_membership_tamper_rejected(self):
        result = prepare_dataset(self.source, self.root / "out")
        manifest = result["manifest"]
        manifest["splits"]["dev"]["task_ids"][0] = "unseen"
        Path(result["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "membership"):
            verify_manifest(result["manifest_path"])


if __name__ == "__main__":
    unittest.main()
