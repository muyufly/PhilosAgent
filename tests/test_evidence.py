import json
from pathlib import Path
import tempfile
import unittest

from philoagent.evidence import append_event, classify_code_change, record_code_snapshot, record_runtime_call, sha256_file


class EvidenceTests(unittest.TestCase):
    def test_actual_operator_and_numeric_changes_are_structural(self):
        original = "def score(age):\n    return age * 0.5\n"
        changed = "def score(age):\n    return age + 0.8\n"
        result = classify_code_change(original, changed)
        self.assertEqual(result["category"], "executable_syntax_changed")
        self.assertTrue(result["non_string_ast_changed"])
        self.assertFalse(result["causal_effect_established"])
        self.assertFalse(result["behavioral_change_established"])

    def test_prompt_and_logic_string_changes_require_review(self):
        for before, after in (("PROMPT = 'reflect'", "PROMPT = 'reflect critically'"), ("value = record['new']", "value = record['old']")):
            with self.subTest(before=before):
                result = classify_code_change(before, after)
                self.assertEqual(result["category"], "string_literal_changes_require_review")
                self.assertFalse(result["non_string_ast_changed"])
                self.assertTrue(result["requires_review"])

    def test_whitespace_comments_and_docstrings(self):
        before = '"""old module"""\ndef f(x):\n    """old function"""\n    return x + 1\n'
        after = '"""new module"""\n# changed comment\ndef f(x):\n    """different documentation"""\n    return x+1\n'
        self.assertEqual(classify_code_change(before, after)["category"], "format_or_docstring_only")
        self.assertEqual(classify_code_change(before, before)["category"], "unchanged")

    def test_invalid_python_is_not_classified_as_effective_change(self):
        result = classify_code_change("x=1", "def broken(")
        self.assertEqual(result["category"], "invalid_python")
        self.assertIn("after", result["parse_errors"])
        self.assertFalse(result["non_string_ast_changed"])

    def test_runtime_distinguishes_disk_from_loaded_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code, orientation, log = root / "memory.py", root / "orientation.md", root / "events.jsonl"
            code.write_text("def retrieve():\n    return []\n", encoding="utf-8")
            orientation.write_text("Test orientation", encoding="utf-8")
            loaded_hash = sha256_file(code)
            snapshot = record_code_snapshot(log, code, orientation, group="test")
            self.assertEqual(snapshot["orientation_sha256"], sha256_file(orientation))
            self.assertEqual(snapshot["event"], "code_snapshot")
            code.write_text("def retrieve():\n    return [1]\n", encoding="utf-8")
            call = record_runtime_call(log, code, "retrieve", orientation_path=orientation, loaded_code_sha256=loaded_hash, task_id="q1")
            self.assertFalse(call["disk_matches_loaded_code"])
            self.assertEqual(call["loaded_code_sha256"], loaded_hash)
            unverified = record_runtime_call(log, code, "update")
            self.assertIsNone(unverified["disk_matches_loaded_code"])
            self.assertEqual(unverified["loaded_hash_source"], "not_provided")
            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 3)
            self.assertIn("timestamp_utc", events[0])
            with self.assertRaises(ValueError):
                record_runtime_call(log, code, "retrieve", loaded_code_sha256="not-a-hash")
            with self.assertRaises(ValueError):
                append_event(log, {"value": float("nan")})
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
