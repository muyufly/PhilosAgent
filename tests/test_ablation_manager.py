"""Prevent incomplete real-model jobs from being advertised as finished."""

import json
from pathlib import Path
import tempfile
import unittest

from philoagent.ablation import fingerprint, summarize, validate_job_result, write_json


class ManagerTests(unittest.TestCase):
    def test_incomplete_rounds_or_heldout_cannot_pass(self):
        protocol = {"rounds": 2, "heldout_task_count": 4}
        complete = {"rounds_completed": 2, "heldout_count": 4, "heldout_accuracy": 0.5}
        validate_job_result(complete, protocol)
        for delta in ({"rounds_completed": 1}, {"heldout_count": 3},
                      {"heldout_accuracy": None}, {"status": "failed"}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                validate_job_result({**complete, **delta}, protocol)

    def test_configuration_fingerprint_does_not_contain_credentials(self):
        values = {"OPENAI_API_KEY": "private-model-key", "DEEPSEEK_API_KEY": "private-search-key",
                  "DEFAULT_MODEL": "offline-model"}
        text = json.dumps(fingerprint(values))
        self.assertNotIn("private-model-key", text)
        self.assertNotIn("private-search-key", text)
        self.assertIn("offline-model", text)

    def test_partial_matrix_is_not_complete_and_pairs_match_replicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for rep in (0, 1):
                for p in (False, True):
                    jobs.append({"job_id": f"p{int(p)}r{rep}", "seed": "none", "replicate": rep,
                                 "code_evolve": p, "orientation_evolve": False})
            write_json(root / "manifest.json", {"design_sha256": "offline", "jobs": jobs,
                       "seeds": {"none": {}}, "repetitions": [{}, {}], "inference_limit": "small"})
            for job, score in ((jobs[0], 0.25), (jobs[1], 0.75), (jobs[2], 1.0)):
                write_json(root / "jobs" / job["job_id"] / "result.json",
                           {"status": "complete", "heldout_accuracy": score})
            result = summarize(root)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["counts"]["complete"], 3)
        paired = result["paired_effects"][0]
        self.assertEqual(paired["paired_replicates"], 1)
        self.assertEqual(paired["accuracy_differences"], [0.5])


if __name__ == "__main__":
    unittest.main()
