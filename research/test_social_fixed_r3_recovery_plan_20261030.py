"""Pure planner regression checks against retained Cairn paid evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.social_fixed_r3_recovery_plan_20261030 import ATTEMPTS, plan_cairn_ingestion

ROOT = Path(__file__).resolve().parents[1] / "runs/hss-business-formal-20261029-r01"
RECORD = ROOT / "jobs/social-fixed-r3/evolution/round_00/eval_batch/cairn_memory/1.json"


class CairnRecoveryPlanTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))
        self.events = [json.loads((ROOT / "budget/attempts" / f"{n:06d}.json").read_text(encoding="utf-8"))
                       for n in ATTEMPTS]

    def test_paid_prompts_and_durable_prestate_reconstruct_without_writes(self):
        store = ROOT / "jobs/social-fixed-r3/workspace/storage/cairn_memory/memory.json"
        before_record, before_store = RECORD.read_bytes(), store.read_bytes()
        original_events = deepcopy(self.events)
        plan = plan_cairn_ingestion(ROOT, self.record, self.events)
        context = plan["runtime_context"]
        self.assertEqual(plan["storage_rollbacks"], [])
        self.assertEqual(plan["paid_ingestion_attempts_preserved"], [16076, 16100])
        self.assertEqual(sum(m["retrieval_trials"] for m in context["db"]["memories"]), 6)
        self.assertEqual(context["task_id_counter"], 1)
        self.assertEqual(context["task_context"]["task_id"], "cairn_task_1")
        self.assertEqual(len(context["task_context"]["step_summaries"]), 3)
        self.assertEqual(len(plan["pinned_files"]), 15)
        self.assertNotIn(str(RECORD.relative_to(ROOT)), plan["pinned_files"])
        self.assertEqual(RECORD.read_bytes(), before_record)
        self.assertEqual(store.read_bytes(), before_store)
        self.assertEqual(self.events, original_events)

    def test_changed_paid_delivery_inputs_cannot_authorize_restoration(self):
        event = next(x for x in self.events if x["attempt"] == 15871)
        parsed = json.loads(event["response"]["choices"][0]["message"]["content"])
        parsed["selected_mem_ids"] = parsed["selected_mem_ids"][::-1]
        event["response"]["choices"][0]["message"]["content"] = json.dumps(parsed)
        with self.assertRaises(AssertionError):
            plan_cairn_ingestion(ROOT, self.record, self.events)

    def test_truncated_paid_ingestion_is_rejected(self):
        event = next(x for x in self.events if x["attempt"] == 16100)
        event["response"]["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(AssertionError):
            plan_cairn_ingestion(ROOT, self.record, self.events)


if __name__ == "__main__":
    unittest.main()
