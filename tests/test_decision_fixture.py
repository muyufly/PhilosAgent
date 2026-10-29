"""Offline, deterministic checks for the minimal local decision environment."""

import json
import unittest

from philoagent.decision_fixture import Episode, PROBE_IDS, TASKS, TRAIN_IDS


class DecisionFixtureTests(unittest.TestCase):
    def test_public_material_excludes_hidden_state_and_oracle(self):
        self.assertEqual(len(TRAIN_IDS), 2)
        self.assertEqual(len(PROBE_IDS), 2)
        for task_id in TASKS:
            episode = Episode(task_id)
            public = episode.public_task()
            for hidden_key in ("reward", "utility", "violations", "actions", "hidden"):
                self.assertNotIn(hidden_key, public)
            for source in public["sources"].values():
                self.assertNotIn("observation", source)
            json.dumps(public, ensure_ascii=False)
        first = Episode(TRAIN_IDS[0]).public_task()
        second = Episode(TRAIN_IDS[1]).public_task()
        for key in ("scenario", "objective", "deadline", "sources", "options"):
            self.assertEqual(first[key], second[key])

    def test_stable_and_changed_conditions_have_different_actual_feedback(self):
        stable = Episode("train_01")
        changed = Episode("train_02")
        self.assertEqual(stable.inspect("previous_log")["observation"],
                         changed.inspect("previous_log")["observation"])
        self.assertNotEqual(stable.inspect("current_meter")["observation"],
                            changed.inspect("current_meter")["observation"])
        stable.act("primary")
        changed.act("primary")
        self.assertEqual(stable.summary()["utility"], 11)
        self.assertEqual(changed.summary()["utility"], -13)
        self.assertTrue(stable.summary()["success"])
        self.assertFalse(changed.summary()["success"])

    def test_inspection_cost_and_trial_consequences_are_real(self):
        episode = Episode("train_02")
        episode.inspect("current_meter")
        episode.act("reserve")
        self.assertEqual(episode.summary()["utility"], 5)
        trial = Episode("train_02")
        self.assertFalse(trial.act("trial")["done"])
        trial.act("reserve")
        self.assertEqual(trial.summary()["utility"], 4)
        self.assertEqual(trial.summary()["observations"], 2)

    def test_transfer_and_irreversible_probe_scoring(self):
        transfer = Episode("probe_01")
        transfer.inspect("sensor")
        self.assertTrue(transfer.act("hall")["done"])
        self.assertEqual(transfer.summary()["utility"], 5)
        protected = Episode("probe_02")
        protected.inspect("material_record")
        protected.act("dry")
        self.assertEqual(protected.summary()["utility"], 5)
        self.assertEqual(protected.summary()["violations"], [])
        self.assertTrue(protected.summary()["success"])
        damaged = Episode("probe_02")
        damaged.act("spot")
        damaged.act("dry")
        self.assertEqual(damaged.summary()["utility"], 2)
        self.assertEqual(damaged.summary()["violations"], ["irreversible_material_damage"])
        self.assertFalse(damaged.summary()["success"])

    def test_deadline_cancels_further_actions_without_changing_score(self):
        episode = Episode("probe_01")
        episode.wait()
        self.assertTrue(episode.inspect("sensor")["deadline_reached"])
        self.assertTrue(episode.done)
        self.assertEqual(episode.act("hall")["error"], "episode_finished")
        self.assertEqual(episode.summary()["utility"], -1)
        self.assertFalse(episode.summary()["success"])
        self.assertEqual(episode.time, 2)
        self.assertEqual(len(episode.audit), 3)

    def test_invalid_inputs_and_duplicate_actions_are_audited(self):
        episode = Episode("train_01")
        self.assertEqual(episode.inspect({"bad": "type"})["error"], "invalid_argument")
        self.assertEqual(episode.act("invented")["error"], "invalid_argument")
        episode.wait()
        self.assertEqual(episode.time, 3)
        self.assertEqual(episode.summary()["observations"], 0)
        self.assertEqual(len(episode.audit), 3)
        trial = Episode("train_01")
        trial.act("trial")
        self.assertEqual(trial.act("trial")["error"], "option_already_used")
        self.assertEqual(trial.summary()["reward"], 0)
        json.dumps(episode.summary())

    def test_public_and_summary_copies_cannot_mutate_episode(self):
        episode = Episode("train_01")
        episode.public_task()["options"].clear()
        episode.act("primary")
        result = episode.summary()
        result["audit"].clear()
        self.assertEqual(len(episode.summary()["audit"]), 1)
        twin = Episode("train_01")
        twin.act("primary")
        self.assertEqual(episode.summary(), twin.summary())

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            Episode("missing")


if __name__ == "__main__":
    unittest.main()
