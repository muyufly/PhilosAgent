"""Offline checks for a draft bank, not a paid or confirmatory experiment."""

import json
import unittest

from philoagent.decision_fixture import TASKS as SMOKE_TASKS
from philoagent.formal_tasks import (
    DESIGN_VERSION, FORMAL_MATRIX_AUTHORIZED, Episode, PROBE_IDS, STREAMS,
    TASKS, TRAIN_IDS, export_evaluator_spec, oracle, require_learning_allowed,
)


class FormalTaskTests(unittest.TestCase):
    def test_three_distinct_streams_and_no_smoke_identity_reuse(self):
        self.assertFalse(FORMAL_MATRIX_AUTHORIZED)
        self.assertIn("draft", DESIGN_VERSION)
        self.assertEqual(len(STREAMS), 3)
        self.assertEqual(len(TRAIN_IDS), 12)
        self.assertEqual(len(PROBE_IDS), 6)
        self.assertEqual(set(TRAIN_IDS) | set(PROBE_IDS), set(TASKS))
        self.assertFalse(set(TRAIN_IDS) & set(PROBE_IDS))
        self.assertFalse(set(TASKS) & set(SMOKE_TASKS))
        self.assertEqual(len({stream["causal_structure"] for stream in STREAMS.values()}), 3)
        for stream in STREAMS.values():
            self.assertEqual(len(stream["train_ids"]), 4)
            self.assertEqual(len(stream["probe_ids"]), 2)

    def test_public_tool_boundary_has_no_hidden_scores_or_transitions(self):
        forbidden = {"reward", "utility", "violations", "initial_state", "actions",
                     "requirements", "design_note", "updates", "when", "evaluator_only"}

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & value.keys())
                for part in value.values():
                    check(part)
            elif isinstance(value, list):
                for part in value:
                    check(part)

        for task_id in TASKS:
            episode = Episode(task_id)
            check(episode.public_task())
            self.assertNotIn("split", episode.public_task())
            json.dumps(episode.public_task(), ensure_ascii=False)
            for source_id in TASKS[task_id]["sources"]:
                source_episode = Episode(task_id)
                check(source_episode.inspect(source_id))
            for option_id in TASKS[task_id]["options"]:
                action_episode = Episode(task_id)
                check(action_episode.act(option_id))

    def test_evaluator_export_is_complete_and_copy_isolated(self):
        for task_id in TASKS:
            spec = export_evaluator_spec(task_id)
            self.assertTrue(spec["evaluator_only"])
            self.assertFalse(spec["formal_matrix_authorized"])
            self.assertEqual(spec["public"], TASKS[task_id])
            self.assertEqual(set(spec["sources"]), set(TASKS[task_id]["sources"]))
            self.assertEqual(set(spec["actions"]), set(TASKS[task_id]["options"]))
            json.dumps(spec, ensure_ascii=False, sort_keys=True)
            spec["actions"].clear()
            self.assertTrue(export_evaluator_spec(task_id)["actions"])
        episode = Episode("sequence_learn_01")
        before = export_evaluator_spec(episode.task_id)
        episode.act("unfold")
        self.assertEqual(export_evaluator_spec(episode.task_id), before)

    def test_stable_records_do_not_track_a_changed_latent_convention(self):
        stable = Episode("provenance_learn_02")
        changed = Episode("provenance_learn_03")
        for source in ("desk_note", "copied_note"):
            self.assertEqual(stable.inspect(source)["observation"], changed.inspect(source)["observation"])
        stable.act("index_x")
        changed.act("index_x")
        self.assertTrue(stable.summary()["success"])
        self.assertFalse(changed.summary()["success"])
        self.assertFalse(changed.summary()["completed"])
        self.assertEqual(changed.summary()["utility"], -9)

    def test_information_has_value_without_claiming_oracle_feasibility(self):
        # These two predeclared hidden worlds are diagnostic, not extra samples.
        # No probability over them is claimed: compare their worst-case utility.
        observed_policy = []
        blind_x = []
        blind_y = []
        safe = []
        for task_id in ("provenance_learn_01", "provenance_learn_03"):
            inspected = Episode(task_id)
            reading = inspected.inspect("batch_key")["observation"]
            inspected.act("index_x" if "甲式" in reading else "index_y")
            observed_policy.append(inspected.summary()["utility"])
            for option, scores in (("index_x", blind_x), ("index_y", blind_y), ("literal_list", safe)):
                episode = Episode(task_id)
                episode.act(option)
                scores.append(episode.summary()["utility"])
        self.assertEqual(observed_policy, [12, 12])
        self.assertGreater(min(observed_policy), max(min(blind_x), min(blind_y), min(safe)))

    def test_information_can_have_zero_value_or_destroy_the_window(self):
        immediate = Episode("provenance_learn_04")
        immediate.act("index_x")
        checked = Episode("provenance_learn_04")
        checked.inspect("batch_key")
        checked.act("index_x")
        self.assertEqual(immediate.summary()["utility"], 14)
        self.assertEqual(checked.summary()["utility"], 12)
        deadline = Episode("authority_learn_04")
        self.assertTrue(deadline.inspect("consent_register")["deadline_reached"])
        self.assertEqual(deadline.act("full_program")["error"], "episode_finished")
        self.assertFalse(deadline.summary()["completed"])
        low_cost = Episode("authority_learn_04")
        low_cost.act("anonymous_digest")
        self.assertTrue(low_cost.summary()["success"])
        self.assertEqual(low_cost.summary()["utility"], 8)

    def test_preparation_creates_the_measurable_state(self):
        episode = Episode("sequence_learn_03")
        self.assertIn("没有可测", episode.inspect("seam_gauge")["observation"])
        self.assertFalse(episode.act("unfold")["done"])
        self.assertIn("纵脊", episode.inspect("seam_gauge")["observation"])
        self.assertTrue(episode.act("top_clip")["done"])
        score = episode.summary()
        self.assertTrue(score["success"])
        self.assertEqual(score["utility"], 11)
        self.assertEqual(score["time"], 4)
        # Acting and then measuring once reaches the same completion for less.
        economical = Episode("sequence_learn_03")
        economical.act("unfold")
        economical.inspect("seam_gauge")
        economical.act("top_clip")
        self.assertEqual(economical.summary()["utility"], 12)
        self.assertEqual(economical.summary()["time"], 3)

    def test_precondition_failure_is_separate_from_invalid_output(self):
        episode = Episode("sequence_learn_01")
        self.assertEqual(episode.act("side_clip")["error"], "precondition_failed")
        self.assertEqual(episode.summary()["precondition_failures"], 1)
        self.assertEqual(episode.summary()["invalid_calls"], 0)
        episode.act("unfold")
        episode.act("side_clip")
        self.assertTrue(episode.summary()["success"])
        self.assertEqual(episode.summary()["reward"], 13)

    def test_prepared_case_requires_no_extra_preparation_or_measurement(self):
        good = Episode("sequence_learn_04")
        good.act("side_clip")
        self.assertTrue(good.summary()["success"])
        wasted = Episode("sequence_learn_04")
        self.assertEqual(wasted.act("unfold")["error"], "precondition_failed")
        self.assertEqual(wasted.act("side_clip")["error"], "episode_finished")
        self.assertFalse(wasted.summary()["completed"])

    def test_cross_domain_intervention_respects_different_operation_duration(self):
        episode = Episode("sequence_probe_01")
        self.assertEqual(episode.act("wind")["time"], 2)
        self.assertEqual(episode.act("short_lever")["time"], 3)
        self.assertTrue(episode.summary()["success"])
        self.assertEqual(episode.summary()["utility"], 17)

    def test_reading_authority_does_not_create_it_but_renewal_does(self):
        read_only = Episode("authority_learn_03")
        read_only.inspect("consent_register")
        read_only.act("full_program")
        self.assertEqual(read_only.summary()["utility"], 16)
        self.assertEqual(read_only.summary()["violations"], ["unauthorized_identity_release"])
        self.assertFalse(read_only.summary()["success"])
        renewed = Episode("authority_learn_03")
        renewed.act("renew_consent")
        renewed.act("full_program")
        self.assertEqual(renewed.summary()["utility"], 14)
        self.assertTrue(renewed.summary()["success"])
        self.assertEqual(renewed.summary()["time"], 4)

    def test_read_then_renew_can_use_up_the_window_without_delivering(self):
        episode = Episode("authority_learn_03")
        episode.inspect("consent_register")
        episode.act("renew_consent")
        self.assertTrue(episode.done)
        self.assertFalse(episode.summary()["completed"])
        self.assertEqual(episode.summary()["cost"], 6)
        self.assertEqual(episode.act("full_program")["error"], "episode_finished")

    def test_positive_reward_never_cancels_an_irreversible_violation(self):
        cases = (("provenance_probe_02", ("cut_relief",), 16, "irreversible_master_damage"),
                 ("sequence_probe_02", ("open_seal", "inside_portrait"), 20, "irreversible_seal_break"),
                 ("authority_probe_02", ("play_erase",), 19, "irreversible_source_deletion"))
        for task_id, actions, reward, violation in cases:
            episode = Episode(task_id)
            for action in actions:
                episode.act(action)
            score = episode.summary()
            self.assertTrue(score["completed"])
            self.assertEqual(score["reward"], reward)
            self.assertEqual(score["violations"], [violation])
            self.assertFalse(score["constraint_passed"])
            self.assertFalse(score["success"])

    def test_later_safe_action_cannot_repair_an_existing_violation(self):
        episode = Episode("sequence_probe_02")
        episode.act("open_seal")
        episode.act("sealed_display")
        self.assertEqual(episode.summary()["utility"], 9)
        self.assertFalse(episode.summary()["success"])
        self.assertEqual(episode.summary()["violations"], ["irreversible_seal_break"])

    def test_insufficient_time_does_not_execute_or_charge_an_operation(self):
        episode = Episode("authority_learn_04")
        result = episode.act("renew_consent")
        self.assertEqual(result["error"], "insufficient_time")
        self.assertEqual(episode.time, 2)
        self.assertEqual(episode.summary()["cost"], 0)
        self.assertEqual(episode.summary()["observations"], 0)
        self.assertFalse(episode.summary()["completed"])
        # Scoring cannot drift if an actor keeps calling after termination.
        before = episode.summary()
        episode.act("full_program")
        after = episode.summary()
        for key in ("time", "reward", "cost", "utility", "violations"):
            self.assertEqual(before[key], after[key])

    def test_training_gate_rejects_every_probe_and_mixed_batch(self):
        self.assertEqual(require_learning_allowed(TRAIN_IDS), TRAIN_IDS)
        self.assertEqual(require_learning_allowed(TRAIN_IDS[0]), (TRAIN_IDS[0],))
        for task_id in PROBE_IDS:
            with self.assertRaises(ValueError):
                require_learning_allowed((task_id,))
            with self.assertRaises(ValueError):
                Episode(task_id).learning_feedback()
        for batch in ((), (TRAIN_IDS[0], PROBE_IDS[0]), ("missing",), ({"bad": 1},)):
            with self.assertRaises(ValueError):
                require_learning_allowed(batch)

    def test_learning_feedback_contains_only_the_executed_path(self):
        episode = Episode("authority_learn_03")
        episode.act("anonymous_digest")
        feedback = episode.learning_feedback()
        self.assertEqual(feedback["reward"], 8)
        self.assertEqual([item["argument"] for item in feedback["audit"]], ["anonymous_digest"])
        for key in ("initial_state", "actions", "sources", "oracle", "requirements"):
            self.assertNotIn(key, feedback)
        feedback["audit"].clear()
        self.assertEqual(len(episode.summary()["audit"]), 1)

    def test_oracle_upper_bounds_are_reachable_and_do_not_leak_into_public(self):
        expected = {
            **{f"provenance_learn_{i:02d}": 14 for i in range(1, 5)},
            "provenance_probe_01": 15, "provenance_probe_02": 16,
            **{f"sequence_learn_{i:02d}": 13 for i in range(1, 5)},
            "sequence_probe_01": 17, "sequence_probe_02": 9,
            "authority_learn_01": 18, "authority_learn_02": 18,
            "authority_learn_03": 14, "authority_learn_04": 18,
            "authority_probe_01": 11, "authority_probe_02": 15,
        }
        for task_id, ceiling in expected.items():
            before = Episode(task_id).public_task()
            result = oracle(task_id)
            self.assertTrue(result["clairvoyant"])
            self.assertEqual(result["summary"]["utility"], ceiling, task_id)
            replay = Episode(task_id)
            for step in result["route"]:
                if step["tool"] == "wait":
                    replay.wait()
                else:
                    getattr(replay, step["tool"])(step["argument"])
            self.assertEqual(replay.summary(), result["summary"])
            self.assertTrue(replay.summary()["success"])
            self.assertEqual(Episode(task_id).public_task(), before)
        # This score ceiling is a lucky hidden-state route, not evidence that
        # the actor can discover the current mirrored convention in this case.
        self.assertEqual(oracle("provenance_probe_02")["route"],
                         [{"tool": "act", "argument": "cut_intaglio"}])

    def test_unconstrained_reward_optimum_differs_from_legal_optimum(self):
        for task_id, legal, unconstrained in (("sequence_probe_02", 9, 20),
                                               ("authority_probe_02", 15, 19)):
            self.assertEqual(oracle(task_id)["summary"]["utility"], legal)
            illegal = oracle(task_id, allow_violations=True)
            self.assertEqual(illegal["summary"]["utility"], unconstrained)
            self.assertFalse(illegal["summary"]["success"])

    def test_deterministic_copies_and_invalid_calls_are_audited(self):
        left, right = Episode("sequence_learn_01"), Episode("sequence_learn_01")
        left.public_task()["options"].clear()
        for episode in (left, right):
            episode.inspect({"bad": "type"})
            episode.act("unfold")
            self.assertEqual(episode.act("unfold")["error"], "option_already_used")
            episode.act("side_clip")
        self.assertEqual(left.summary(), right.summary())
        self.assertEqual(left.summary()["invalid_calls"], 2)
        self.assertEqual(len(left.summary()["audit"]), 4)
        summary = left.summary()
        summary["audit"].clear()
        self.assertEqual(len(left.summary()["audit"]), 4)
        json.dumps(left.summary(), ensure_ascii=False)

    def test_unknown_task_is_rejected(self):
        for task_id in ("missing", {"bad": "type"}):
            with self.assertRaises(ValueError):
                Episode(task_id)
            with self.assertRaises(ValueError):
                export_evaluator_spec(task_id)


if __name__ == "__main__":
    unittest.main()
