"""Offline regression checks for bounded native reflection inputs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from philoagent.orientation import bootstrap, load
from philoagent.reflection_view import (PHI_BEGIN, PHI_END, ReflectionClient, ReflectionCoverage,
    analysis_prompt, build_view, compact, deduplicate_phi, generation_prompt,
    project_record, replace_phi, sha)


def record(snapshot, task_id="train_01"):
    return {
        "status": "success", "question": "Public deadline=3; trial may incur rework.",
        "agent_result": "The actor claims the philosophy influenced trial; attribution uncertain.",
        "is_correct": True, "judgement": "correct", "orientation_snapshot": snapshot.to_dict(),
        "agent_messages": ["REPEATED_FULL_NATIVE_MESSAGES"],
        "budget_after": {"irrelevant": "BUDGET_METADATA"},
        "trajectory_summary": "GENERATED_CAUSAL_ASSERTION_PHI_CAUSED_SUCCESS",
        "evaluation": {"task_id": task_id, "done": True, "reward": 4, "cost": 0, "utility": 4,
                       "feedback": "备用完成6单位，试验损耗2单位", "success": True, "completed": True,
                       "unchosen_outcomes": {"SECRET_ORACLE": 100},
                       "audit": [{"tool": "act", "argument": "trial", "result": {"observation": "试验损耗2单位"}}]},
        "agent_trajectory": [
            {"name": "plan", "value": "Plan conditional branches", "think": "Time is scarce",
             "memory_guidance": "header\n- " + "\n- ".join(snapshot.items) + "\n- Ordinary M recommends trial."},
            {"name": "action", "think": "I treated reversible as sufficiently cheap.",
             "tool_calls": [{"name": "act", "arguments": {"option": "trial"}}],
             "obs": "Actual observation: 试验损耗2单位", "memory_guidance": "Other ordinary advice"},
            {"name": "action", "think": "Report after acting", "tool_calls": [{"name": "final_answer", "arguments": "answer"}],
             "obs": "answer"}],
    }


class ReflectionViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        bootstrap(self.root / "lineage", "parent", ["【ONT】对象不等于模型。", "【EPI】实践反馈需解释。", "【ACT】比较行动后果。"])
        self.snapshot = load(self.root / "lineage", "parent")

    def test_projection_preserves_feedback_reasoning_and_ordinary_memory(self):
        original = record(self.snapshot)
        result = project_record(original, "train_01", self.snapshot)
        summary = json.loads(result["trajectory_summary"])
        self.assertEqual(summary["actual_environment_events"], original["evaluation"]["audit"])
        self.assertEqual(summary["realized_outcome"]["utility"], 4)
        self.assertEqual(result["question"], original["question"])
        self.assertEqual(result["agent_result"], original["agent_result"])
        self.assertEqual(result["agent_trajectory"][1]["think"], original["agent_trajectory"][1]["think"])
        self.assertEqual(result["agent_trajectory"][1]["obs"], original["agent_trajectory"][1]["obs"])
        self.assertIn("Ordinary M recommends trial.", result["agent_trajectory"][0]["memory_guidance"])
        self.assertEqual(result["agent_trajectory"][1]["memory_guidance"], "Other ordinary advice")
        self.assertIn("not causal evidence", result["agent_trajectory"][0]["memory_guidance_provenance"])

    def test_projection_removes_duplicates_not_philosophy_references(self):
        result = project_record(record(self.snapshot), "train_01", self.snapshot)
        encoded = compact(result)
        for forbidden in ("REPEATED_FULL_NATIVE_MESSAGES", "BUDGET_METADATA", "SECRET_ORACLE",
                          "GENERATED_CAUSAL_ASSERTION_PHI_CAUSED_SUCCESS", *self.snapshot.items):
            self.assertNotIn(forbidden, encoded)
        self.assertIn(self.snapshot.sha256[:12], encoded)
        for label in ("ONT", "EPI", "ACT"):
            self.assertIn(label, encoded)
        self.assertEqual(result["agent_trajectory"][2]["tool_calls"][0]["arguments"],
                         "[see agent_result: actor self-report]")

    def test_projection_rejects_probe_wrong_identity_or_changed_phi(self):
        for task_id, source in (("probe_01", record(self.snapshot, "probe_01")),
                                ("train_01", record(self.snapshot, "train_02"))):
            with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                project_record(source, task_id, self.snapshot)
        source = record(self.snapshot)
        source["orientation_snapshot"]["items"] = ["wrong parent"]
        with self.assertRaises(ValueError):
            project_record(source, "train_01", self.snapshot)

    def test_exact_replacement_preserves_unmatched_advice(self):
        text = "Advice before. " + self.snapshot.items[1] + " Advice after, with EPI paraphrase."
        result = replace_phi(text, self.snapshot)
        self.assertTrue(result.startswith("Advice before. "))
        self.assertTrue(result.endswith(" Advice after, with EPI paraphrase."))
        self.assertIn("/ EPI", result)

    def test_view_maps_hashes_and_leaves_raw_logs_unchanged(self):
        training = self.root / "training"
        training.mkdir()
        original = {}
        for task in ("train_01", "train_02"):
            data = compact(record(self.snapshot, task)).encode("utf-8")
            (training / f"{task}.json").write_bytes(data)
            original[task] = data
        manifest = build_view(training, self.root / "view", self.snapshot)
        self.assertEqual(manifest["generated_summary_calls"], 0)
        for task, raw in original.items():
            self.assertEqual((training / f"{task}.json").read_bytes(), raw)
            mapping = manifest["sources"][task]
            self.assertEqual(mapping["source_sha256"], sha(raw))
            self.assertEqual(mapping["projection_sha256"], sha(Path(mapping["projection"]).read_bytes()))
        self.assertEqual(json.loads((self.root / "view/current_phi.json").read_text(encoding="utf-8")), self.snapshot.to_dict())
        with self.assertRaises(FileExistsError):
            build_view(training, self.root / "view", self.snapshot)

    def test_view_rejects_extra_task_before_creating_output(self):
        training = self.root / "training"
        training.mkdir()
        for task in ("train_01", "train_02", "probe_01"):
            (training / f"{task}.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_view(training, self.root / "view", self.snapshot)
        self.assertFalse((self.root / "view").exists())

    def test_phi_only_prompt_contains_each_original_paragraph_once_without_source(self):
        parent = {"source_sha256": "hash-only", "code": "class SECRET_PROVIDER_SOURCE: pass"}
        prompt = analysis_prompt(self.snapshot, parent, self.root / "view", [self.root / "memory.json"])
        self.assertNotIn("SECRET_PROVIDER_SOURCE", prompt)
        for item in self.snapshot.items:
            self.assertEqual(prompt.count(item), 1)
        self.assertIn("Keeping Phi unchanged", prompt)
        self.assertIn("cannot be revoked", prompt)

    def test_native_repeated_task_keeps_phi_once_before_audit_boundary(self):
        prompt = analysis_prompt(self.snapshot, {"source_sha256": "hash"}, self.root, [])
        messages = [{"role": "user", "content": prompt},
                    {"role": "user", "content": [{"type": "text", "text": "Native repeats task:\n" + prompt}]}]
        before = deepcopy(messages)
        result = deduplicate_phi(messages)
        encoded = compact(result)
        self.assertEqual(encoded.count(PHI_BEGIN), 1)
        self.assertEqual(encoded.count(PHI_END), 1)
        for item in self.snapshot.items:
            self.assertEqual(encoded.count(item), 1)
        self.assertEqual(messages, before)
        calls = []
        raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs))))
        ReflectionClient(raw).create(model="offline", messages=messages)
        self.assertEqual(calls[0]["messages"], result)

    def test_generation_retains_full_phi_without_implementation_reference(self):
        prompt = generation_prompt("Evidence quotes " + self.snapshot.items[1], self.snapshot, ["existing"])
        for item in self.snapshot.items:
            self.assertEqual(prompt.count(item), 1)
        self.assertIn("omit orientation_items", prompt)
        self.assertIn("six-chapter worldview", prompt)
        self.assertIn("**Class Name**", prompt)
        self.assertNotIn("class LightweightMemoryProvider", prompt)

    def test_tool_json_uses_utf8_without_extra_indentation(self):
        result = compact({"observation": "实际损耗", "value": [1, 2]})
        self.assertNotIn("\\u", result)
        self.assertNotIn("\n", result)
        self.assertEqual(json.loads(result)["observation"], "实际损耗")

    def test_coverage_requires_successful_summary_and_valid_step_for_each_task(self):
        coverage = ReflectionCoverage()
        coverage.trajectories(["train_01", "train_02"], {
            "train_01": {"trajectory_summary": "observed outcome"},
            "train_02": {"error": "not found"}})
        coverage.specific_steps("train_01", [0, 99], {"task_id": "train_01", "requested_steps": {
            "step_0": {"name": "plan", "source_step": 0}, "step_99": {"error": "out of range"}}})
        coverage.specific_steps("train_02", [0], {"task_id": "train_02", "requested_steps": {
            "step_0": {"name": "action", "source_step": 0}}})
        self.assertFalse(coverage.report()["complete"])
        self.assertEqual(coverage.report()["step_indices"]["train_01"], [0])
        with self.assertRaises(ValueError):
            coverage.require_complete()
        coverage.trajectories(["train_02"], {"train_02": {"trajectory_summary": "observed outcome"}})
        coverage.require_complete()
        self.assertTrue(coverage.report()["complete"])

    def test_coverage_rejects_unreturned_indices_mismatched_tasks_and_self_report(self):
        coverage = ReflectionCoverage()
        coverage.trajectories(["train_01", "train_02"], {"agent_result": "I read both tasks"})
        coverage.specific_steps("train_01", [0], {"task_id": "train_02", "requested_steps": {
            "step_0": {"name": "plan", "source_step": 0}}})
        coverage.specific_steps("train_02", [0], {"task_id": "train_02", "requested_steps": {
            "step_0": {"name": "plan", "source_step": 1}}})
        self.assertEqual(coverage.report()["summary_tasks"], [])
        self.assertEqual(coverage.report()["step_indices"], {"train_01": [], "train_02": []})


if __name__ == "__main__":
    unittest.main()
