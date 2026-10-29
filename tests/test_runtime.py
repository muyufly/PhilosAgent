"""Offline execution checks using the real lightweight memory implementation."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "MemEvolve" / "Flash-Searcher-main"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from EvolveLab.memory_types import MemoryItemType, MemoryRequest, MemoryStatus, TrajectoryData
from philoagent.orientation import OrientationIntegrityError, bootstrap, fork, load
from philoagent.runtime import OrientedProvider, configure_provider


BASELINE_SOURCE = VENDOR_ROOT / "EvolveLab" / "providers" / "lightweight_memory_provider.py"


def import_source(path, module_name):
    """Load actual source without importing unrelated optional provider packages."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoNetworkModel:
    def __call__(self, *args, **kwargs):
        raise AssertionError("This test must use the offline _call_llm stub")


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_module = import_source(BASELINE_SOURCE, "_philo_test_lightweight")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.lineage = self.work / "lineage"
        self.evidence = self.work / "evidence"
        environment = patch.dict(os.environ, {
            "PHILO_LINEAGE_ROOT": str(self.lineage),
            "PHILO_EVIDENCE_DIR": str(self.evidence),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.methods = ["Treat explanations as provisional.", "Check contrary evidence before retaining a claim."]
        self.learned_strategy = "When sources disagree, compare their dates before combining their claims."
        self.learned_operation = "Preserve the source URL alongside each extracted statement for later checking."

    def _offline_response(self, prompt):
        if "Select and synthesize the most relevant memories" in prompt:
            return json.dumps({"selected_indices": [1], "guidance": "Consider checking the available sources."})
        if "Extract reusable learnings" in prompt:
            return json.dumps({
                "strategic": [self.learned_strategy],
                "operational": [self.learned_operation],
            })
        raise AssertionError("Unexpected native LLM operation in offline test")

    def _native_provider(self, provider_id="lightweight_memory", **overrides):
        original_config = {
            "storage_dir": str(self.work / "episodic" / provider_id),
            "model": NoNetworkModel(),
            "enable_longterm_provision": False,
            **overrides,
        }
        config, snapshot = configure_provider(provider_id, original_config)
        native = self.baseline_module.LightweightMemoryProvider(config)
        previously_disabled = native.logger.disabled
        native.logger.disabled = True
        self.addCleanup(setattr, native.logger, "disabled", previously_disabled)
        native._call_llm = Mock(side_effect=self._offline_response)
        code_hash = hashlib.sha256(BASELINE_SOURCE.read_bytes()).hexdigest()
        wrapper = OrientedProvider(native, snapshot, BASELINE_SOURCE, code_hash)
        return wrapper, native, snapshot, original_config

    def _trajectory(self):
        return TrajectoryData(
            query="Which claim is supported?",
            trajectory=[{"role": "assistant", "content": "Compared two dated source documents."}],
            result="The claim supported by the newer primary source.",
            metadata={"is_correct": True, "caller_value": "preserved"},
        )

    def _events(self):
        return [
            json.loads(line)
            for path in self.evidence.glob("calls-*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_real_provider_receives_same_phi_for_actor_and_extraction_and_saves_experience(self):
        phi_path = bootstrap(self.lineage, "lightweight_memory", self.methods)
        phi_before = phi_path.read_bytes()
        wrapper, native, snapshot, original_config = self._native_provider()
        self.assertFalse(original_config["enable_longterm_provision"])
        self.assertNotIn("philo_orientation_items", original_config)
        self.assertTrue(native.enable_longterm_provision)
        self.assertEqual(native.config["philo_orientation_items"], self.methods)
        self.assertTrue(wrapper.initialize())

        request = MemoryRequest(
            query="Which claim is supported?", context="", status=MemoryStatus.BEGIN,
            additional_params={"caller_value": "preserved"},
        )
        response = wrapper.provide_memory(request)
        self.assertEqual(response.total_count, len(response.memories))
        orientation = [item for item in response.memories if item.metadata.get("kind") == "current_orientation"]
        self.assertEqual(len(orientation), 1)
        self.assertEqual(orientation[0].type, MemoryItemType.TEXT)
        self.assertEqual(orientation[0].metadata["sha256"], snapshot.sha256)
        self.assertTrue(any(item.metadata.get("source") == "lightweight_longterm" for item in response.memories))
        self.assertEqual(request.additional_params, {"caller_value": "preserved"})

        trajectory = self._trajectory()
        success, description = wrapper.take_in_memory(trajectory)
        self.assertTrue(success, description)
        self.assertEqual(trajectory.metadata, {"is_correct": True, "caller_value": "preserved"})
        extraction_prompts = [
            call.args[0] for call in native._call_llm.call_args_list
            if "Extract reusable learnings" in call.args[0]
        ]
        self.assertEqual(len(extraction_prompts), 1)
        for method in self.methods:
            self.assertIn(method, orientation[0].content)
            self.assertIn(method, extraction_prompts[0])
        ordinary = json.loads(Path(native.longterm_memory_path).read_text(encoding="utf-8"))
        self.assertIn(self.learned_strategy, [item["content"] for item in ordinary["strategic"]])
        self.assertIn(self.learned_operation, [item["content"] for item in ordinary["operational"]])
        self.assertEqual(phi_path.read_bytes(), phi_before)
        self.assertEqual(load(self.lineage, "lightweight_memory"), snapshot)
        events = self._events()
        self.assertEqual([event["operation"] for event in events], ["loaded", "initialize", "provide_memory", "take_in_memory"])
        self.assertTrue(all(event["disk_matches_loaded_code"] for event in events))

    def test_empty_fork_stays_absent_from_actor_and_extraction(self):
        bootstrap(self.lineage, "lightweight_memory", self.methods)
        fork(self.lineage, "lightweight_memory", "cleared", [])
        fork(self.lineage, "cleared", "empty_descendant")
        wrapper, native, snapshot, _ = self._native_provider("empty_descendant", enable_longterm_provision=True)
        self.assertEqual(snapshot.items, ())
        before = snapshot.path.read_bytes()
        response = wrapper.provide_memory(MemoryRequest("A different task", "", MemoryStatus.BEGIN))
        self.assertFalse(any(item.metadata.get("kind") == "current_orientation" for item in response.memories))
        self.assertTrue(wrapper.take_in_memory(self._trajectory())[0])
        prompts = [call.args[0] for call in native._call_llm.call_args_list]
        self.assertTrue(any("Extract reusable learnings" in text for text in prompts))
        for text in prompts:
            self.assertNotIn("Current revisable working principles", text)
            for method in self.methods:
                self.assertNotIn(method, text)
        self.assertEqual(snapshot.path.read_bytes(), before)
        self.assertEqual(load(self.lineage, "lightweight_memory").items, tuple(self.methods))

    def test_corrupt_orientation_stops_before_native_operation(self):
        bootstrap(self.lineage, "lightweight_memory", self.methods)
        wrapper, native, snapshot, _ = self._native_provider()
        native.provide_memory = Mock(wraps=native.provide_memory)
        saved = json.loads(snapshot.path.read_text(encoding="utf-8"))
        saved["items"] = ["unauthorized task-time change"]
        snapshot.path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaises(OrientationIntegrityError):
            wrapper.provide_memory(MemoryRequest("query", "", MemoryStatus.BEGIN))
        native.provide_memory.assert_not_called()

    def test_guard_also_detects_orientation_mutation_inside_native_call(self):
        bootstrap(self.lineage, "lightweight_memory", self.methods)
        wrapper, native, snapshot, _ = self._native_provider()

        def mutate_during_extraction(prompt):
            if "Extract reusable learnings" in prompt:
                saved = json.loads(snapshot.path.read_text(encoding="utf-8"))
                saved["items"] = []
                snapshot.path.write_text(json.dumps(saved), encoding="utf-8")
            return self._offline_response(prompt)

        native._call_llm.side_effect = mutate_during_extraction
        with self.assertRaises(OrientationIntegrityError):
            wrapper.take_in_memory(self._trajectory())
        self.assertNotIn("take_in_memory", [event["operation"] for event in self._events()])

    def _fixture(self, provider_id, selection):
        source = textwrap.dedent('''\
            from EvolveLab.memory_types import MemoryItem, MemoryResponse, MemoryType

            class Provider:
                def initialize(self):
                    return True

                def provide_memory(self, request):
                    rows = ["earth observations", "art catalog"]
                    query = request.query.casefold()
                    selected = SELECTION_EXPRESSION
                    items = [MemoryItem(str(i), text, {"source": "fixture"}) for i, text in enumerate(selected)]
                    return MemoryResponse(items, MemoryType.LIGHTWEIGHT_MEMORY, len(items))

                def take_in_memory(self, trajectory):
                    return True, "fixture ingestion"
        ''').replace("SELECTION_EXPRESSION", selection)
        path = self.work / f"{provider_id}.py"
        path.write_text(source, encoding="utf-8")
        loaded_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        module = import_source(path, f"_philo_fixture_{provider_id}")
        wrapper = OrientedProvider(module.Provider(), load(self.lineage, provider_id), path, loaded_hash)
        return wrapper, path, loaded_hash

    def test_distinct_imported_memory_implementations_change_behavior_and_record_loaded_hashes(self):
        bootstrap(self.lineage, "base", [])
        fork(self.lineage, "base", "substring")
        fork(self.lineage, "base", "whole_word")
        first, first_path, first_hash = self._fixture("substring", "[row for row in rows if query in row.casefold()]")
        second, second_path, second_hash = self._fixture("whole_word", "[row for row in rows if query in row.casefold().split()]")
        request = MemoryRequest("art", "", MemoryStatus.BEGIN)
        first_result = first.provide_memory(request)
        second_result = second.provide_memory(request)
        self.assertEqual([item.content for item in first_result.memories], ["earth observations", "art catalog"])
        self.assertEqual([item.content for item in second_result.memories], ["art catalog"])
        self.assertNotEqual(first_hash, second_hash)
        calls = [event for event in self._events() if event["operation"] == "provide_memory"]
        self.assertEqual([event["loaded_code_sha256"] for event in calls], [first_hash, second_hash])
        self.assertEqual([event["code_path"] for event in calls], [str(first_path.resolve()), str(second_path.resolve())])
        self.assertTrue(all(event["disk_matches_loaded_code"] for event in calls))
        self.assertNotEqual(calls[0]["details"]["response_sha256"], calls[1]["details"]["response_sha256"])

    def test_source_change_after_import_is_rejected(self):
        bootstrap(self.lineage, "fixture", [])
        wrapper, path, _ = self._fixture("fixture", "rows[:1]")
        path.write_text(path.read_text(encoding="utf-8") + "\n# changed after loading\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            wrapper.provide_memory(MemoryRequest("art", "", MemoryStatus.BEGIN))
        self.assertNotIn("provide_memory", [event["operation"] for event in self._events()])

    def test_without_environment_configure_preserves_legacy_config(self):
        with patch.dict(os.environ, {}, clear=True):
            original = {"enable_longterm_provision": False, "custom": "retained"}
            configured, snapshot = configure_provider("lightweight_memory", original)
        self.assertIsNone(snapshot)
        self.assertEqual(configured, original)
        self.assertIsNot(configured, original)


if __name__ == "__main__":
    unittest.main()
